//! Pipeline-to-scene adapter. Fitting, shaping and rasterization remain upstream.
use anyhow::{Context, Result, ensure};
use image::{DynamicImage, GrayImage, RgbaImage};
use imageproc::{
    contours::{BorderType, find_contours_with_threshold},
    geometry::{approximate_polygon_dp, arc_length, contour_area},
};
use koharu_rasterizer::{RasterOptions, Rasterizer};
use koharu_renderer::{Frame, LayerKind, Renderer, SnapshotFont, SnapshotLine, WritingMode};
use koharu_scene::{
    AssetInput, AssetMetadata, AssetRole, At, Authored, EntityId, FitsTo, FlowsIn, FontStyle,
    Geometry, LanguageTag, OcrAnalysis, Origin, PageDraft, Point, RecognizedFrom, Region,
    RegionKind, Session, Snapshot, SourceText, TextAlignment, TextDirection, TextLayout,
    TextLayoutKind, Translation, Typography,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    io::Cursor,
    sync::Arc,
};

pub const UPSTREAM_REVISION: &str = "4a133539f204ab1182ff64901ba5226b4e868fb0";

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Transform {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    #[serde(default)]
    pub rotation_deg: f64,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Block {
    pub node_id: u64,
    pub transform: Transform,
    pub translation: String,
    #[serde(default)]
    pub source_text: String,
    #[serde(default)]
    pub points: Option<Vec<Point>>,
    #[serde(default)]
    pub bubble_id: Option<u32>,
    #[serde(default)]
    pub source_direction: Option<String>,
    #[serde(default)]
    pub writing_mode: Option<String>,
    #[serde(default)]
    pub font: Option<String>,
    #[serde(default)]
    pub font_size: Option<f32>,
    #[serde(default)]
    pub font_weight: Option<u16>,
    #[serde(default)]
    pub font_style: Option<FontStyle>,
    #[serde(default)]
    pub alignment: Option<TextAlignment>,
    #[serde(default)]
    pub color: Option<[u8; 4]>,
    #[serde(default)]
    pub stroke_color: Option<[u8; 4]>,
    #[serde(default)]
    pub stroke_width: Option<f32>,
    #[serde(default)]
    pub point_text: bool,
}

#[derive(Clone, Deserialize)]
pub struct Bubble {
    pub id: u32,
    pub points: Vec<Point>,
}

#[derive(Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Options {
    #[serde(default)]
    pub document_font: Option<String>,
    #[serde(default)]
    pub target_language: Option<String>,
    #[serde(default)]
    pub bubbles: Vec<Bubble>,
    #[serde(default)]
    pub supersampling: Option<u32>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BlockOutput {
    pub node_id: u64,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub rotation_deg: f32,
    pub font_size: f32,
    pub rendered_direction: &'static str,
    pub text_color: [u8; 3],
    pub stroke_color: [u8; 3],
    pub stroke_width: f32,
    pub lines: Vec<SnapshotLine>,
    pub geometry: Vec<Point>,
    pub diagnostics: Vec<String>,
}

pub struct PageOutput {
    pub rgba: Vec<u8>,
    pub blocks: Vec<BlockOutput>,
    pub fonts: Vec<SnapshotFont>,
}

/// Identical contour extraction to koharu-pipeline detection::mask_geometry.
pub fn mask_geometry(mask: &GrayImage) -> Option<Geometry> {
    let mut padded = GrayImage::new(mask.width() + 2, mask.height() + 2);
    image::imageops::replace(&mut padded, mask, 1, 1);
    let contours = find_contours_with_threshold::<i32>(&padded, 0);
    let contour = contours
        .iter()
        .filter(|contour| contour.border_type == BorderType::Outer)
        .max_by(|left, right| {
            contour_area(&left.points)
                .partial_cmp(&contour_area(&right.points))
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
    if contour.points.len() < 3 {
        return None;
    }
    let epsilon = (arc_length(&contour.points, true) * 0.001).max(f64::EPSILON);
    let points = approximate_polygon_dp(&contour.points, epsilon, true)
        .into_iter()
        .map(|point| Point {
            x: f64::from(point.x - 1),
            y: f64::from(point.y - 1),
        })
        .collect::<Vec<_>>();
    (points.len() >= 3).then_some(Geometry {
        origin: Origin::User,
        points,
    })
}

fn region_geometry(block: &Block) -> Geometry {
    if let Some(points) = &block.points {
        return Geometry {
            origin: Origin::User,
            points: points.clone(),
        };
    }
    let t = &block.transform;
    let mut geometry = Geometry::rectangle(t.x, t.y, t.width, t.height);
    let (sin, cos) = t.rotation_deg.to_radians().sin_cos();
    let (cx, cy) = (t.x + t.width * 0.5, t.y + t.height * 0.5);
    for point in &mut geometry.points {
        let (x, y) = (point.x - cx, point.y - cy);
        point.x = cx + x * cos - y * sin;
        point.y = cy + x * sin + y * cos;
    }
    geometry
}

fn direction(value: Option<&str>) -> Result<TextDirection> {
    Ok(match value {
        None | Some("auto") => TextDirection::Auto,
        Some("horizontal") => TextDirection::Horizontal,
        Some("vertical") => TextDirection::Vertical,
        Some(other) => anyhow::bail!("invalid source direction: {other}"),
    })
}

fn invalid(message: impl Into<String>) -> koharu_scene::Error {
    koharu_scene::Error::Invalid(message.into())
}

/// Sibling texts share a FlowsIn target and retain their individual source anchors.
pub async fn page_scene(
    image: RgbaImage,
    blocks: &[Block],
    options: &Options,
) -> Result<(Snapshot, EntityId, Vec<(u64, EntityId)>)> {
    let (width, height) = image.dimensions();
    ensure!(width > 0 && height > 0, "page dimensions must be positive");
    // The page reaches the scene as an encoded blob that the renderer decodes again. BMP is
    // lossless like PNG (the pixels arrive unchanged) but costs a copy instead of a full
    // deflate: PNG spent the better part of a second per 6 MP page encoding it.
    let mut bytes = Cursor::new(Vec::with_capacity(width as usize * height as usize * 4 + 256));
    DynamicImage::ImageRgba8(image).write_to(&mut bytes, image::ImageFormat::Bmp)?;
    let bytes: Arc<[u8]> = bytes.into_inner().into();
    let language = options
        .target_language
        .as_ref()
        .map(LanguageTag::new)
        .transpose()?;
    let mut session = Session::memory().await?;
    let mut page = None;
    let mut layers = Vec::new();
    let patch = session.snapshot().patch(|edit| {
        let created = edit.add_page(
            PageDraft::new("render", f64::from(width), f64::from(height)),
            At::End,
        )?;
        page = Some(created);
        edit.set_asset(
            created,
            &AssetRole::new("source")?,
            AssetInput::new(
                bytes.clone(),
                "image/png",
                AssetMetadata {
                    width: Some(width),
                    height: Some(height),
                    attributes: BTreeMap::new(),
                },
            ),
        )?;
        let mut bubbles = HashMap::new();
        for bubble in &options.bubbles {
            if bubbles.contains_key(&bubble.id) {
                return Err(invalid("duplicate bubble ID"));
            }
            let entity = edit.add_entity(created, At::End)?;
            edit.set(
                entity,
                &Geometry {
                    origin: Origin::User,
                    points: bubble.points.clone(),
                },
            )?;
            edit.set(
                entity,
                &Region {
                    origin: Origin::User,
                    kind: RegionKind::new("dev.koharu.region.bubble")?,
                    label: None,
                },
            )?;
            bubbles.insert(bubble.id, entity);
        }
        let mut node_ids = HashSet::new();
        for block in blocks {
            if !node_ids.insert(block.node_id) {
                return Err(invalid("duplicate node ID"));
            }
            let source = edit.add_entity(created, At::End)?;
            edit.set(source, &region_geometry(block))?;
            edit.set(
                source,
                &Region {
                    origin: Origin::User,
                    kind: RegionKind::new("dev.koharu.region.text")?,
                    label: None,
                },
            )?;
            let source_direction =
                direction(block.source_direction.as_deref()).map_err(|e| invalid(e.to_string()))?;
            edit.set(
                source,
                &OcrAnalysis {
                    origin: Origin::User,
                    direction: source_direction,
                    confidence: None,
                    line_boundaries: Vec::new(),
                },
            )?;
            let content = edit.add_text_content(created, At::End)?;
            edit.set(
                content,
                &SourceText {
                    text: Authored::user(block.source_text.clone()),
                    language: None,
                },
            )?;
            edit.set(
                content,
                &Translation {
                    text: Authored::user(block.translation.clone()),
                    language: language.clone(),
                },
            )?;
            edit.relate::<RecognizedFrom>(content, source)?;
            let layer = edit.add_text_layer(
                created,
                At::End,
                content,
                &TextLayout {
                    origin: Origin::User,
                    kind: if block.point_text {
                        TextLayoutKind::Point
                    } else {
                        TextLayoutKind::Paragraph
                    },
                    angle_degrees: None,
                },
            )?;
            let writing_mode = match block.writing_mode.as_deref() {
                None => None,
                Some("horizontal") => Some(koharu_scene::WritingMode::Horizontal),
                Some("vertical") => Some(koharu_scene::WritingMode::Vertical),
                Some(_) => return Err(invalid("invalid writing mode")),
            };
            edit.set(
                layer,
                &Typography {
                    origin: Origin::User,
                    preferred_font: block.font.clone().or_else(|| options.document_font.clone()),
                    font_weight: block.font_weight,
                    font_style: block.font_style,
                    size: block.font_size,
                    auto_fit: block.font_size.is_none(),
                    color: block.color,
                    stroke_color: block.stroke_color,
                    stroke_width: block.stroke_width,
                    alignment: block.alignment,
                    writing_mode,
                    extensions: BTreeMap::new(),
                },
            )?;
            if let Some(id) = block.bubble_id {
                let bubble = bubbles
                    .get(&id)
                    .ok_or_else(|| invalid("unknown bubble ID"))?;
                edit.relate::<FlowsIn>(layer, *bubble)?;
            } else {
                edit.relate::<FitsTo>(layer, source)?;
            }
            layers.push((block.node_id, layer));
        }
        Ok(())
    })?;
    let snapshot = session.commit(patch).await?.snapshot;
    Ok((snapshot, page.context("page creation failed")?, layers))
}

/// Resolve upstream layout without rasterizing a page. Paint can then depend on
/// the fitted size without replacing automatic fitting with a fixed-size layout.
pub async fn layout_page(
    renderer: &Renderer,
    image: RgbaImage,
    blocks: &[Block],
    options: &Options,
) -> Result<Vec<BlockOutput>> {
    let (snapshot, page, layers) = page_scene(image, blocks, options).await?;
    let frame = renderer.render(&snapshot, page).await?;
    Ok(page_metadata(&frame, &layers, blocks).0)
}

pub async fn render_page(
    renderer: &Renderer,
    rasterizer: &Rasterizer,
    image: RgbaImage,
    blocks: &[Block],
    options: &Options,
) -> Result<PageOutput> {
    let (snapshot, page, layers) = page_scene(image, blocks, options).await?;
    let frame = renderer.render(&snapshot, page).await?;
    let raster = rasterizer.rasterize(
        &frame.raster_frame()?,
        RasterOptions::supersampled(options.supersampling.unwrap_or(1)),
    )?;
    let (blocks, fonts) = page_metadata(&frame, &layers, blocks);
    Ok(PageOutput {
        rgba: raster.image.into_raw(),
        blocks,
        fonts,
    })
}

fn page_metadata(
    frame: &Frame,
    layers: &[(u64, EntityId)],
    blocks: &[Block],
) -> (Vec<BlockOutput>, Vec<SnapshotFont>) {
    let mut output = Vec::new();
    let mut fonts = BTreeMap::new();
    for (&(id, entity), input) in layers.iter().zip(blocks) {
        let Some(layer) = frame.layer(entity) else {
            continue;
        };
        let LayerKind::Text(meta) = layer.kind() else {
            continue;
        };
        for font in &meta.snapshot.fonts {
            fonts.insert(
                (font.post_script_name.clone(), font.face_index),
                font.clone(),
            );
        }
        let bounds = layer.bounds();
        let stroke = input.stroke_width.unwrap_or(0.0);
        let stroke_color = if stroke > 0.0 {
            input.stroke_color.unwrap_or([255; 4])
        } else {
            meta.color
        };
        output.push(BlockOutput {
            node_id: id,
            x: bounds.x,
            y: bounds.y,
            width: bounds.width,
            height: bounds.height,
            rotation_deg: meta.angle_degrees,
            font_size: meta.font_size,
            rendered_direction: if meta.writing_mode == WritingMode::Horizontal {
                "horizontal"
            } else {
                "vertical"
            },
            text_color: [meta.color[0], meta.color[1], meta.color[2]],
            stroke_color: [stroke_color[0], stroke_color[1], stroke_color[2]],
            stroke_width: stroke,
            lines: meta.snapshot.lines.clone(),
            geometry: layer.geometry().points.clone(),
            diagnostics: frame
                .diagnostics()
                .iter()
                .map(|d| format!("{d:?}"))
                .collect(),
        });
    }
    (output, fonts.into_values().collect())
}
