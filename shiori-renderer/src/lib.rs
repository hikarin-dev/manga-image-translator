//! Python bindings for the vendored koharu renderer.
//!
//! Exposes a single `PageRenderer` class whose `render_page` mirrors
//! `koharu-app`'s pipeline call: inpainted page + optional bubble-ID mask +
//! per-block inputs (translation, box, optional font prediction) → final
//! composited page + per-block placement info.

mod core_types;
mod driver;

use image::{DynamicImage, GrayImage, RgbaImage};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::{Deserialize, Serialize};

use core_types::{FontPrediction, TextDirection, TextShaderEffect, TextStrokeStyle, TextStyle, Transform};
use koharu_renderer::renderer::RasterOptions;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct BlockIn {
    node_id: u64,
    transform: Transform,
    translation: String,
    #[serde(default)]
    style: Option<TextStyle>,
    #[serde(default)]
    font_prediction: Option<FontPrediction>,
    #[serde(default)]
    source_direction: Option<TextDirection>,
    #[serde(default)]
    rendered_direction: Option<TextDirection>,
    #[serde(default)]
    lock_layout_box: bool,
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct OptionsIn {
    #[serde(default)]
    shader_effect: TextShaderEffect,
    #[serde(default)]
    shader_stroke: Option<TextStrokeStyle>,
    #[serde(default)]
    document_font: Option<String>,
    #[serde(default)]
    target_language: Option<String>,
    /// Optional supersampling factor; omitted → koharu's `RasterOptions::default()`.
    #[serde(default)]
    supersampling: Option<u32>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BlockOut {
    node_id: u64,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    rotation_deg: f32,
    font_size: f32,
    rendered_direction: TextDirection,
    /// Colors the block was actually drawn with; strokeColor == textColor and
    /// strokeWidth == 0 when no stroke was drawn.
    text_color: [u8; 3],
    stroke_color: [u8; 3],
    stroke_width: f32,
}

#[pyclass]
struct PageRenderer {
    inner: driver::Renderer,
}

#[pymethods]
impl PageRenderer {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self {
            inner: driver::Renderer::new()?,
        })
    }

    /// Register a font file (ttf/otf). Returns `(family_name, post_script_name)`;
    /// pass either as `documentFont` in the render options.
    fn register_font(&self, path: &str) -> PyResult<(String, String)> {
        Ok(self.inner.register_font_file(path)?)
    }

    /// Render a full page.
    ///
    /// - `image_rgba`: raw RGBA bytes of the inpainted page, `width * height * 4`.
    /// - `blocks_json` / `options_json`: camelCase JSON (see `BlockIn` / `OptionsIn`).
    /// - `bubble_mask`: optional raw grayscale bytes (same page dimensions) where
    ///   each bubble interior carries a distinct non-zero ID and background is 0.
    ///
    /// Returns `(final_rgba_bytes, blocks_info_json)`.
    #[pyo3(signature = (image_rgba, width, height, blocks_json, options_json, bubble_mask=None))]
    fn render_page(
        &self,
        py: Python<'_>,
        image_rgba: Vec<u8>,
        width: u32,
        height: u32,
        blocks_json: &str,
        options_json: &str,
        bubble_mask: Option<Vec<u8>>,
    ) -> PyResult<(Py<PyBytes>, String)> {
        let blocks_in: Vec<BlockIn> = serde_json::from_str(blocks_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("blocks_json: {e}")))?;
        let opts_in: OptionsIn = serde_json::from_str(options_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("options_json: {e}")))?;

        let base = RgbaImage::from_raw(width, height, image_rgba).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("image_rgba does not match width*height*4")
        })?;
        let base = DynamicImage::ImageRgba8(base);

        let mask_img = match bubble_mask {
            Some(data) => Some(DynamicImage::ImageLuma8(
                GrayImage::from_raw(width, height, data).ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "bubble_mask does not match width*height",
                    )
                })?,
            )),
            None => None,
        };

        let blocks: Vec<driver::RenderBlockInput> = blocks_in
            .into_iter()
            .map(|b| driver::RenderBlockInput {
                node_id: b.node_id,
                transform: b.transform,
                translation: b.translation,
                style: b.style,
                font_prediction: b.font_prediction,
                source_direction: b.source_direction,
                rendered_direction: b.rendered_direction,
                lock_layout_box: b.lock_layout_box,
            })
            .collect();

        let raster = match opts_in.supersampling {
            Some(factor) => RasterOptions::supersampled(factor),
            None => RasterOptions::default(),
        };
        let page_opts = driver::PageRenderOptions {
            shader_effect: opts_in.shader_effect,
            shader_stroke: opts_in.shader_stroke,
            document_font: opts_in.document_font,
            target_language: opts_in.target_language,
            raster,
        };

        let inner = &self.inner;
        let output = py.allow_threads(move || {
            inner.render_page(
                &base,
                None,
                mask_img.as_ref(),
                width,
                height,
                &blocks,
                &page_opts,
            )
        })?;

        let infos: Vec<BlockOut> = output
            .blocks
            .iter()
            .map(|b| {
                let t = b.expanded_transform.unwrap_or_default();
                BlockOut {
                    node_id: b.node_id,
                    x: t.x,
                    y: t.y,
                    width: t.width,
                    height: t.height,
                    rotation_deg: t.rotation_deg,
                    font_size: b.font_size,
                    rendered_direction: b.rendered_direction,
                    text_color: [b.text_color[0], b.text_color[1], b.text_color[2]],
                    stroke_color: [b.stroke_color[0], b.stroke_color[1], b.stroke_color[2]],
                    stroke_width: b.stroke_width,
                }
            })
            .collect();
        let info_json = serde_json::to_string(&infos)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("info_json: {e}")))?;

        let raw = output.final_render.to_rgba8().into_raw();
        Ok((PyBytes::new(py, &raw).unbind(), info_json))
    }
}

#[pymodule]
fn shiori_renderer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PageRenderer>()?;
    Ok(())
}
