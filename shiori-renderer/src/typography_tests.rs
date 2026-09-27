// Tests ported from the same pinned upstream detection module; only container fields differ.
use super::*;
use image::Rgb;

fn masked_text(
    local_width: f64,
    local_height: f64,
    angle_degrees: f64,
    color: [u8; 3],
) -> (RgbImage, Detection) {
    let width = 96;
    let height = 96;
    let center_x = f64::from(width) * 0.5;
    let center_y = f64::from(height) * 0.5;
    let (sin, cos) = angle_degrees.to_radians().sin_cos();
    let mut image = RgbImage::from_pixel(width, height, Rgb([200, 180, 160]));
    let mut pixels = vec![0; width as usize * height as usize];
    for y in 0..height {
        for x in 0..width {
            let dx = f64::from(x) + 0.5 - center_x;
            let dy = f64::from(y) + 0.5 - center_y;
            let local_x = dx * cos + dy * sin;
            let local_y = -dx * sin + dy * cos;
            if local_x.abs() <= local_width * 0.5 && local_y.abs() <= local_height * 0.5 {
                pixels[y as usize * width as usize + x as usize] = u8::MAX;
                image.put_pixel(x, y, Rgb(color));
            }
        }
    }
    (
        image,
        Detection {
            bbox: [0.0, 0.0, width as f32, height as f32],
            mask: Mask {
                x: 0,
                y: 0,
                width,
                height,
                pixels,
            },
        },
    )
}

fn outlined_text(stroke_width: u32, fill: [u8; 3], stroke: [u8; 3]) -> (RgbImage, Detection) {
    outlined_text_on_background(stroke_width, fill, stroke, [16, 24, 88])
}

fn outlined_text_on_background(
    stroke_width: u32,
    fill: [u8; 3],
    stroke: [u8; 3],
    background: [u8; 3],
) -> (RgbImage, Detection) {
    let width = 96;
    let height = 96;
    let [left, top, right, bottom] = [20, 38, 76, 58];
    let mut image = RgbImage::from_pixel(width, height, Rgb(background));
    let mut pixels = vec![0; width as usize * height as usize];
    for y in top..bottom {
        for x in left..right {
            pixels[y as usize * width as usize + x as usize] = u8::MAX;
            let is_fill = x >= left + stroke_width
                && x < right - stroke_width
                && y >= top + stroke_width
                && y < bottom - stroke_width;
            image.put_pixel(x, y, if is_fill { Rgb(fill) } else { Rgb(stroke) });
        }
    }
    (
        image,
        Detection {
            bbox: [left as f32, top as f32, right as f32, bottom as f32],
            mask: Mask {
                x: 0,
                y: 0,
                width,
                height,
                pixels,
            },
        },
    )
}

fn region_masked_text(fill: [u8; 3], background: [u8; 3]) -> (RgbImage, Detection) {
    let width = 96;
    let height = 96;
    let [left, top, right, bottom] = [20, 20, 76, 76];
    let mut image = RgbImage::from_pixel(width, height, Rgb(background));
    let mut pixels = vec![0; width as usize * height as usize];
    for y in top..bottom {
        for x in left..right {
            pixels[y as usize * width as usize + x as usize] = u8::MAX;
        }
    }
    for x in [27, 35, 43, 51, 59, 67] {
        for y in 28..68 {
            for ink_x in x..x + 3 {
                image.put_pixel(ink_x, y, Rgb(fill));
            }
        }
    }
    (
        image,
        Detection {
            bbox: [left as f32, top as f32, right as f32, bottom as f32],
            mask: Mask {
                x: 0,
                y: 0,
                width,
                height,
                pixels,
            },
        },
    )
}

fn outlined_text_on_textured_background() -> (RgbImage, Detection) {
    let width = 96;
    let height = 96;
    let [left, top, right, bottom] = [20, 20, 76, 76];
    let mut image = RgbImage::from_pixel(width, height, Rgb([24, 32, 96]));
    let mut pixels = vec![0; width as usize * height as usize];
    for y in top..bottom {
        for x in left..right {
            pixels[y as usize * width as usize + x as usize] = u8::MAX;
            let background = match (x / 7 + y / 5) % 3 {
                0 => [0, 0, 0],
                1 => [38, 48, 128],
                _ => [24, 32, 96],
            };
            image.put_pixel(x, y, Rgb(background));
        }
    }
    for glyph_left in [26, 38, 50, 62] {
        for y in 28..68 {
            for x in glyph_left..glyph_left + 9 {
                let is_fill = x >= glyph_left + 3 && x < glyph_left + 6 && (31..65).contains(&y);
                image.put_pixel(
                    x,
                    y,
                    if is_fill {
                        Rgb([0, 0, 0])
                    } else {
                        Rgb([255, 255, 255])
                    },
                );
            }
        }
    }
    (
        image,
        Detection {
            bbox: [left as f32, top as f32, right as f32, bottom as f32],
            mask: Mask {
                x: 0,
                y: 0,
                width,
                height,
                pixels,
            },
        },
    )
}

#[test]
fn typography_comes_from_horizontal_text_mask() {
    let (image, detection) = masked_text(52.0, 12.0, 12.0, [24, 80, 160]);

    let inferred = infer_typography(&image, &detection).unwrap();

    assert!((inferred.angle_degrees - 12.0).abs() < 1.0);
    assert_eq!(inferred.color, [24, 80, 160]);
    assert_eq!(inferred.stroke_color, None);
    assert_eq!(inferred.stroke_width, None);
    assert_eq!(inferred.writing_mode, WritingMode::Horizontal);
}

#[test]
fn vertical_text_angle_is_relative_to_upright_vertical() {
    let (image, detection) = masked_text(12.0, 52.0, 9.0, [120, 80, 40]);

    let inferred = infer_typography(&image, &detection).unwrap();

    assert!((inferred.angle_degrees - 9.0).abs() < 1.0);
    assert_eq!(inferred.writing_mode, WritingMode::Vertical);
}

#[test]
fn tall_multiline_mask_uses_text_lines_instead_of_block_aspect() {
    let width = 96;
    let height = 96;
    let angle_degrees = 8.0_f64;
    let (sin, cos) = angle_degrees.to_radians().sin_cos();
    let mut image = RgbImage::from_pixel(width, height, Rgb([240, 240, 240]));
    let mut pixels = vec![0; width as usize * height as usize];
    for y in 0..height {
        for x in 0..width {
            let dx = f64::from(x) + 0.5 - f64::from(width) * 0.5;
            let dy = f64::from(y) + 0.5 - f64::from(height) * 0.5;
            let local_x = dx * cos + dy * sin;
            let local_y = -dx * sin + dy * cos;
            let inside = [-24.0, 0.0, 24.0]
                .into_iter()
                .any(|line_y| local_x.abs() <= 15.0 && (local_y - line_y).abs() <= 2.0);
            if inside {
                pixels[y as usize * width as usize + x as usize] = u8::MAX;
                image.put_pixel(x, y, Rgb([8, 8, 8]));
            }
        }
    }
    let detection = Detection {
        bbox: [0.0, 0.0, width as f32, height as f32],
        mask: Mask {
            x: 0,
            y: 0,
            width,
            height,
            pixels,
        },
    };

    let inferred = infer_typography(&image, &detection).unwrap();

    assert_eq!(inferred.writing_mode, WritingMode::Horizontal);
    assert!((inferred.angle_degrees - 8.0).abs() < 1.0);
}

#[test]
fn antialiased_dark_text_uses_the_high_contrast_core() {
    let (mut image, detection) = masked_text(52.0, 12.0, 0.0, [96, 94, 92]);
    for (index, &mask) in detection.mask.pixels.iter().enumerate() {
        if mask != 0 && index.is_multiple_of(3) {
            let x = index as u32 % image.width();
            let y = index as u32 / image.width();
            image.put_pixel(x, y, Rgb([8, 9, 7]));
        }
    }

    let inferred = infer_typography(&image, &detection).unwrap();

    assert_eq!(inferred.color, [0, 0, 0]);
}

#[test]
fn text_colors_snap_only_at_luminance_extremes() {
    assert_eq!(normalize_text_color([20, 31, 24]), [0, 0, 0]);
    assert_eq!(normalize_text_color([230, 240, 250]), [255, 255, 255]);
    assert_eq!(normalize_text_color([255, 0, 0]), [0, 0, 0]);
    assert_eq!(normalize_text_color([0, 255, 0]), [0, 255, 0]);
    assert_eq!(normalize_text_color([20, 115, 235]), [20, 115, 235]);
    assert_eq!(normalize_text_color([40, 70, 40]), [0, 0, 0]);
}

#[test]
fn outlined_text_uses_the_deep_fill_and_measures_the_border() {
    for stroke_width in [2, 3, 5] {
        let (image, detection) = outlined_text(stroke_width, [0, 0, 0], [255, 255, 255]);

        let inferred = infer_typography(&image, &detection).unwrap();

        assert_eq!(inferred.color, [0, 0, 0]);
        assert_eq!(inferred.stroke_color, Some([255, 255, 255]));
        assert_eq!(inferred.stroke_width, Some(stroke_width as f32));
    }
}

#[test]
fn outlined_text_roles_do_not_flip_with_inverse_luminance() {
    let (image, detection) = outlined_text(3, [255, 255, 255], [0, 0, 0]);

    let inferred = infer_typography(&image, &detection).unwrap();

    assert_eq!(inferred.color, [255, 255, 255]);
    assert_eq!(inferred.stroke_color, Some([0, 0, 0]));
    assert_eq!(inferred.stroke_width, Some(3.0));
}

#[test]
fn glyph_holes_matching_the_background_do_not_invert_the_paint() {
    let (image, detection) =
        outlined_text_on_background(3, [255, 255, 255], [0, 0, 0], [255, 255, 255]);

    let inferred = infer_typography(&image, &detection).unwrap();

    assert_eq!(inferred.color, [0, 0, 0]);
    assert_eq!(inferred.stroke_color, None);
    assert_eq!(inferred.stroke_width, None);
}

#[test]
fn outlined_text_is_separated_from_matching_dark_artwork() {
    let (image, detection) = outlined_text_on_textured_background();

    let inferred = infer_typography(&image, &detection).unwrap();

    assert_eq!(inferred.color, [0, 0, 0]);
    assert_eq!(inferred.stroke_color, Some([255, 255, 255]));
    assert_eq!(inferred.stroke_width, Some(3.0));
}

#[test]
fn colors_in_the_same_luminance_class_do_not_create_a_border() {
    let (image, detection) = outlined_text(3, [0, 0, 0], [48, 48, 48]);

    let inferred = infer_typography(&image, &detection).unwrap();

    assert_eq!(inferred.color, [0, 0, 0]);
    assert_eq!(inferred.stroke_color, None);
    assert_eq!(inferred.stroke_width, None);
}

#[test]
fn wide_antialias_bands_do_not_create_borders() {
    for (fill, antialias, background) in [
        ([20, 115, 235], [133, 180, 240], [245, 245, 245]),
        ([0, 0, 0], [96, 96, 96], [245, 245, 245]),
    ] {
        let (image, detection) = outlined_text_on_background(4, fill, antialias, background);

        let inferred = infer_typography(&image, &detection).unwrap();

        assert_eq!(inferred.color, fill);
        assert_eq!(inferred.stroke_color, None);
        assert_eq!(inferred.stroke_width, None);
    }
}

#[test]
fn text_region_background_is_not_mistaken_for_the_fill() {
    for (fill, background, expected) in [
        ([24, 40, 80], [225, 130, 175], [0, 0, 0]),
        ([220, 240, 255], [16, 24, 88], [255, 255, 255]),
        ([20, 115, 235], [245, 245, 245], [20, 115, 235]),
    ] {
        let (image, detection) = region_masked_text(fill, background);

        let inferred = infer_typography(&image, &detection).unwrap();

        assert_eq!(inferred.color, expected);
        assert_eq!(inferred.stroke_color, None);
        assert_eq!(inferred.stroke_width, None);
    }
}

#[test]
fn color_palette_is_bounded_independently_of_crop_area() {
    let pixels = (0..65_536_u32)
        .map(|index| MaskPixel {
            x: 0,
            y: 0,
            color: [
                index as u8,
                (index >> 8) as u8,
                index.wrapping_mul(31) as u8,
            ],
            inside_mask: true,
        })
        .collect::<Vec<_>>();

    assert!(color_palette(&pixels).len() <= 16 * 16 * 16);
}
