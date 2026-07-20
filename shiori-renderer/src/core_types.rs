//! Minimal inlined copies of the `koharu-core` types the render driver uses.
//!
//! Field sets, defaults, and semantics match `koharu-core` (`scene.rs::Transform`,
//! `style.rs::{TextAlign, TextShaderEffect, TextStrokeStyle, TextStyle}`,
//! `font.rs::{FontPrediction, TextDirection}`) — only the scene/app machinery
//! (schemars/utoipa schemas, `top_fonts`/`named_fonts` classification lists) is
//! dropped, none of which the driver reads.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Transform {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    #[serde(default)]
    pub rotation_deg: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TextDirection {
    Horizontal,
    Vertical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum TextAlign {
    #[default]
    Left,
    Center,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct TextShaderEffect {
    #[serde(default)]
    pub italic: bool,
    #[serde(default)]
    pub bold: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextStrokeStyle {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_stroke_color")]
    pub color: [u8; 4],
    #[serde(default)]
    pub width_px: Option<f32>,
}

impl Default for TextStrokeStyle {
    fn default() -> Self {
        Self {
            enabled: true,
            color: [255, 255, 255, 255],
            width_px: None,
        }
    }
}

const fn default_true() -> bool {
    true
}

const fn default_stroke_color() -> [u8; 4] {
    [255, 255, 255, 255]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextStyle {
    #[serde(default)]
    pub font_families: Vec<String>,
    #[serde(default)]
    pub font_size: Option<f32>,
    #[serde(default = "default_text_color")]
    pub color: [u8; 4],
    #[serde(default)]
    pub effect: Option<TextShaderEffect>,
    #[serde(default)]
    pub stroke: Option<TextStrokeStyle>,
    #[serde(default)]
    pub text_align: Option<TextAlign>,
}

const fn default_text_color() -> [u8; 4] {
    [0, 0, 0, 255]
}

impl Default for TextStyle {
    fn default() -> Self {
        Self {
            font_families: Vec::new(),
            font_size: None,
            color: [0, 0, 0, 255],
            effect: None,
            stroke: None,
            text_align: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FontPrediction {
    #[serde(default = "default_direction")]
    pub direction: TextDirection,
    #[serde(default)]
    pub text_color: [u8; 3],
    #[serde(default)]
    pub stroke_color: [u8; 3],
    #[serde(default)]
    pub font_size_px: f32,
    #[serde(default)]
    pub stroke_width_px: f32,
    #[serde(default = "default_line_height")]
    pub line_height: f32,
    #[serde(default)]
    pub angle_deg: f32,
}

const fn default_direction() -> TextDirection {
    TextDirection::Horizontal
}

const fn default_line_height() -> f32 {
    1.0
}

impl Default for FontPrediction {
    fn default() -> Self {
        Self {
            direction: TextDirection::Horizontal,
            text_color: [0, 0, 0],
            stroke_color: [0, 0, 0],
            font_size_px: 0.0,
            stroke_width_px: 0.0,
            line_height: 1.0,
            angle_deg: 0.0,
        }
    }
}
