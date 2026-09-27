//! Python boundary for the pinned upstream renderer. See UPSTREAM.md for the port contract.
pub mod driver;
mod typography;

use image::{GrayImage, RgbImage, RgbaImage};
use koharu_rasterizer::Rasterizer;
use koharu_renderer::{Renderer, SnapshotFont, TypesettingConfig};
use pyo3::{prelude::*, pybacked::PyBackedBytes, types::PyBytes};
use std::sync::Mutex;

// Page and mask buffers arrive as `PyBackedBytes`: a zero-copy view of the caller's `bytes`.
// A `Vec<u8>` parameter is filled one Python integer per byte with the GIL held — ~50-200 ms
// per page image and ~15-50 ms per balloon mask, during which every other thread of the worker
// stalls. The view is `Send`, so the one copy into an image buffer happens with the GIL released.

struct State {
    runtime: tokio::runtime::Runtime,
    renderer: Renderer,
    rasterizer: Option<Rasterizer>,
    fonts: Vec<SnapshotFont>,
}

#[pyclass]
struct PageRenderer {
    state: Mutex<State>,
}

#[pymethods]
impl PageRenderer {
    #[new]
    #[pyo3(signature = (font_families=None))]
    fn new(font_families: Option<Vec<String>>) -> PyResult<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()?;
        let config = font_families
            .map(|font_families| TypesettingConfig { font_families })
            .unwrap_or_default();
        Ok(Self {
            state: Mutex::new(State {
                runtime,
                renderer: Renderer::with_typesetting(config),
                rasterizer: None,
                fonts: Vec::new(),
            }),
        })
    }

    fn register_font(&self, py: Python<'_>, path: String) -> PyResult<(String, String)> {
        Ok(py.allow_threads(|| -> anyhow::Result<_> {
            let state = self
                .state
                .lock()
                .map_err(|_| anyhow::anyhow!("renderer lock poisoned"))?;
            let bytes = std::fs::read(path)?;
            Ok(state
                .runtime
                .block_on(state.renderer.register_font_bytes(bytes))?)
        })?)
    }

    fn snapshot_fonts(&self, py: Python<'_>) -> PyResult<Vec<(String, u32, Py<PyBytes>)>> {
        let state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("renderer lock poisoned"))?;
        Ok(state
            .fonts
            .iter()
            .map(|font| {
                (
                    font.post_script_name.clone(),
                    font.face_index,
                    PyBytes::new(py, &font.data).unbind(),
                )
            })
            .collect())
    }

    /// Convert one full-page segmentation mask using upstream contour extraction.
    fn bubble_geometry(
        &self,
        py: Python<'_>,
        mask: PyBackedBytes,
        width: u32,
        height: u32,
    ) -> PyResult<String> {
        py.allow_threads(|| {
            let mask = GrayImage::from_raw(width, height, mask.to_vec())
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("mask size mismatch"))?;
            serde_json::to_string(&driver::mask_geometry(&mask).map(|g| g.points))
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
        })
    }

    /// Infer fill, outline, angle and source direction with upstream's pixel formula.
    fn analyze_typography(
        &self,
        py: Python<'_>,
        source_rgb: PyBackedBytes,
        width: u32,
        height: u32,
        boxes: Vec<[f32; 4]>,
        masks: Vec<(u32, u32, u32, u32, PyBackedBytes)>,
    ) -> PyResult<String> {
        Ok(py.allow_threads(|| -> anyhow::Result<String> {
            anyhow::ensure!(boxes.len() == masks.len(), "text boxes and masks differ");
            let image = RgbImage::from_raw(width, height, source_rgb.to_vec())
                .ok_or_else(|| anyhow::anyhow!("RGB size mismatch"))?;
            let mut predictions = Vec::new();
            for (bbox, (x, y, width, height, pixels)) in boxes.into_iter().zip(masks) {
                anyhow::ensure!(
                    pixels.len() as u64 == u64::from(width) * u64::from(height),
                    "text mask size mismatch"
                );
                anyhow::ensure!(bbox.iter().all(|v| v.is_finite()), "invalid text bounds");
                predictions.push(typography::infer_typography(
                    &image,
                    &typography::Detection {
                        bbox,
                        mask: typography::Mask {
                            x,
                            y,
                            width,
                            height,
                            pixels: pixels.to_vec(),
                        },
                    },
                ));
            }
            Ok(serde_json::to_string(&predictions)?)
        })?)
    }

    fn layout_page(
        &self,
        py: Python<'_>,
        image_rgba: PyBackedBytes,
        width: u32,
        height: u32,
        blocks_json: &str,
        options_json: &str,
    ) -> PyResult<String> {
        let blocks: Vec<driver::Block> = serde_json::from_str(blocks_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let options: driver::Options = serde_json::from_str(options_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        if image_rgba.len() as u64 != u64::from(width) * u64::from(height) * 4 {
            return Err(pyo3::exceptions::PyValueError::new_err("RGBA size mismatch"));
        }
        Ok(py.allow_threads(|| -> anyhow::Result<String> {
            let image = RgbaImage::from_raw(width, height, image_rgba.to_vec())
                .ok_or_else(|| anyhow::anyhow!("RGBA size mismatch"))?;
            let state = self
                .state
                .lock()
                .map_err(|_| anyhow::anyhow!("renderer lock poisoned"))?;
            let output = state.runtime.block_on(driver::layout_page(
                &state.renderer,
                image,
                &blocks,
                &options,
            ));
            // Every page is a new scene, so nodes retained for re-rendering it are never reused;
            // kept, they accumulate (up to 2,048 nodes, each page's image among them).
            state.renderer.discard_retained_nodes();
            Ok(serde_json::to_string(&output?)?)
        })?)
    }

    fn render_page(
        &self,
        py: Python<'_>,
        image_rgba: PyBackedBytes,
        width: u32,
        height: u32,
        blocks_json: &str,
        options_json: &str,
    ) -> PyResult<(Py<PyBytes>, String)> {
        let blocks: Vec<driver::Block> = serde_json::from_str(blocks_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let options: driver::Options = serde_json::from_str(options_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        if image_rgba.len() as u64 != u64::from(width) * u64::from(height) * 4 {
            return Err(pyo3::exceptions::PyValueError::new_err("RGBA size mismatch"));
        }
        let output = py.allow_threads(|| -> anyhow::Result<_> {
            let image = RgbaImage::from_raw(width, height, image_rgba.to_vec())
                .ok_or_else(|| anyhow::anyhow!("RGBA size mismatch"))?;
            let mut state = self
                .state
                .lock()
                .map_err(|_| anyhow::anyhow!("renderer lock poisoned"))?;
            if state.rasterizer.is_none() {
                state.rasterizer = Some(Rasterizer::new()?);
            }
            let output = state.runtime.block_on(driver::render_page(
                &state.renderer,
                state.rasterizer.as_ref().unwrap(),
                image,
                &blocks,
                &options,
            ));
            state.renderer.discard_retained_nodes();
            let output = output?;
            state.fonts = output.fonts;
            Ok((output.rgba, serde_json::to_string(&output.blocks)?))
        })?;
        Ok((PyBytes::new(py, &output.0).unbind(), output.1))
    }
}

#[pymodule]
fn shiori_renderer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("UPSTREAM_REVISION", driver::UPSTREAM_REVISION)?;
    m.add_class::<PageRenderer>()?;
    Ok(())
}
