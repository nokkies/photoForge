//! Quantitative image telemetry and coordinate-annotated previews for external AI agents.

use std::collections::HashMap;

use photocraft_codecs::{ChannelLayout, Format, Image};
use photocraft_doc::Document;
use serde::Serialize;
use serde_json::Value;

use crate::AutomationError;

const DEFAULT_ANALYZE_MAX_SIDE: u32 = 512;
const DEFAULT_ANNOTATE_MAX_SIDE: u32 = 1024;
const MIN_SIDE: u32 = 32;
const MAX_SIDE: u32 = 2048;

/// Quantitative pixel telemetry for a document or sub-region.
#[derive(Debug, Clone, Serialize)]
pub struct DocTelemetry {
    pub document: DocumentMeta,
    pub analyzed_region: AnalyzedRegionMeta,
    pub luminance: LuminanceStats,
    pub exposure: ExposureStats,
    pub channels: ChannelStats,
    pub histogram_16_bins: Vec<f32>,
    pub spatial_grid_3x3: Vec<GridCellStats>,
    pub dominant_colors: Vec<DominantColor>,
    pub edge_sharpness: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct DocumentMeta {
    pub width: u32,
    pub height: u32,
    pub mode: String,
    pub depth_bits: u32,
    pub layer_count: usize,
    pub has_selection: bool,
    pub selection_bounds: Option<[i32; 4]>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnalyzedRegionMeta {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
    pub sampled_pixels: usize,
    pub preview_width: u32,
    pub preview_height: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct LuminanceStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std_dev: f32,
    pub p05: f32,
    pub p50: f32,
    pub p95: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExposureStats {
    pub shadow_clip_pct: f32,
    pub highlight_clip_pct: f32,
    pub midtone_pct: f32,
    pub transparent_pct: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChannelStats {
    pub r_mean: f32,
    pub g_mean: f32,
    pub b_mean: f32,
    pub a_mean: f32,
    pub mean_saturation: f32,
    pub vivid_pct: f32,
    pub warm_cool_bias: f32,
    pub tint_bias: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct GridCellStats {
    pub cell: &'static str,
    pub mean_luma: f32,
    pub r_mean: f32,
    pub g_mean: f32,
    pub b_mean: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct DominantColor {
    pub hex: String,
    pub pct: f32,
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

/// Options controlling coordinate-annotated preview rendering.
#[derive(Debug, Clone)]
pub struct AnnotateOptions {
    pub max_side: u32,
    pub grid_step: Option<u32>,
    pub show_grid: bool,
    pub show_axes: bool,
    pub show_selection: bool,
    pub highlight_box: Option<[i32; 4]>,
}

impl Default for AnnotateOptions {
    fn default() -> Self {
        Self {
            max_side: DEFAULT_ANNOTATE_MAX_SIDE,
            grid_step: None,
            show_grid: true,
            show_axes: true,
            show_selection: true,
            highlight_box: None,
        }
    }
}

/// Metadata accompanying an annotated PNG preview.
#[derive(Debug, Clone, Serialize)]
pub struct AnnotatedPreviewMeta {
    pub doc_width: u32,
    pub doc_height: u32,
    pub image_width: u32,
    pub image_height: u32,
    pub grid_step_px: Option<u32>,
    pub selection_bounds: Option<[i32; 4]>,
    pub highlight_box: Option<[i32; 4]>,
}

/// Analyze a flattened document composite and return quantitative JSON telemetry.
pub fn analyze_document(doc: &Document, max_side: Option<u32>, region: Option<[i32; 4]>) -> Result<Value, AutomationError> {
    if doc.size.width == 0 || doc.size.height == 0 {
        return Err(AutomationError::BadRequest("cannot analyze a zero-sized document".into()));
    }
    let side = max_side.unwrap_or(DEFAULT_ANALYZE_MAX_SIDE).clamp(MIN_SIDE, MAX_SIDE);
    let thumb = photocraft_compose::thumbnail(doc, side);
    let selection_bounds = doc_selection_bounds(doc);
    let doc_meta = DocumentMeta {
        width: doc.size.width,
        height: doc.size.height,
        mode: format!("{:?}", doc.mode),
        depth_bits: doc.depth.bits(),
        layer_count: doc.layers.len(),
        has_selection: selection_bounds.is_some(),
        selection_bounds,
    };
    let doc_region = resolve_doc_region(doc.size.width, doc.size.height, region)?;
    let telemetry = analyze_rgba_bytes(
        &thumb.pixels,
        thumb.width,
        thumb.height,
        doc_meta,
        doc_region,
    )?;
    serde_json::to_value(telemetry).map_err(|e| AutomationError::Other(format!("failed to serialize telemetry: {e}")))
}

/// Extract selection bounding box `[x, y, width, height]` if a selection exists and is non-empty.
pub fn doc_selection_bounds(doc: &Document) -> Option<[i32; 4]> {
    let sel = doc.selection.as_ref()?;
    let r = sel.content_bounds();
    if r.is_empty() {
        return None;
    }
    let w = i32::try_from(r.width()).ok()?;
    let h = i32::try_from(r.height()).ok()?;
    Some([r.x0, r.y0, w, h])
}

fn resolve_doc_region(doc_w: u32, doc_h: u32, region: Option<[i32; 4]>) -> Result<[i32; 4], AutomationError> {
    let Some([x, y, w, h]) = region else {
        let dw = i32::try_from(doc_w).map_err(|_| AutomationError::BadRequest("document width exceeds i32::MAX".into()))?;
        let dh = i32::try_from(doc_h).map_err(|_| AutomationError::BadRequest("document height exceeds i32::MAX".into()))?;
        return Ok([0, 0, dw, dh]);
    };
    if w <= 0 || h <= 0 {
        return Err(AutomationError::BadRequest(format!("region width and height must be positive, got [{x}, {y}, {w}, {h}]")));
    }
    let x0 = x.max(0);
    let y0 = y.max(0);
    let x1 = x.saturating_add(w).min(i32::try_from(doc_w).unwrap_or(i32::MAX));
    let y1 = y.saturating_add(h).min(i32::try_from(doc_h).unwrap_or(i32::MAX));
    if x1 <= x0 || y1 <= y0 {
        return Err(AutomationError::BadRequest(format!("region [{x}, {y}, {w}, {h}] lies outside document {doc_w}x{doc_h}")));
    }
    Ok([x0, y0, x1 - x0, y1 - y0])
}

/// Compute quantitative telemetry over raw straight RGBA8 pixels for a document region.
pub fn analyze_rgba_bytes(
    pixels: &[u8],
    img_w: u32,
    img_h: u32,
    doc_meta: DocumentMeta,
    doc_region: [i32; 4],
) -> Result<DocTelemetry, AutomationError> {
    let expected = (img_w as usize).checked_mul(img_h as usize).and_then(|p| p.checked_mul(4)).ok_or_else(|| {
        AutomationError::BadRequest("image dimensions overflow".into())
    })?;
    if pixels.len() < expected || img_w == 0 || img_h == 0 {
        return Err(AutomationError::BadRequest("invalid RGBA buffer dimensions for analysis".into()));
    }
    let px_bounds = map_doc_rect_to_preview(doc_region, doc_meta.width, doc_meta.height, img_w, img_h);
    let accum = collect_pixel_accum(pixels, img_w, px_bounds)?;
    let (luminance, histogram_16_bins) = build_luminance_stats(&accum);
    let exposure = build_exposure_stats(&accum);
    let channels = build_channel_stats(&accum);
    let spatial_grid_3x3 = build_spatial_grid(&accum);
    let dominant_colors = build_dominant_colors(&accum);
    let edge_sharpness = compute_edge_sharpness(pixels, img_w, px_bounds);

    Ok(DocTelemetry {
        document: doc_meta,
        analyzed_region: AnalyzedRegionMeta {
            x: doc_region[0],
            y: doc_region[1],
            width: u32::try_from(doc_region[2]).unwrap_or(0),
            height: u32::try_from(doc_region[3]).unwrap_or(0),
            sampled_pixels: accum.total,
            preview_width: img_w,
            preview_height: img_h,
        },
        luminance,
        exposure,
        channels,
        histogram_16_bins,
        spatial_grid_3x3,
        dominant_colors,
        edge_sharpness,
    })
}

#[derive(Clone, Copy)]
struct PixelBounds {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
}

fn map_doc_rect_to_preview(rect: [i32; 4], doc_w: u32, doc_h: u32, img_w: u32, img_h: u32) -> PixelBounds {
    let dw = f64::from(doc_w.max(1));
    let dh = f64::from(doc_h.max(1));
    let iw = f64::from(img_w);
    let ih = f64::from(img_h);
    let rx0 = f64::from(rect[0].max(0));
    let ry0 = f64::from(rect[1].max(0));
    let rx1 = f64::from(rect[0].saturating_add(rect[2]).max(1));
    let ry1 = f64::from(rect[1].saturating_add(rect[3]).max(1));

    let x0 = ((rx0 / dw) * iw).floor().clamp(0.0, (img_w - 1) as f64) as u32;
    let y0 = ((ry0 / dh) * ih).floor().clamp(0.0, (img_h - 1) as f64) as u32;
    let x1 = ((rx1 / dw) * iw).ceil().clamp((x0 + 1) as f64, iw) as u32;
    let y1 = ((ry1 / dh) * ih).ceil().clamp((y0 + 1) as f64, ih) as u32;
    PixelBounds { x0, y0, x1, y1 }
}

struct PixelAccum {
    total: usize,
    opaque_count: usize,
    transparent_count: usize,
    shadow_clip: usize,
    highlight_clip: usize,
    midtone: usize,
    vivid_count: usize,
    sum_luma: f64,
    sum_luma_sq: f64,
    sum_r: f64,
    sum_g: f64,
    sum_b: f64,
    sum_a: f64,
    sum_sat: f64,
    min_luma: f32,
    max_luma: f32,
    hist256: [usize; 256],
    hist16: [usize; 16],
    grid_counts: [usize; 9],
    grid_luma: [f64; 9],
    grid_r: [f64; 9],
    grid_g: [f64; 9],
    grid_b: [f64; 9],
    color_bins: HashMap<u16, usize>,
}

impl PixelAccum {
    fn new() -> Self {
        Self {
            total: 0,
            opaque_count: 0,
            transparent_count: 0,
            shadow_clip: 0,
            highlight_clip: 0,
            midtone: 0,
            vivid_count: 0,
            sum_luma: 0.0,
            sum_luma_sq: 0.0,
            sum_r: 0.0,
            sum_g: 0.0,
            sum_b: 0.0,
            sum_a: 0.0,
            sum_sat: 0.0,
            min_luma: 1.0,
            max_luma: 0.0,
            hist256: [0; 256],
            hist16: [0; 16],
            grid_counts: [0; 9],
            grid_luma: [0.0; 9],
            grid_r: [0.0; 9],
            grid_g: [0.0; 9],
            grid_b: [0.0; 9],
            color_bins: HashMap::new(),
        }
    }
}

fn collect_pixel_accum(pixels: &[u8], img_w: u32, b: PixelBounds) -> Result<PixelAccum, AutomationError> {
    let mut acc = PixelAccum::new();
    let rw = (b.x1 - b.x0).max(1);
    let rh = (b.y1 - b.y0).max(1);
    for y in b.y0..b.y1 {
        let row_cell = (((y - b.y0) * 3) / rh).min(2) as usize;
        for x in b.x0..b.x1 {
            let idx = ((y as usize) * (img_w as usize) + (x as usize)) * 4;
            let Some(px) = pixels.get(idx..idx + 4) else { continue };
            let col_cell = (((x - b.x0) * 3) / rw).min(2) as usize;
            let cell = row_cell * 3 + col_cell;
            acc.push_pixel(px, cell);
        }
    }
    if acc.total == 0 {
        return Err(AutomationError::BadRequest("analyzed region contains no pixels".into()));
    }
    Ok(acc)
}

impl PixelAccum {
    fn push_pixel(&mut self, px: &[u8], cell: usize) {
        self.total += 1;
        let r = f32::from(px[0]) / 255.0;
        let g = f32::from(px[1]) / 255.0;
        let b = f32::from(px[2]) / 255.0;
        let a = f32::from(px[3]) / 255.0;
        self.sum_a += f64::from(a);
        if px[3] < 8 {
            self.transparent_count += 1;
            return;
        }
        self.opaque_count += 1;
        let luma = (0.2126 * r + 0.7152 * g + 0.0722 * b).clamp(0.0, 1.0);
        self.min_luma = self.min_luma.min(luma);
        self.max_luma = self.max_luma.max(luma);
        self.sum_luma += f64::from(luma);
        self.sum_luma_sq += f64::from(luma) * f64::from(luma);
        self.sum_r += f64::from(r);
        self.sum_g += f64::from(g);
        self.sum_b += f64::from(b);

        if luma <= 0.02 {
            self.shadow_clip += 1;
        } else if luma >= 0.98 {
            self.highlight_clip += 1;
        }
        if (0.25..=0.75).contains(&luma) {
            self.midtone += 1;
        }

        let max_c = r.max(g).max(b);
        let min_c = r.min(g).min(b);
        let sat = if max_c > 0.0 { (max_c - min_c) / max_c } else { 0.0 };
        self.sum_sat += f64::from(sat);
        if sat >= 0.5 {
            self.vivid_count += 1;
        }

        let b256 = ((luma * 255.0).round() as usize).min(255);
        if let Some(slot) = self.hist256.get_mut(b256) {
            *slot += 1;
        }
        let b16 = (b256 / 16).min(15);
        if let Some(slot) = self.hist16.get_mut(b16) {
            *slot += 1;
        }
        if let Some(cnt) = self.grid_counts.get_mut(cell) {
            *cnt += 1;
            self.grid_luma[cell] += f64::from(luma);
            self.grid_r[cell] += f64::from(r);
            self.grid_g[cell] += f64::from(g);
            self.grid_b[cell] += f64::from(b);
        }

        // Quantize RGB into 5-bit/5-bit/5-bit key for dominant palette clustering.
        let key = ((u16::from(px[0] >> 3)) << 10) | ((u16::from(px[1] >> 3)) << 5) | u16::from(px[2] >> 3);
        *self.color_bins.entry(key).or_insert(0) += 1;
    }
}

fn percentile_from_hist256(hist: &[usize; 256], total: usize, q: f64) -> f32 {
    if total == 0 {
        return 0.0;
    }
    let target = ((total as f64) * q).ceil() as usize;
    let mut seen = 0usize;
    for (bin, &count) in hist.iter().enumerate() {
        seen += count;
        if seen >= target {
            return round3(bin as f32 / 255.0);
        }
    }
    1.0
}

fn build_luminance_stats(acc: &PixelAccum) -> (LuminanceStats, Vec<f32>) {
    if acc.opaque_count == 0 {
        return (
            LuminanceStats { min: 0.0, max: 0.0, mean: 0.0, std_dev: 0.0, p05: 0.0, p50: 0.0, p95: 0.0 },
            vec![0.0; 16],
        );
    }
    let n = acc.opaque_count as f64;
    let mean = acc.sum_luma / n;
    let var = (acc.sum_luma_sq / n - mean * mean).max(0.0);
    let hist = acc.hist16.iter().map(|&c| round3((c as f64 / n) as f32)).collect();
    (
        LuminanceStats {
            min: round3(acc.min_luma),
            max: round3(acc.max_luma),
            mean: round3(mean as f32),
            std_dev: round3(var.sqrt() as f32),
            p05: percentile_from_hist256(&acc.hist256, acc.opaque_count, 0.05),
            p50: percentile_from_hist256(&acc.hist256, acc.opaque_count, 0.50),
            p95: percentile_from_hist256(&acc.hist256, acc.opaque_count, 0.95),
        },
        hist,
    )
}

fn build_exposure_stats(acc: &PixelAccum) -> ExposureStats {
    let t = acc.total.max(1) as f64;
    let o = acc.opaque_count.max(1) as f64;
    ExposureStats {
        shadow_clip_pct: pct(acc.shadow_clip, o),
        highlight_clip_pct: pct(acc.highlight_clip, o),
        midtone_pct: pct(acc.midtone, o),
        transparent_pct: pct(acc.transparent_count, t),
    }
}

fn build_channel_stats(acc: &PixelAccum) -> ChannelStats {
    let t = acc.total.max(1) as f64;
    let o = acc.opaque_count.max(1) as f64;
    let r = (acc.sum_r / o) as f32;
    let g = (acc.sum_g / o) as f32;
    let b = (acc.sum_b / o) as f32;
    ChannelStats {
        r_mean: round3(r),
        g_mean: round3(g),
        b_mean: round3(b),
        a_mean: round3((acc.sum_a / t) as f32),
        mean_saturation: round3((acc.sum_sat / o) as f32),
        vivid_pct: pct(acc.vivid_count, o),
        warm_cool_bias: round3(r - b),
        tint_bias: round3(g - 0.5 * (r + b)),
    }
}

fn build_spatial_grid(acc: &PixelAccum) -> Vec<GridCellStats> {
    const NAMES: [&str; 9] = [
        "top-left",
        "top-center",
        "top-right",
        "mid-left",
        "center",
        "mid-right",
        "bottom-left",
        "bottom-center",
        "bottom-right",
    ];
    NAMES
        .iter()
        .enumerate()
        .map(|(i, &cell)| {
            let c = acc.grid_counts.get(i).copied().unwrap_or(0);
            if c == 0 {
                return GridCellStats { cell, mean_luma: 0.0, r_mean: 0.0, g_mean: 0.0, b_mean: 0.0 };
            }
            let n = c as f64;
            GridCellStats {
                cell,
                mean_luma: round3((acc.grid_luma[i] / n) as f32),
                r_mean: round3((acc.grid_r[i] / n) as f32),
                g_mean: round3((acc.grid_g[i] / n) as f32),
                b_mean: round3((acc.grid_b[i] / n) as f32),
            }
        })
        .collect()
}

fn build_dominant_colors(acc: &PixelAccum) -> Vec<DominantColor> {
    if acc.opaque_count == 0 {
        return Vec::new();
    }
    let mut entries: Vec<(u16, usize)> = acc.color_bins.iter().map(|(&k, &v)| (k, v)).collect();
    entries.sort_by_key(|a| std::cmp::Reverse(a.1));
    let total = acc.opaque_count as f64;
    entries
        .into_iter()
        .take(5)
        .map(|(key, count)| {
            let r = (((key >> 10) & 31) as u8) * 8 + 4;
            let g = (((key >> 5) & 31) as u8) * 8 + 4;
            let b = ((key & 31) as u8) * 8 + 4;
            DominantColor { hex: format!("#{r:02X}{g:02X}{b:02X}"), pct: pct(count, total), r, g, b }
        })
        .collect()
}

fn compute_edge_sharpness(pixels: &[u8], img_w: u32, b: PixelBounds) -> f32 {
    if b.x1.saturating_sub(b.x0) < 3 || b.y1.saturating_sub(b.y0) < 3 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for y in (b.y0 + 1)..(b.y1 - 1) {
        for x in (b.x0 + 1)..(b.x1 - 1) {
            let gx = luma_at(pixels, img_w, x + 1, y) - luma_at(pixels, img_w, x - 1, y);
            let gy = luma_at(pixels, img_w, x, y + 1) - luma_at(pixels, img_w, x, y - 1);
            sum += f64::from((gx * gx + gy * gy).sqrt());
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { round3((sum / count as f64) as f32) }
}

fn luma_at(pixels: &[u8], img_w: u32, x: u32, y: u32) -> f32 {
    let idx = ((y as usize) * (img_w as usize) + (x as usize)) * 4;
    let Some(px) = pixels.get(idx..idx + 4) else { return 0.0 };
    (0.2126 * f32::from(px[0]) + 0.7152 * f32::from(px[1]) + 0.0722 * f32::from(px[2])) / 255.0
}

fn round3(v: f32) -> f32 {
    (v * 1000.0).round() / 1000.0
}

fn pct(count: usize, total: f64) -> f32 {
    round3(((count as f64 / total) * 100.0) as f32)
}

/// Render a document thumbnail with coordinate gridlines, ruler labels, and selection/box overlays.
pub fn render_annotated_png(doc: &Document, opts: &AnnotateOptions) -> Result<(Vec<u8>, AnnotatedPreviewMeta), AutomationError> {
    if doc.size.width == 0 || doc.size.height == 0 {
        return Err(AutomationError::BadRequest("cannot render annotated preview of a zero-sized document".into()));
    }
    let side = opts.max_side.clamp(MIN_SIDE, MAX_SIDE);
    let mut thumb = photocraft_compose::thumbnail(doc, side);
    let sel = doc_selection_bounds(doc);
    let meta = annotate_rgba_in_place(
        &mut thumb.pixels,
        thumb.width,
        thumb.height,
        doc.size.width,
        doc.size.height,
        sel,
        opts,
    )?;
    let image = Image::from_u8(thumb.width, thumb.height, ChannelLayout::Rgba, thumb.pixels)
        .map_err(|e| AutomationError::Other(e.to_string()))?;
    let png = photocraft_codecs::encode(&image, Format::Png, &Default::default()).map_err(|e| AutomationError::Other(e.to_string()))?;
    Ok((png, meta))
}

/// Annotate an in-memory RGBA8 buffer in place using original document coordinate space.
pub fn annotate_rgba_in_place(
    pixels: &mut [u8],
    img_w: u32,
    img_h: u32,
    doc_w: u32,
    doc_h: u32,
    selection_bounds: Option<[i32; 4]>,
    opts: &AnnotateOptions,
) -> Result<AnnotatedPreviewMeta, AutomationError> {
    let need = (img_w as usize).checked_mul(img_h as usize).and_then(|n| n.checked_mul(4)).ok_or_else(|| {
        AutomationError::BadRequest("annotated image dimensions overflow".into())
    })?;
    if pixels.len() < need || img_w == 0 || img_h == 0 || doc_w == 0 || doc_h == 0 {
        return Err(AutomationError::BadRequest("invalid dimensions for annotated preview".into()));
    }
    let step = if opts.show_grid || opts.show_axes {
        Some(opts.grid_step.unwrap_or_else(|| pick_grid_step(doc_w.max(doc_h))).max(1))
    } else {
        None
    };

    let mut canvas = Canvas { pixels, width: img_w, height: img_h };
    if let Some(step_px) = step {
        draw_grid_and_axes(&mut canvas, doc_w, doc_h, step_px, opts);
    }
    if opts.show_selection && let Some(sel) = selection_bounds {
        draw_doc_box(&mut canvas, DocBox { rect: sel, doc_w, doc_h, color: [0, 229, 255, 240], label_prefix: "SEL " });
    }
    if let Some(hl) = opts.highlight_box {
        draw_doc_box(&mut canvas, DocBox { rect: hl, doc_w, doc_h, color: [255, 179, 0, 240], label_prefix: "BOX " });
    }

    Ok(AnnotatedPreviewMeta {
        doc_width: doc_w,
        doc_height: doc_h,
        image_width: img_w,
        image_height: img_h,
        grid_step_px: step,
        selection_bounds,
        highlight_box: opts.highlight_box,
    })
}

fn pick_grid_step(max_dim: u32) -> u32 {
    const STEPS: [u32; 11] = [16, 32, 50, 100, 200, 250, 500, 1000, 2000, 4000, 8000];
    for &s in &STEPS {
        if max_dim / s <= 8 {
            return s;
        }
    }
    (max_dim / 6).max(1)
}

fn draw_grid_and_axes(canvas: &mut Canvas<'_>, doc_w: u32, doc_h: u32, step_px: u32, opts: &AnnotateOptions) {
    let mut x = step_px;
    while x < doc_w {
        let px = ((f64::from(x) / f64::from(doc_w)) * f64::from(canvas.width)).round() as i32;
        if opts.show_grid {
            canvas.draw_dashed_vline(px, [255, 255, 255, 120], [0, 0, 0, 110]);
        }
        if opts.show_axes {
            let label = x.to_string();
            canvas.draw_pill(px.saturating_add(2), 2, &label, [18, 22, 28, 210], [240, 246, 255, 255]);
        }
        x = x.saturating_add(step_px);
    }

    let mut y = step_px;
    while y < doc_h {
        let py = ((f64::from(y) / f64::from(doc_h)) * f64::from(canvas.height)).round() as i32;
        if opts.show_grid {
            canvas.draw_dashed_hline(py, [255, 255, 255, 120], [0, 0, 0, 110]);
        }
        if opts.show_axes {
            let label = y.to_string();
            canvas.draw_pill(2, py.saturating_add(2), &label, [18, 22, 28, 210], [240, 246, 255, 255]);
        }
        y = y.saturating_add(step_px);
    }

    if opts.show_axes {
        let dims = format!("{doc_w}x{doc_h}");
        canvas.draw_pill(2, 2, &dims, [18, 22, 28, 220], [130, 230, 255, 255]);
    }
}

struct DocBox {
    rect: [i32; 4],
    doc_w: u32,
    doc_h: u32,
    color: [u8; 4],
    label_prefix: &'static str,
}

fn draw_doc_box(canvas: &mut Canvas<'_>, box_spec: DocBox) {
    let [x, y, w, h] = box_spec.rect;
    if w <= 0 || h <= 0 {
        return;
    }
    let sx = f64::from(canvas.width) / f64::from(box_spec.doc_w.max(1));
    let sy = f64::from(canvas.height) / f64::from(box_spec.doc_h.max(1));
    let x0 = (f64::from(x) * sx).round() as i32;
    let y0 = (f64::from(y) * sy).round() as i32;
    let x1 = (f64::from(x.saturating_add(w)) * sx).round() as i32;
    let y1 = (f64::from(y.saturating_add(h)) * sy).round() as i32;

    canvas.draw_box_outline(x0, y0, x1, y1, [0, 0, 0, 200]);
    canvas.draw_box_outline(x0.saturating_add(1), y0.saturating_add(1), x1.saturating_sub(1), y1.saturating_sub(1), box_spec.color);
    let label = format!("{}{x},{y} {w}x{h}", box_spec.label_prefix);
    let ly = if y0 > 14 { y0 - 13 } else { y0.saturating_add(3) };
    canvas.draw_pill(x0.max(2), ly.max(2), &label, [12, 16, 22, 225], box_spec.color);
}

struct Canvas<'a> {
    pixels: &'a mut [u8],
    width: u32,
    height: u32,
}

impl<'a> Canvas<'a> {
    fn blend_px(&mut self, x: i32, y: i32, src: [u8; 4]) {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return;
        }
        let idx = ((y as usize) * (self.width as usize) + (x as usize)) * 4;
        let Some(dst) = self.pixels.get_mut(idx..idx + 4) else { return };
        let a = f32::from(src[3]) / 255.0;
        let inv = 1.0 - a;
        dst[0] = (f32::from(dst[0]) * inv + f32::from(src[0]) * a).round().clamp(0.0, 255.0) as u8;
        dst[1] = (f32::from(dst[1]) * inv + f32::from(src[1]) * a).round().clamp(0.0, 255.0) as u8;
        dst[2] = (f32::from(dst[2]) * inv + f32::from(src[2]) * a).round().clamp(0.0, 255.0) as u8;
        dst[3] = dst[3].max(src[3]);
    }

    fn draw_dashed_vline(&mut self, x: i32, light: [u8; 4], dark: [u8; 4]) {
        for y in 0..self.height as i32 {
            let color = if (y / 4) % 2 == 0 { light } else { dark };
            self.blend_px(x, y, color);
        }
    }

    fn draw_dashed_hline(&mut self, y: i32, light: [u8; 4], dark: [u8; 4]) {
        for x in 0..self.width as i32 {
            let color = if (x / 4) % 2 == 0 { light } else { dark };
            self.blend_px(x, y, color);
        }
    }

    fn draw_box_outline(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, color: [u8; 4]) {
        let lx0 = x0.min(x1);
        let rx1 = x0.max(x1);
        let ty0 = y0.min(y1);
        let by1 = y0.max(y1);
        for x in lx0..=rx1 {
            self.blend_px(x, ty0, color);
            self.blend_px(x, by1, color);
        }
        for y in ty0..=by1 {
            self.blend_px(lx0, y, color);
            self.blend_px(rx1, y, color);
        }
    }

    fn fill_rect(&mut self, x: i32, y: i32, w: i32, h: i32, color: [u8; 4]) {
        for dy in 0..h {
            for dx in 0..w {
                self.blend_px(x.saturating_add(dx), y.saturating_add(dy), color);
            }
        }
    }

    fn draw_pill(&mut self, x: i32, y: i32, text: &str, bg: [u8; 4], fg: [u8; 4]) {
        let chars: Vec<char> = text.chars().take(32).collect();
        if chars.is_empty() {
            return;
        }
        let text_w = (chars.len() as i32) * 6 - 1;
        let pad_x = 3;
        let pad_y = 2;
        let box_w = text_w + pad_x * 2;
        let box_h = 7 + pad_y * 2;
        let clamped_x = x.clamp(0, (self.width as i32).saturating_sub(box_w).max(0));
        let clamped_y = y.clamp(0, (self.height as i32).saturating_sub(box_h).max(0));
        self.fill_rect(clamped_x, clamped_y, box_w, box_h, bg);
        for (i, ch) in chars.into_iter().enumerate() {
            self.draw_glyph(clamped_x + pad_x + (i as i32) * 6, clamped_y + pad_y, ch, fg);
        }
    }

    fn draw_glyph(&mut self, x: i32, y: i32, ch: char, fg: [u8; 4]) {
        let rows = glyph_5x7(ch);
        for (ry, bits) in rows.into_iter().enumerate() {
            for rx in 0..5u8 {
                if (bits >> (4 - rx)) & 1 == 1 {
                    self.blend_px(x + i32::from(rx), y + ry as i32, fg);
                }
            }
        }
    }
}

fn glyph_5x7(ch: char) -> [u8; 7] {
    match ch.to_ascii_uppercase() {
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111],
        '3' => [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        '-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        ',' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00110, 0b01000],
        ':' => [0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b00000],
        _ => [0; 7],
    }
}
