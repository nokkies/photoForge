//! ComfyUI HTTP orchestration, workflow templates, and crop-and-stitch inpainting helpers.

use std::time::{Duration, Instant};

use photocraft_codecs::{ChannelLayout, Format, Image};
use photocraft_doc::{Document, Rect};
use photocraft_engine::Session;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;

use crate::AutomationError;

/// Default ComfyUI endpoint when no parameter or env override is provided.
pub const DEFAULT_COMFY_URL: &str = "http://127.0.0.1:8188";
const ENV_COMFY_URL: &str = "PHOTOCRAFT_COMFY_URL";
const STATUS_TIMEOUT: Duration = Duration::from_secs(5);
const UPLOAD_TIMEOUT: Duration = Duration::from_secs(30);
const PROMPT_TIMEOUT: Duration = Duration::from_secs(15);
const POLL_INTERVAL: Duration = Duration::from_millis(150);
const DEFAULT_RUN_TIMEOUT_MS: u64 = 120_000;
const MIN_RUN_TIMEOUT_MS: u64 = 500;
const MAX_RUN_TIMEOUT_MS: u64 = 600_000;
const MAX_HTTP_HEADER_BYTES: usize = 256 * 1024;
const MAX_HTTP_BODY_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_CHECKPOINT: &str = "v1-5-pruned-emaonly.safetensors";

/// Resolve the ComfyUI base URL from an explicit override, `PHOTOCRAFT_COMFY_URL`, or default.
pub fn resolve_comfy_url(override_url: Option<&str>) -> String {
    override_url
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| std::env::var(ENV_COMFY_URL).ok().map(|s| s.trim().to_owned()).filter(|s| !s.is_empty()))
        .unwrap_or_else(|| DEFAULT_COMFY_URL.to_owned())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Endpoint {
    host: String,
    port: u16,
    base_path: String,
}

impl Endpoint {
    fn host_header(&self) -> String {
        if self.host.contains(':') { format!("[{}]:{}", self.host, self.port) } else { format!("{}:{}", self.host, self.port) }
    }

    fn connect_addr(&self) -> String {
        if self.host.contains(':') { format!("[{}]:{}", self.host, self.port) } else { format!("{}:{}", self.host, self.port) }
    }

    fn full_path(&self, rel_path: &str) -> String {
        let rel = if rel_path.starts_with('/') { rel_path } else { return format!("{}/{}", self.base_path, rel_path) };
        if self.base_path.is_empty() { rel.to_owned() } else { format!("{}{}", self.base_path, rel) }
    }
}

fn parse_http_endpoint(url: &str) -> Result<Endpoint, AutomationError> {
    let trimmed = url.trim().trim_end_matches('/');
    let rest = if let Some(after) = trimmed.strip_prefix("http://") {
        after
    } else if trimmed.starts_with("https://") {
        return Err(AutomationError::BadRequest(
            "https:// ComfyUI URLs are not supported directly; point PHOTOCRAFT_COMFY_URL to an http:// endpoint".into(),
        ));
    } else if !trimmed.contains("://") {
        trimmed
    } else {
        return Err(AutomationError::BadRequest(format!("unsupported ComfyUI URL scheme: `{trimmed}`")));
    };

    let (authority, base_path) = match rest.find('/') {
        Some(idx) => (&rest[..idx], rest[idx..].trim_end_matches('/').to_owned()),
        None => (rest, String::new()),
    };
    let (host, port) = parse_host_port(authority)?;
    Ok(Endpoint { host, port, base_path })
}

fn parse_host_port(authority: &str) -> Result<(String, u16), AutomationError> {
    if authority.is_empty() {
        return Err(AutomationError::BadRequest("empty host in ComfyUI URL".into()));
    }
    if let Some(after_open) = authority.strip_prefix('[') {
        let close = after_open.find(']').ok_or_else(|| AutomationError::BadRequest("unclosed IPv6 bracket in ComfyUI URL".into()))?;
        let host = &after_open[..close];
        let tail = &after_open[close + 1..];
        let port = match tail.strip_prefix(':') {
            Some(p) => p.parse::<u16>().map_err(|_| AutomationError::BadRequest(format!("invalid port in ComfyUI URL: `{p}`")))?,
            None if tail.is_empty() => 8188,
            None => return Err(AutomationError::BadRequest(format!("invalid ComfyUI URL authority: `{authority}`"))),
        };
        return Ok((host.to_owned(), port));
    }
    if let Some((host, port_str)) = authority.rsplit_once(':') {
        if host.is_empty() {
            return Err(AutomationError::BadRequest("empty host in ComfyUI URL".into()));
        }
        let port = port_str.parse::<u16>().map_err(|_| AutomationError::BadRequest(format!("invalid port in ComfyUI URL: `{port_str}`")))?;
        return Ok((host.to_owned(), port));
    }
    Ok((authority.to_owned(), 8188))
}

struct HttpResponse {
    status: u16,
    body: Vec<u8>,
}

async fn send_http_request(
    endpoint: &Endpoint,
    method: &str,
    path: &str,
    headers: &[(&str, String)],
    body: &[u8],
    timeout: Duration,
) -> Result<HttpResponse, AutomationError> {
    let target = endpoint.connect_addr();
    let req_path = endpoint.full_path(path);
    let host_hdr = endpoint.host_header();

    tokio::time::timeout(timeout, async {
        let mut stream = TcpStream::connect(&target)
            .await
            .map_err(|e| AutomationError::App(format!("ComfyUI connection to `{target}` failed: {e}")))?;

        let mut raw = Vec::with_capacity(256 + body.len());
        let head = format!("{method} {req_path} HTTP/1.1\r\nHost: {host_hdr}\r\nConnection: close\r\nContent-Length: {}\r\n", body.len());
        raw.extend_from_slice(head.as_bytes());
        for (name, val) in headers {
            raw.extend_from_slice(format!("{name}: {val}\r\n").as_bytes());
        }
        raw.extend_from_slice(b"\r\n");
        raw.extend_from_slice(body);

        stream.write_all(&raw).await.map_err(|e| AutomationError::Io(format!("HTTP write failed: {e}")))?;
        read_http_response(stream).await
    })
    .await
    .map_err(|_| AutomationError::App(format!("ComfyUI HTTP {method} `{path}` timed out after {} ms", timeout.as_millis())))?
}

async fn read_http_response(stream: TcpStream) -> Result<HttpResponse, AutomationError> {
    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    let n = reader.read_line(&mut status_line).await.map_err(|e| AutomationError::Io(format!("HTTP status read failed: {e}")))?;
    if n == 0 || status_line.len() > 8192 {
        return Err(AutomationError::App("empty or oversized HTTP status line from ComfyUI".into()));
    }
    let status = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse::<u16>().ok())
        .ok_or_else(|| AutomationError::App(format!("malformed HTTP status line: `{}`", status_line.trim())))?;

    let mut header_bytes = 0usize;
    let mut content_length: Option<usize> = None;
    let mut chunked = false;
    loop {
        let mut line = String::new();
        let read = reader.read_line(&mut line).await.map_err(|e| AutomationError::Io(format!("HTTP header read failed: {e}")))?;
        header_bytes = header_bytes.saturating_add(read);
        if header_bytes > MAX_HTTP_HEADER_BYTES {
            return Err(AutomationError::App("ComfyUI HTTP headers exceed size limit".into()));
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }
        if let Some((k, v)) = trimmed.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse::<usize>().ok();
            } else if k.trim().eq_ignore_ascii_case("transfer-encoding") && v.to_ascii_lowercase().contains("chunked") {
                chunked = true;
            }
        }
    }

    let body = if chunked {
        read_chunked_body(&mut reader).await?
    } else if let Some(len) = content_length {
        if len > MAX_HTTP_BODY_BYTES {
            return Err(AutomationError::App(format!("ComfyUI response ({len} bytes) exceeds 64 MiB limit")));
        }
        let mut buf = vec![0u8; len];
        reader.read_exact(&mut buf).await.map_err(|e| AutomationError::Io(format!("HTTP body read failed: {e}")))?;
        buf
    } else {
        let mut limited = reader.take((MAX_HTTP_BODY_BYTES + 1) as u64);
        let mut buf = Vec::new();
        limited.read_to_end(&mut buf).await.map_err(|e| AutomationError::Io(format!("HTTP stream read failed: {e}")))?;
        if buf.len() > MAX_HTTP_BODY_BYTES {
            return Err(AutomationError::App("ComfyUI response exceeds 64 MiB limit".into()));
        }
        buf
    };

    Ok(HttpResponse { status, body })
}

async fn read_chunked_body<R: AsyncBufReadExt + Unpin>(reader: &mut R) -> Result<Vec<u8>, AutomationError> {
    let mut out = Vec::new();
    loop {
        let mut line = String::new();
        let read = reader.read_line(&mut line).await.map_err(|e| AutomationError::Io(format!("chunk header read failed: {e}")))?;
        if read == 0 {
            break;
        }
        let hex_len = line.trim().split(';').next().unwrap_or("").trim();
        if hex_len.is_empty() {
            continue;
        }
        let size = usize::from_str_radix(hex_len, 16).map_err(|_| AutomationError::App(format!("invalid HTTP chunk size `{hex_len}`")))?;
        if size == 0 {
            break;
        }
        let next_total = out.len().checked_add(size).ok_or_else(|| AutomationError::App("chunked body overflow".into()))?;
        if next_total > MAX_HTTP_BODY_BYTES {
            return Err(AutomationError::App("chunked ComfyUI response exceeds 64 MiB limit".into()));
        }
        let start = out.len();
        out.resize(next_total, 0);
        reader.read_exact(&mut out[start..]).await.map_err(|e| AutomationError::Io(format!("chunk payload read failed: {e}")))?;
        let mut crlf = [0u8; 2];
        let _ = reader.read_exact(&mut crlf).await;
    }
    Ok(out)
}

async fn http_get_json(endpoint: &Endpoint, path: &str, timeout: Duration) -> Result<Value, AutomationError> {
    let headers = [("Accept", "application/json".into())];
    let resp = send_http_request(endpoint, "GET", path, &headers, &[], timeout).await?;
    parse_json_response(path, &resp)
}

async fn http_get_bytes(endpoint: &Endpoint, path: &str, timeout: Duration) -> Result<Vec<u8>, AutomationError> {
    let resp = send_http_request(endpoint, "GET", path, &[], &[], timeout).await?;
    if !(200..300).contains(&resp.status) {
        let snippet = String::from_utf8_lossy(&resp.body[..resp.body.len().min(256)]);
        return Err(AutomationError::App(format!("ComfyUI GET `{path}` returned HTTP {}: {snippet}", resp.status)));
    }
    Ok(resp.body)
}

async fn http_post_json(endpoint: &Endpoint, path: &str, body: &Value, timeout: Duration) -> Result<Value, AutomationError> {
    let bytes = serde_json::to_vec(body).map_err(|e| AutomationError::Other(format!("failed to encode request JSON: {e}")))?;
    let headers = [("Content-Type", "application/json".into()), ("Accept", "application/json".into())];
    let resp = send_http_request(endpoint, "POST", path, &headers, &bytes, timeout).await?;
    parse_json_response(path, &resp)
}

async fn http_upload_png(endpoint: &Endpoint, filename: &str, png_bytes: &[u8], timeout: Duration) -> Result<String, AutomationError> {
    let boundary = "----PhotoForgeComfyBoundary7a9c2e";
    let safe_name: String = filename.chars().map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' { c } else { '_' }).collect();
    let mut body = Vec::with_capacity(png_bytes.len() + 512);
    body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n").as_bytes());
    body.extend_from_slice(
        format!("--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{safe_name}\"\r\nContent-Type: image/png\r\n\r\n").as_bytes(),
    );
    body.extend_from_slice(png_bytes);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

    let headers = [("Content-Type", format!("multipart/form-data; boundary={boundary}")), ("Accept", "application/json".into())];
    let resp = send_http_request(endpoint, "POST", "/upload/image", &headers, &body, timeout).await?;
    let v = parse_json_response("/upload/image", &resp)?;
    let name = v.get("name").and_then(Value::as_str).unwrap_or(&safe_name).to_owned();
    let subfolder = v.get("subfolder").and_then(Value::as_str).unwrap_or("");
    if subfolder.is_empty() { Ok(name) } else { Ok(format!("{subfolder}/{name}")) }
}

fn parse_json_response(path: &str, resp: &HttpResponse) -> Result<Value, AutomationError> {
    if !(200..300).contains(&resp.status) {
        let snippet = String::from_utf8_lossy(&resp.body[..resp.body.len().min(300)]);
        return Err(AutomationError::App(format!("ComfyUI `{path}` returned HTTP {}: {snippet}", resp.status)));
    }
    serde_json::from_slice(&resp.body).map_err(|e| AutomationError::App(format!("invalid JSON from ComfyUI `{path}`: {e}")))
}

/// Inspect ComfyUI server reachability, hardware summary, and installed checkpoints.
pub async fn check_status(comfy_url: Option<&str>) -> Result<Value, AutomationError> {
    let url = resolve_comfy_url(comfy_url);
    let endpoint = parse_http_endpoint(&url)?;
    let stats = match http_get_json(&endpoint, "/system_stats", STATUS_TIMEOUT).await {
        Ok(v) => v,
        Err(e) => {
            return Ok(json!({
                "online": false,
                "comfy_url": url,
                "error": e.to_string(),
                "checkpoints": [],
            }));
        }
    };
    let checkpoints = fetch_checkpoints(&endpoint).await.unwrap_or_default();
    Ok(json!({
        "online": true,
        "comfy_url": url,
        "system": stats.get("system").cloned().unwrap_or(Value::Null),
        "devices": stats.get("devices").cloned().unwrap_or_else(|| json!([])),
        "checkpoints": checkpoints,
    }))
}

async fn fetch_checkpoints(endpoint: &Endpoint) -> Result<Vec<String>, AutomationError> {
    let info = http_get_json(endpoint, "/object_info/CheckpointLoaderSimple", STATUS_TIMEOUT).await?;
    let list = info
        .get("CheckpointLoaderSimple")
        .and_then(|v| v.get("input"))
        .and_then(|v| v.get("required"))
        .and_then(|v| v.get("ckpt_name"))
        .and_then(|v| v.get(0))
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(Value::as_str).take(256).map(ToOwned::to_owned).collect())
        .unwrap_or_default();
    Ok(list)
}

async fn resolve_checkpoint(endpoint: &Endpoint, requested: Option<&str>) -> String {
    if let Some(c) = requested.map(str::trim).filter(|s| !s.is_empty()) {
        return c.to_owned();
    }
    if let Ok(list) = fetch_checkpoints(endpoint).await
        && let Some(first) = list.into_iter().next()
    {
        return first;
    }
    DEFAULT_CHECKPOINT.to_owned()
}

/// Parameters for building a ComfyUI text-to-image workflow graph.
pub struct Txt2ImgSpec {
    pub prompt: String,
    pub negative_prompt: String,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub cfg_scale: f64,
    pub seed: u64,
    pub checkpoint: String,
}

/// Parameters for building a ComfyUI inpainting workflow graph.
pub struct InpaintSpec {
    pub prompt: String,
    pub negative_prompt: String,
    pub uploaded_image: String,
    pub steps: u32,
    pub cfg_scale: f64,
    pub denoise: f64,
    pub seed: u64,
    pub checkpoint: String,
}

/// Build a standard ComfyUI txt2img API prompt graph.
pub fn build_txt2img_workflow(spec: &Txt2ImgSpec) -> Value {
    json!({
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": spec.seed,
                "steps": spec.steps,
                "cfg": spec.cfg_scale,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0]
            }
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": { "ckpt_name": spec.checkpoint }
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": { "width": spec.width, "height": spec.height, "batch_size": 1 }
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": { "text": spec.prompt, "clip": ["4", 1] }
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": { "text": spec.negative_prompt, "clip": ["4", 1] }
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": { "samples": ["3", 0], "vae": ["4", 2] }
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": { "filename_prefix": "photoforge_item", "images": ["8", 0] }
        }
    })
}

/// Build a standard ComfyUI inpaint API prompt graph using an RGBA image whose alpha channel holds the hole mask.
pub fn build_inpaint_workflow(spec: &InpaintSpec) -> Value {
    json!({
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": spec.seed,
                "steps": spec.steps,
                "cfg": spec.cfg_scale,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": spec.denoise,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["12", 0]
            }
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": { "ckpt_name": spec.checkpoint }
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": { "text": spec.prompt, "clip": ["4", 1] }
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": { "text": spec.negative_prompt, "clip": ["4", 1] }
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": { "samples": ["3", 0], "vae": ["4", 2] }
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": { "filename_prefix": "photoforge_fill", "images": ["8", 0] }
        },
        "10": {
            "class_type": "LoadImage",
            "inputs": { "image": spec.uploaded_image }
        },
        "12": {
            "class_type": "VAEEncodeForInpaint",
            "inputs": {
                "pixels": ["10", 0],
                "vae": ["4", 2],
                "mask": ["10", 1],
                "grow_mask_by": 6
            }
        }
    })
}

struct TemplateVars<'a> {
    prompt: &'a str,
    negative_prompt: &'a str,
    checkpoint: &'a str,
    image: Option<&'a str>,
    width: Option<u32>,
    height: Option<u32>,
    steps: u32,
    cfg: f64,
    seed: u64,
    denoise: f64,
}

fn substitute_workflow_vars(node: &mut Value, vars: &TemplateVars<'_>, depth: usize) {
    if depth > 32 {
        return;
    }
    match node {
        Value::String(s) => replace_string_placeholders(s, vars),
        Value::Array(arr) => {
            for item in arr {
                substitute_workflow_vars(item, vars, depth + 1);
            }
        }
        Value::Object(map) => {
            for (_, v) in map.iter_mut() {
                substitute_workflow_vars(v, vars, depth + 1);
            }
        }
        _ => {}
    }
}

fn replace_string_placeholders(s: &mut String, vars: &TemplateVars<'_>) {
    let mut replaced = s
        .replace("{{PROMPT}}", vars.prompt)
        .replace("{{NEGATIVE_PROMPT}}", vars.negative_prompt)
        .replace("{{CHECKPOINT}}", vars.checkpoint)
        .replace("{{STEPS}}", &vars.steps.to_string())
        .replace("{{CFG}}", &vars.cfg.to_string())
        .replace("{{SEED}}", &vars.seed.to_string())
        .replace("{{DENOISE}}", &vars.denoise.to_string());
    if let Some(img) = vars.image {
        replaced = replaced.replace("{{IMAGE}}", img).replace("{{MASK}}", img);
    }
    if let Some(w) = vars.width {
        replaced = replaced.replace("{{WIDTH}}", &w.to_string());
    }
    if let Some(h) = vars.height {
        replaced = replaced.replace("{{HEIGHT}}", &h.to_string());
    }
    *s = replaced;
}

/// Request arguments for standalone txt2img item generation.
pub struct GenerateItemRequest {
    pub prompt: String,
    pub negative_prompt: String,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub cfg_scale: f64,
    pub seed: Option<u64>,
    pub checkpoint: Option<String>,
    pub comfy_url: Option<String>,
    pub workflow: Option<Value>,
    pub timeout_ms: Option<u64>,
}

/// Run txt2img in ComfyUI and return the generated PNG bytes along with metadata.
pub async fn generate_item_png(req: &GenerateItemRequest) -> Result<(Vec<u8>, Value), AutomationError> {
    if req.prompt.trim().is_empty() {
        return Err(AutomationError::BadRequest("`prompt` must not be empty".into()));
    }
    if req.width == 0 || req.height == 0 || req.width > 4096 || req.height > 4096 {
        return Err(AutomationError::BadRequest(format!("generation size {}x{} must be within 1..=4096", req.width, req.height)));
    }
    let url = resolve_comfy_url(req.comfy_url.as_deref());
    let endpoint = parse_http_endpoint(&url)?;
    let checkpoint = resolve_checkpoint(&endpoint, req.checkpoint.as_deref()).await;
    let seed = req.seed.unwrap_or_else(default_seed);
    let steps = req.steps.clamp(1, 150);
    let cfg_scale = req.cfg_scale.clamp(0.5, 30.0);

    let workflow = match req.workflow.clone() {
        Some(mut custom) => {
            let vars = TemplateVars {
                prompt: &req.prompt,
                negative_prompt: &req.negative_prompt,
                checkpoint: &checkpoint,
                image: None,
                width: Some(req.width),
                height: Some(req.height),
                steps,
                cfg: cfg_scale,
                seed,
                denoise: 1.0,
            };
            substitute_workflow_vars(&mut custom, &vars, 0);
            custom
        }
        None => build_txt2img_workflow(&Txt2ImgSpec {
            prompt: req.prompt.clone(),
            negative_prompt: req.negative_prompt.clone(),
            width: req.width,
            height: req.height,
            steps,
            cfg_scale,
            seed,
            checkpoint: checkpoint.clone(),
        }),
    };

    let timeout_ms = req.timeout_ms.unwrap_or(DEFAULT_RUN_TIMEOUT_MS).clamp(MIN_RUN_TIMEOUT_MS, MAX_RUN_TIMEOUT_MS);
    let png = run_workflow_and_fetch_png(&endpoint, workflow, timeout_ms).await?;
    let meta = json!({
        "comfy_url": url,
        "checkpoint": checkpoint,
        "seed": seed,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "generated_size": [req.width, req.height],
    });
    Ok((png, meta))
}

/// Crop + feathered selection mask extracted from a document prior to ComfyUI upload.
#[derive(Debug, Clone)]
pub struct InpaintCrop {
    pub rect: [i32; 4],
    pub width: u32,
    pub height: u32,
    pub upload_png: Vec<u8>,
    pub feathered_mask: Vec<f32>,
}

/// Extract a padded RGB crop + feathered inpaint mask from the document selection or explicit bounds.
pub fn prepare_inpaint_crop(doc: &Document, bounds: Option<[i32; 4]>, padding: u32, feather: u32) -> Result<InpaintCrop, AutomationError> {
    if doc.size.width == 0 || doc.size.height == 0 {
        return Err(AutomationError::BadRequest("cannot inpaint a zero-sized document".into()));
    }
    let target = resolve_inpaint_target(doc, bounds)?;
    let doc_w_i32 = i32::try_from(doc.size.width).map_err(|_| AutomationError::BadRequest("document too wide".into()))?;
    let doc_h_i32 = i32::try_from(doc.size.height).map_err(|_| AutomationError::BadRequest("document too tall".into()))?;
    let pad = i32::try_from(padding.min(512)).unwrap_or(512);

    let cx0 = target.x0.saturating_sub(pad).clamp(0, doc_w_i32.saturating_sub(1));
    let cy0 = target.y0.saturating_sub(pad).clamp(0, doc_h_i32.saturating_sub(1));
    let cx1 = target.x1.saturating_add(pad).clamp(cx0 + 1, doc_w_i32);
    let cy1 = target.y1.saturating_add(pad).clamp(cy0 + 1, doc_h_i32);
    let cw = u32::try_from(cx1 - cx0).map_err(|_| AutomationError::BadRequest("invalid crop width".into()))?;
    let ch = u32::try_from(cy1 - cy0).map_err(|_| AutomationError::BadRequest("invalid crop height".into()))?;
    let total_px = (cw as usize).checked_mul(ch as usize).ok_or_else(|| AutomationError::BadRequest("inpaint crop overflows memory".into()))?;
    if total_px > 16_777_216 {
        return Err(AutomationError::BadRequest(format!("inpaint crop {cw}x{ch} exceeds 16 MP limit")));
    }

    let raw_mask = extract_crop_mask(doc, bounds.is_some(), Rect::new(cx0, cy0, cx1, cy1), target)?;
    let feathered_mask = feather_mask(&raw_mask, cw, ch, feather.min(64));

    let max_doc_side = doc.size.width.max(doc.size.height).clamp(1, 4096);
    let thumb = photocraft_compose::thumbnail(doc, max_doc_side);
    let crop_rgba = sample_crop_from_composite(
        &thumb.pixels,
        thumb.width,
        thumb.height,
        doc.size.width,
        doc.size.height,
        Rect::new(cx0, cy0, cx1, cy1),
    )?;
    let upload_rgba = apply_comfy_hole_alpha(&crop_rgba, &feathered_mask)?;
    let upload_img = Image::from_u8(cw, ch, ChannelLayout::Rgba, upload_rgba).map_err(|e| AutomationError::Other(e.to_string()))?;
    let upload_png = photocraft_codecs::encode(&upload_img, Format::Png, &Default::default()).map_err(|e| AutomationError::Other(e.to_string()))?;

    Ok(InpaintCrop { rect: [cx0, cy0, cw as i32, ch as i32], width: cw, height: ch, upload_png, feathered_mask })
}

fn resolve_inpaint_target(doc: &Document, bounds: Option<[i32; 4]>) -> Result<Rect, AutomationError> {
    if let Some([x, y, w, h]) = bounds {
        if w <= 0 || h <= 0 {
            return Err(AutomationError::BadRequest(format!("`bounds` width and height must be positive, got [{x}, {y}, {w}, {h}]")));
        }
        let doc_rect = Rect::new(0, 0, i32::try_from(doc.size.width).unwrap_or(i32::MAX), i32::try_from(doc.size.height).unwrap_or(i32::MAX));
        let clipped = Rect::new(x, y, x.saturating_add(w), y.saturating_add(h)).intersect(&doc_rect);
        if clipped.is_empty() {
            return Err(AutomationError::BadRequest("`bounds` lies completely outside the document".into()));
        }
        return Ok(clipped);
    }
    let Some(sel) = doc.selection.as_ref() else {
        return Err(AutomationError::BadRequest("ai_generative_fill requires an active selection or explicit `bounds` [x, y, width, height]".into()));
    };
    let doc_rect = Rect::new(0, 0, i32::try_from(doc.size.width).unwrap_or(i32::MAX), i32::try_from(doc.size.height).unwrap_or(i32::MAX));
    let r = sel.content_bounds().intersect(&doc_rect);
    if r.is_empty() {
        return Err(AutomationError::BadRequest("active selection is empty; pass `bounds` or create a non-empty selection".into()));
    }
    Ok(r)
}

fn extract_crop_mask(doc: &Document, from_explicit_bounds: bool, crop: Rect, target: Rect) -> Result<Vec<f32>, AutomationError> {
    let cw = crop.width() as usize;
    let ch = crop.height() as usize;
    let mut mask = vec![0.0f32; cw * ch];
    if !from_explicit_bounds && let Some(sel) = doc.selection.as_ref() {
        let samples = sel.try_read_region(crop).map_err(|e| AutomationError::Other(format!("selection read failed: {e}")))?;
        let stride = sel.channels().max(1);
        for (i, m) in mask.iter_mut().enumerate() {
            *m = samples.get(i * stride).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        }
        return Ok(mask);
    }
    for y in crop.y0..crop.y1 {
        let ry = (y - crop.y0) as usize;
        for x in crop.x0..crop.x1 {
            if x >= target.x0 && x < target.x1 && y >= target.y0 && y < target.y1 {
                let rx = (x - crop.x0) as usize;
                if let Some(slot) = mask.get_mut(ry * cw + rx) {
                    *slot = 1.0;
                }
            }
        }
    }
    Ok(mask)
}

fn feather_mask(src: &[f32], w: u32, h: u32, radius: u32) -> Vec<f32> {
    if radius == 0 || w == 0 || h == 0 {
        return src.to_vec();
    }
    let wi = w as i32;
    let hi = h as i32;
    let r = radius as i32;
    let mut horiz = vec![0.0f32; src.len()];
    let norm = (r * 2 + 1) as f32;
    for y in 0..hi {
        for x in 0..wi {
            let mut sum = 0.0f32;
            for dx in -r..=r {
                let sx = (x + dx).clamp(0, wi - 1) as usize;
                sum += src.get((y as usize) * (w as usize) + sx).copied().unwrap_or(0.0);
            }
            if let Some(dst) = horiz.get_mut((y as usize) * (w as usize) + (x as usize)) {
                *dst = sum / norm;
            }
        }
    }
    let mut out = vec![0.0f32; src.len()];
    for y in 0..hi {
        for x in 0..wi {
            let mut sum = 0.0f32;
            for dy in -r..=r {
                let sy = (y + dy).clamp(0, hi - 1) as usize;
                sum += horiz.get(sy * (w as usize) + (x as usize)).copied().unwrap_or(0.0);
            }
            if let Some(dst) = out.get_mut((y as usize) * (w as usize) + (x as usize)) {
                *dst = (sum / norm).clamp(0.0, 1.0);
            }
        }
    }
    out
}

fn sample_crop_from_composite(
    pixels: &[u8],
    comp_w: u32,
    comp_h: u32,
    doc_w: u32,
    doc_h: u32,
    crop: Rect,
) -> Result<Vec<u8>, AutomationError> {
    let cw = crop.width() as usize;
    let ch = crop.height() as usize;
    let mut out = vec![0u8; cw * ch * 4];
    let sx = f64::from(comp_w) / f64::from(doc_w.max(1));
    let sy = f64::from(comp_h) / f64::from(doc_h.max(1));

    for dy in 0..ch {
        let src_y = ((f64::from(crop.y0 + dy as i32) + 0.5) * sy).floor().clamp(0.0, (comp_h.saturating_sub(1)) as f64) as usize;
        for dx in 0..cw {
            let src_x = ((f64::from(crop.x0 + dx as i32) + 0.5) * sx).floor().clamp(0.0, (comp_w.saturating_sub(1)) as f64) as usize;
            let s_idx = (src_y * (comp_w as usize) + src_x) * 4;
            let d_idx = (dy * cw + dx) * 4;
            if let (Some(src), Some(dst)) = (pixels.get(s_idx..s_idx + 4), out.get_mut(d_idx..d_idx + 4)) {
                dst.copy_from_slice(src);
            }
        }
    }
    Ok(out)
}

fn apply_comfy_hole_alpha(rgba: &[u8], mask: &[f32]) -> Result<Vec<u8>, AutomationError> {
    if rgba.len() != mask.len() * 4 {
        return Err(AutomationError::Other("RGBA/mask length mismatch in crop".into()));
    }
    let mut out = rgba.to_vec();
    for (i, &m) in mask.iter().enumerate() {
        // ComfyUI's LoadImage derives its MASK output from (1.0 - alpha/255.0).
        if let Some(a) = out.get_mut(i * 4 + 3) {
            *a = ((1.0 - m.clamp(0.0, 1.0)) * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }
    Ok(out)
}

/// Request arguments for ComfyUI inpainting over a crop.
pub struct InpaintRunRequest {
    pub prompt: String,
    pub negative_prompt: String,
    pub denoise: f64,
    pub steps: u32,
    pub cfg_scale: f64,
    pub seed: Option<u64>,
    pub checkpoint: Option<String>,
    pub comfy_url: Option<String>,
    pub workflow: Option<Value>,
    pub timeout_ms: Option<u64>,
}

/// Upload crop to ComfyUI, execute inpaint workflow, fetch result PNG, and apply feathered patch alpha.
pub async fn run_inpaint_and_stitch(crop: &InpaintCrop, req: &InpaintRunRequest) -> Result<(Vec<u8>, Value), AutomationError> {
    if req.prompt.trim().is_empty() {
        return Err(AutomationError::BadRequest("`prompt` must not be empty".into()));
    }
    let url = resolve_comfy_url(req.comfy_url.as_deref());
    let endpoint = parse_http_endpoint(&url)?;
    let checkpoint = resolve_checkpoint(&endpoint, req.checkpoint.as_deref()).await;
    let seed = req.seed.unwrap_or_else(default_seed);
    let steps = req.steps.clamp(1, 150);
    let cfg_scale = req.cfg_scale.clamp(0.5, 30.0);
    let denoise = req.denoise.clamp(0.01, 1.0);

    let uploaded_name = http_upload_png(&endpoint, "photoforge_crop.png", &crop.upload_png, UPLOAD_TIMEOUT).await?;
    let workflow = match req.workflow.clone() {
        Some(mut custom) => {
            let vars = TemplateVars {
                prompt: &req.prompt,
                negative_prompt: &req.negative_prompt,
                checkpoint: &checkpoint,
                image: Some(&uploaded_name),
                width: Some(crop.width),
                height: Some(crop.height),
                steps,
                cfg: cfg_scale,
                seed,
                denoise,
            };
            substitute_workflow_vars(&mut custom, &vars, 0);
            custom
        }
        None => build_inpaint_workflow(&InpaintSpec {
            prompt: req.prompt.clone(),
            negative_prompt: req.negative_prompt.clone(),
            uploaded_image: uploaded_name.clone(),
            steps,
            cfg_scale,
            denoise,
            seed,
            checkpoint: checkpoint.clone(),
        }),
    };

    let timeout_ms = req.timeout_ms.unwrap_or(DEFAULT_RUN_TIMEOUT_MS).clamp(MIN_RUN_TIMEOUT_MS, MAX_RUN_TIMEOUT_MS);
    let generated_png = run_workflow_and_fetch_png(&endpoint, workflow, timeout_ms).await?;
    let stitched_png = build_stitched_patch_png(crop, &generated_png)?;
    let meta = json!({
        "comfy_url": url,
        "checkpoint": checkpoint,
        "uploaded_image": uploaded_name,
        "seed": seed,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "denoise": denoise,
        "crop_rect": crop.rect,
    });
    Ok((stitched_png, meta))
}

/// Decode a ComfyUI output PNG, resize to the crop size if needed, and mask alpha to `feathered_mask`.
pub fn build_stitched_patch_png(crop: &InpaintCrop, generated_png: &[u8]) -> Result<Vec<u8>, AutomationError> {
    let img = photocraft_codecs::decode(generated_png).map_err(|e| AutomationError::App(format!("ComfyUI output decode failed: {e}")))?;
    let (gw, gh) = img.dimensions();
    let rgba = img.to_rgba8();
    let mut resized = if gw == crop.width && gh == crop.height { rgba } else { resize_rgba8_bilinear(&rgba, gw, gh, crop.width, crop.height)? };
    for (i, &m) in crop.feathered_mask.iter().enumerate() {
        if let Some(a) = resized.get_mut(i * 4 + 3) {
            let gen_alpha = f32::from(*a) / 255.0;
            *a = (gen_alpha * m.clamp(0.0, 1.0) * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }
    let out_img = Image::from_u8(crop.width, crop.height, ChannelLayout::Rgba, resized).map_err(|e| AutomationError::Other(e.to_string()))?;
    photocraft_codecs::encode(&out_img, Format::Png, &Default::default()).map_err(|e| AutomationError::Other(e.to_string()))
}

async fn run_workflow_and_fetch_png(endpoint: &Endpoint, workflow: Value, timeout_ms: u64) -> Result<Vec<u8>, AutomationError> {
    let payload = json!({ "prompt": workflow, "client_id": "photoforge-mcp" });
    let queued = http_post_json(endpoint, "/prompt", &payload, PROMPT_TIMEOUT).await?;
    let prompt_id = queued
        .get("prompt_id")
        .and_then(Value::as_str)
        .ok_or_else(|| AutomationError::App(format!("ComfyUI /prompt did not return `prompt_id`: {queued}")))?
        .to_owned();

    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    let history_path = format!("/history/{}", percent_encode(&prompt_id));
    loop {
        if Instant::now() >= deadline {
            return Err(AutomationError::App(format!("ComfyUI prompt `{prompt_id}` timed out after {timeout_ms} ms")));
        }
        let history = http_get_json(endpoint, &history_path, PROMPT_TIMEOUT).await?;
        if let Some(entry) = extract_history_entry(&history, &prompt_id) {
            if let Some(err) = check_history_error(entry) {
                return Err(err);
            }
            if let Some(view_path) = find_first_output_view_path(entry) {
                return http_get_bytes(endpoint, &view_path, UPLOAD_TIMEOUT).await;
            }
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

fn extract_history_entry<'a>(history: &'a Value, prompt_id: &str) -> Option<&'a Value> {
    if let Some(entry) = history.get(prompt_id) {
        return Some(entry);
    }
    if history.get("outputs").is_some() {
        return Some(history);
    }
    None
}

fn check_history_error(entry: &Value) -> Option<AutomationError> {
    let status = entry.get("status")?;
    if status.get("status_str").and_then(Value::as_str) == Some("error") {
        return Some(AutomationError::App(format!("ComfyUI workflow error: {status}")));
    }
    None
}

fn find_first_output_view_path(entry: &Value) -> Option<String> {
    let outputs = entry.get("outputs")?.as_object()?;
    let mut keys: Vec<&String> = outputs.keys().collect();
    keys.sort();
    for k in keys {
        let Some(images) = outputs.get(k).and_then(|n| n.get("images")).and_then(Value::as_array) else {
            continue;
        };
        let Some(first) = images.first() else { continue };
        let Some(filename) = first.get("filename").and_then(Value::as_str) else { continue };
        let subfolder = first.get("subfolder").and_then(Value::as_str).unwrap_or("");
        let kind = first.get("type").and_then(Value::as_str).unwrap_or("output");
        return Some(format!(
            "/view?filename={}&subfolder={}&type={}",
            percent_encode(filename),
            percent_encode(subfolder),
            percent_encode(kind)
        ));
    }
    None
}

fn percent_encode(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for byte in input.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            out.push(char::from(byte));
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

/// Fit an encoded PNG to `target_rect` if provided, place it non-destructively in `Session`, and rename the layer.
pub fn place_png_as_layer(
    session: &mut Session,
    layer_name: &str,
    png_bytes: Vec<u8>,
    target_rect: Option<[i32; 4]>,
    smart_object: bool,
) -> Result<Value, AutomationError> {
    let (final_png, opts) = prepare_place_payload(png_bytes, target_rect, smart_object)?;
    let safe_name = sanitize_layer_name(layer_name);
    let mut placed = photocraft_engine::file_cmds::place_bytes(session, "photoforge_generated.png", final_png, None, &opts)?;
    if let Some(layer_id) = placed.get("layer").and_then(Value::as_u64) {
        let _ = session.execute("layer.renameLayer", json!({ "layer": layer_id, "name": safe_name }));
        if let Some(obj) = placed.as_object_mut() {
            obj.insert("name".into(), json!(safe_name));
        }
    }
    Ok(placed)
}

fn prepare_place_payload(png_bytes: Vec<u8>, target_rect: Option<[i32; 4]>, smart_object: bool) -> Result<(Vec<u8>, Value), AutomationError> {
    let Some([x, y, w, h]) = target_rect else {
        return Ok((png_bytes, json!({ "smartObject": smart_object, "frame": false })));
    };
    if w <= 0 || h <= 0 {
        return Err(AutomationError::BadRequest(format!("placement rectangle width/height must be positive, got [{x}, {y}, {w}, {h}]")));
    }
    let tw = u32::try_from(w).map_err(|_| AutomationError::BadRequest("invalid target width".into()))?;
    let th = u32::try_from(h).map_err(|_| AutomationError::BadRequest("invalid target height".into()))?;
    let decoded = photocraft_codecs::decode(&png_bytes).map_err(|e| AutomationError::App(format!("PNG decode failed before place: {e}")))?;
    let (iw, ih) = decoded.dimensions();
    let final_png = if iw == tw && ih == th {
        png_bytes
    } else {
        let rgba = decoded.to_rgba8();
        let resized = resize_rgba8_bilinear(&rgba, iw, ih, tw, th)?;
        let out_img = Image::from_u8(tw, th, ChannelLayout::Rgba, resized).map_err(|e| AutomationError::Other(e.to_string()))?;
        photocraft_codecs::encode(&out_img, Format::Png, &Default::default()).map_err(|e| AutomationError::Other(e.to_string()))?
    };
    let cx = f64::from(x) + f64::from(w) / 2.0;
    let cy = f64::from(y) + f64::from(h) / 2.0;
    let opts = json!({
        "fit": false,
        "scale": 100.0,
        "center": [cx, cy],
        "smartObject": smart_object,
        "frame": false,
    });
    Ok((final_png, opts))
}

fn sanitize_layer_name(name: &str) -> String {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return "AI Generated".into();
    }
    trimmed.chars().take(64).collect()
}

fn resize_rgba8_bilinear(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Result<Vec<u8>, AutomationError> {
    let need_src = (sw as usize).checked_mul(sh as usize).and_then(|p| p.checked_mul(4)).ok_or_else(|| AutomationError::BadRequest("source resize overflow".into()))?;
    let need_dst = (dw as usize).checked_mul(dh as usize).and_then(|p| p.checked_mul(4)).ok_or_else(|| AutomationError::BadRequest("target resize overflow".into()))?;
    if src.len() < need_src || sw == 0 || sh == 0 || dw == 0 || dh == 0 {
        return Err(AutomationError::BadRequest("invalid dimensions for image resize".into()));
    }
    let mut dst = vec![0u8; need_dst];
    for dy in 0..dh {
        let gy = if dh <= 1 { 0.0 } else { (f64::from(dy) / f64::from(dh - 1)) * f64::from(sh - 1) };
        let y0 = (gy.floor() as u32).min(sh - 1);
        let y1 = (y0 + 1).min(sh - 1);
        let wy = gy - f64::from(y0);
        for dx in 0..dw {
            let gx = if dw <= 1 { 0.0 } else { (f64::from(dx) / f64::from(dw - 1)) * f64::from(sw - 1) };
            let x0 = (gx.floor() as u32).min(sw - 1);
            let x1 = (x0 + 1).min(sw - 1);
            let wx = gx - f64::from(x0);
            let p00 = read_rgba_at(src, sw, x0, y0);
            let p10 = read_rgba_at(src, sw, x1, y0);
            let p01 = read_rgba_at(src, sw, x0, y1);
            let p11 = read_rgba_at(src, sw, x1, y1);
            let out_idx = ((dy as usize) * (dw as usize) + (dx as usize)) * 4;
            if let Some(px) = dst.get_mut(out_idx..out_idx + 4) {
                for c in 0..4 {
                    let top = f64::from(p00[c]) * (1.0 - wx) + f64::from(p10[c]) * wx;
                    let bot = f64::from(p01[c]) * (1.0 - wx) + f64::from(p11[c]) * wx;
                    px[c] = (top * (1.0 - wy) + bot * wy).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    Ok(dst)
}

fn read_rgba_at(src: &[u8], w: u32, x: u32, y: u32) -> [u8; 4] {
    let idx = ((y as usize) * (w as usize) + (x as usize)) * 4;
    match src.get(idx..idx + 4) {
        Some(&[r, g, b, a]) => [r, g, b, a],
        _ => [0, 0, 0, 0],
    }
}

fn default_seed() -> u64 {
    let mut bytes = [0u8; 8];
    let _ = getrandom::fill(&mut bytes);
    u64::from_le_bytes(bytes) & 0x000f_ffff_ffff_ffff
}

/// Normalize JSON value map keys in sorted order.
pub fn sorted_workflow_keys(v: &Value) -> Vec<String> {
    v.as_object()
        .map(|m| {
            let mut keys: Vec<String> = m.keys().cloned().collect();
            keys.sort();
            keys
        })
        .unwrap_or_default()
}
