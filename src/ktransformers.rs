use crate::{
    Error, GenerationMode, Result,
    backend::{
        Capabilities, Completion, CompletionRequest, GenerationInput, RequestBackend,
        RequestBackendInfo,
    },
};
use reqwest::{Method, Url, blocking::Client, redirect::Policy};
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::{Value, json};
use std::{
    io::Read,
    net::IpAddr,
    time::{Duration, Instant},
};

const MAX_RESPONSE_BYTES: u64 = 8 * 1024 * 1024;
pub const API_SOURCE_REVISION: &str = "2c4085a01cf75cde927d35a259b4adaa7b07b3ab";

pub struct KTransformersConfig {
    pub endpoint: String,
    pub expected_model: String,
    pub expected_method: String,
    pub timeout: Duration,
    pub allow_expert_deferral: bool,
}

pub struct KTransformersBackend {
    client: Client,
    endpoint: Url,
    info: RequestBackendInfo,
}

#[derive(Deserialize)]
struct ServerInfo {
    served_model_name: String,
    model_path: String,
    version: String,
    kt_method: Option<String>,
    kt_weight_path: Option<String>,
    kt_cpuinfer: Option<u32>,
    kt_num_gpu_experts: Option<u32>,
    kt_max_deferred_experts_per_token: Option<u32>,
    context_length: Option<u32>,
    disable_radix_cache: bool,
    skip_tokenizer_init: bool,
}

#[derive(Deserialize)]
struct ModelIdentity {
    model_path: String,
    tokenizer_path: Option<String>,
    model_type: Option<String>,
    weight_version: String,
    is_generation: bool,
}

#[derive(Deserialize)]
struct GenerateResponse {
    text: String,
    output_ids: Vec<i32>,
    meta_info: GenerationMetadata,
}

#[derive(Deserialize)]
struct GenerationMetadata {
    prompt_tokens: u32,
    completion_tokens: u32,
    cached_tokens: u32,
    weight_version: String,
    finish_reason: FinishReason,
}

#[derive(Deserialize, Debug)]
struct FinishReason {
    #[serde(rename = "type")]
    kind: String,
    message: Option<String>,
}

fn backend_error(message: impl Into<String>) -> Error {
    Error::Backend(format!("KTransformers: {}", message.into()))
}

fn endpoint_url(endpoint: &str) -> Result<Url> {
    let mut url =
        Url::parse(endpoint).map_err(|error| Error::Invalid(format!("endpoint: {error}")))?;
    if url.scheme() != "http"
        || !url.username().is_empty()
        || url.password().is_some()
        || url.path() != "/"
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(Error::Invalid(
            "endpoint must be a plain local HTTP origin, without credentials, path, query, or fragment".into(),
        ));
    }
    let host = url
        .host_str()
        .ok_or_else(|| Error::Invalid("endpoint has no host".into()))?;
    if host == "localhost" {
        url.set_host(Some("127.0.0.1"))
            .map_err(|error| Error::Invalid(error.to_string()))?;
    } else if !host
        .trim_matches(['[', ']'])
        .parse::<IpAddr>()
        .is_ok_and(|ip| ip.is_loopback())
    {
        return Err(Error::Invalid(
            "only loopback KTransformers endpoints are supported".into(),
        ));
    }
    Ok(url)
}

fn exchange<T: DeserializeOwned>(
    client: &Client,
    endpoint: &Url,
    method: Method,
    path: &str,
    payload: Option<&Value>,
) -> Result<T> {
    let url = endpoint
        .join(path)
        .map_err(|error| backend_error(error.to_string()))?;
    let mut request = client.request(method, url);
    if let Some(payload) = payload {
        request = request.json(payload);
    }
    let response = request.send().map_err(|error| {
        let kind = if error.is_timeout() {
            "request timed out"
        } else {
            "request failed"
        };
        backend_error(format!(
            "{path}: {kind}: {error}; generation is not automatically retried"
        ))
    })?;
    let status = response.status();
    let mut bytes = Vec::new();
    response
        .take(MAX_RESPONSE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| backend_error(format!("{path}: reading response: {error}")))?;
    if bytes.len() as u64 > MAX_RESPONSE_BYTES {
        return Err(backend_error(format!("{path}: response exceeds 8 MiB")));
    }
    if !status.is_success() {
        let detail = String::from_utf8_lossy(&bytes)
            .chars()
            .take(1024)
            .collect::<String>();
        return Err(backend_error(format!("{path}: HTTP {status}: {detail}")));
    }
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|error| backend_error(format!("{path}: invalid JSON: {error}")))?;
    if let Some(error) = value.get("error") {
        return Err(backend_error(format!("{path}: server error: {error}")));
    }
    serde_json::from_value(value)
        .map_err(|error| backend_error(format!("{path}: unsupported response schema: {error}")))
}

impl KTransformersBackend {
    pub fn connect(config: KTransformersConfig) -> Result<Self> {
        if config.expected_model.trim().is_empty()
            || config.expected_method.trim().is_empty()
            || config.timeout.is_zero()
        {
            return Err(Error::Invalid(
                "model, KT method, and nonzero timeout are required".into(),
            ));
        }
        let endpoint = endpoint_url(&config.endpoint)?;
        let client = Client::builder()
            .no_proxy()
            .redirect(Policy::none())
            .retry(reqwest::retry::never())
            .connect_timeout(config.timeout.min(Duration::from_secs(10)))
            .timeout(config.timeout)
            .build()
            .map_err(|error| backend_error(error.to_string()))?;
        let server: ServerInfo = exchange(&client, &endpoint, Method::GET, "server_info", None)?;
        let model: ModelIdentity = exchange(&client, &endpoint, Method::GET, "model_info", None)?;
        let method = server
            .kt_method
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| backend_error("server does not report a configured kt_method"))?;
        let weights = server
            .kt_weight_path
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| backend_error("server does not report a KT CPU weight path"))?;
        let threads = server
            .kt_cpuinfer
            .filter(|&n| n > 0)
            .ok_or_else(|| backend_error("server does not report positive kt_cpuinfer"))?;
        let deferred = server.kt_max_deferred_experts_per_token.ok_or_else(|| {
            backend_error("set --kt-max-deferred-experts-per-token explicitly (0 for baseline)")
        })?;
        if deferred > 0 && !config.allow_expert_deferral {
            return Err(backend_error(
                "expert deferral requires explicit allow_expert_deferral opt-in",
            ));
        }
        if server.served_model_name != config.expected_model
            || !method.eq_ignore_ascii_case(&config.expected_method)
        {
            return Err(backend_error(format!(
                "server identity mismatch: model={}, method={method}",
                server.served_model_name
            )));
        }
        if model.model_path != server.model_path
            || model.model_path.is_empty()
            || model.weight_version.is_empty()
            || server.version.is_empty()
            || !model.is_generation
            || server.skip_tokenizer_init
            || server.context_length == Some(0)
        {
            return Err(backend_error(
                "inconsistent model identity, context, or missing generation/tokenizer support",
            ));
        }
        let info = RequestBackendInfo {
            backend: "ktransformers-sglang-http".into(),
            endpoint: endpoint.to_string(),
            model: server.served_model_name,
            model_path: server.model_path,
            tokenizer_path: model.tokenizer_path,
            model_type: model.model_type,
            weight_version: model.weight_version,
            server_version: server.version,
            kernel_method: method,
            cpu_weight_path: weights,
            cpu_threads: threads,
            num_gpu_experts: server.kt_num_gpu_experts,
            deferred_experts_per_token: deferred,
            context_tokens: server.context_length,
            prefix_cache_enabled: !server.disable_radix_cache,
            capabilities: Capabilities {
                token_step: false,
                chunked_prefill: false,
            },
        };
        Ok(Self {
            client,
            endpoint,
            info,
        })
    }
}

impl RequestBackend for KTransformersBackend {
    fn info(&self) -> &RequestBackendInfo {
        &self.info
    }

    fn generate(&mut self, request: &CompletionRequest) -> Result<Completion> {
        if request.max_tokens == 0 || request.max_tokens > i32::MAX as u32 {
            return Err(Error::Invalid("max_tokens must be in 1..=i32::MAX".into()));
        }
        let fixed = request.mode == GenerationMode::FixedLength;
        let mut payload = json!({
            "stream": false,
            "return_logprob": false,
            "sampling_params": {
                "temperature": 0.0, "top_k": 1, "top_p": 1.0,
                "max_new_tokens": request.max_tokens, "ignore_eos": fixed,
                "min_new_tokens": 0, "frequency_penalty": 0.0,
                "presence_penalty": 0.0, "repetition_penalty": 1.0,
                "stop": [], "stop_token_ids": []
            }
        });
        let input_count = match &request.input {
            GenerationInput::Text(text) if !text.is_empty() => {
                payload["text"] = json!(text);
                None
            }
            GenerationInput::TokenIds(tokens)
                if !tokens.is_empty() && tokens.iter().all(|&t| t >= 0) =>
            {
                payload["input_ids"] = json!(tokens);
                Some(tokens.len() as u64)
            }
            _ => {
                return Err(Error::Invalid(
                    "nonempty text or nonnegative input token IDs are required".into(),
                ));
            }
        };
        if let Some(context) = self.info.context_tokens {
            if u64::from(request.max_tokens) > u64::from(context)
                || input_count
                    .is_some_and(|n| n + u64::from(request.max_tokens) > u64::from(context))
            {
                return Err(Error::Invalid(
                    "request exceeds reported server context".into(),
                ));
            }
        }
        let start = Instant::now();
        let response: GenerateResponse = exchange(
            &self.client,
            &self.endpoint,
            Method::POST,
            "generate",
            Some(&payload),
        )?;
        let meta = response.meta_info;
        if meta.finish_reason.kind != "stop" && meta.finish_reason.kind != "length" {
            return Err(backend_error(format!(
                "generation ended with {}: {}",
                meta.finish_reason.kind,
                meta.finish_reason
                    .message
                    .as_deref()
                    .unwrap_or("no server message")
            )));
        }
        if meta.weight_version != self.info.weight_version {
            return Err(backend_error(
                "server weights changed; reconnect before generating again",
            ));
        }
        if response.output_ids.is_empty()
            || response.output_ids.iter().any(|&t| t < 0)
            || response.output_ids.len() as u64 != u64::from(meta.completion_tokens)
            || meta.completion_tokens > request.max_tokens
            || meta.prompt_tokens == 0
            || meta.cached_tokens > meta.prompt_tokens
            || self.info.context_tokens.is_some_and(|capacity| {
                u64::from(meta.prompt_tokens) + u64::from(meta.completion_tokens)
                    > u64::from(capacity)
            })
            || input_count.is_some_and(|n| n != u64::from(meta.prompt_tokens))
            || ((fixed || meta.finish_reason.kind == "length")
                && meta.completion_tokens != request.max_tokens)
        {
            return Err(backend_error(
                "incomplete generation or inconsistent token accounting",
            ));
        }
        Ok(Completion {
            text: response.text,
            generated_tokens: response.output_ids,
            prompt_tokens: meta.prompt_tokens,
            cached_tokens: meta.cached_tokens,
            finish_reason: meta.finish_reason.kind,
            elapsed_ns: u64::try_from(start.elapsed().as_nanos())
                .map_err(|_| backend_error("request duration overflow"))?,
        })
    }
}
