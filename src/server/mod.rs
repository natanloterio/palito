//! `palito serve`: an OpenAI-compatible HTTP server.
//!
//! Routes: POST /v1/chat/completions, POST /v1/completions (both with
//! `"stream": true` as server-sent events), GET /v1/models and GET /health.
//! The model is opened once and runs one request at a time on a worker thread;
//! other requests wait in a bounded queue (503 when full).

mod openai;
mod reasoning;
mod stream;
mod worker;

use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, Request, State},
    http::{HeaderMap, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use crate::chat_repl::SamplingArgs;
use clap::{Args, ValueEnum};
use openai::{ChatRequest, CompletionRequest, Usage, error_body};
use palito::{
    engine::{Completion, Device, EngineOptions, GenerationOptions},
    sampling::Sampling,
};
use serde_json::{Value, json};
use std::{net::IpAddr, path::PathBuf, sync::Arc, time::Instant};
use stream::Kind;
use worker::{Event, ModelReport, Prompt, Reply, SubmitError, Worker};

/// Request bodies above this are refused (a 65k-token prompt is well below it).
const BODY_LIMIT_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Copy, ValueEnum)]
enum ServeBackend {
    Cpu,
    Cuda,
}

#[derive(Args)]
pub(crate) struct ServeArgs {
    #[arg(long, env = "PALITO_MODEL")]
    model: PathBuf,
    /// Address to listen on. Use 0.0.0.0 to accept other machines (set --api-key).
    #[arg(long, default_value = "127.0.0.1")]
    host: IpAddr,
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// [default: cuda when built with CUDA, else cpu]
    #[arg(long, env = "PALITO_BACKEND", value_enum)]
    backend: Option<ServeBackend>,
    /// Context tokens [default: the model's trained context, fitted to GPU memory].
    #[arg(long)]
    context: Option<u32>,
    #[arg(long, default_value_t = 4)]
    threads: u32,
    #[arg(long, default_value_t = 0)]
    gpu_device: usize,
    #[arg(long)]
    gpu_memory_limit_bytes: Option<u64>,
    /// [default: $PALITO_CUDA_KERNEL_CACHE or ~/.cache/palito/kernels]
    #[arg(long, env = "PALITO_CUDA_KERNEL_CACHE")]
    cuda_kernel_cache: Option<PathBuf>,
    /// Require `Authorization: Bearer <key>` on /v1 routes.
    #[arg(long, env = "PALITO_API_KEY", hide_env_values = true)]
    api_key: Option<String>,
    /// Re-evaluate every prompt in full instead of reusing the previous
    /// request's KV cache for a shared prompt start (CUDA).
    #[arg(long)]
    no_prefix_cache: bool,
    /// Requests allowed to wait while one is running; more get 503.
    #[arg(long, default_value_t = 16)]
    queue: usize,
    /// Model id reported by /v1/models [default: the GGUF file name without extension].
    #[arg(long)]
    model_name: Option<String>,
    /// Defaults for requests that omit temperature, top_p, top_k or seed.
    #[command(flatten)]
    sampling: SamplingArgs,
}

#[derive(Clone)]
struct AppState {
    worker: Worker,
    model: Arc<str>,
    report: Arc<ModelReport>,
    api_key: Option<Arc<str>>,
    created: u64,
    sampling: Sampling,
}

pub(crate) fn run(args: ServeArgs) -> palito::Result<()> {
    let options = EngineOptions {
        device: match args.backend {
            Some(ServeBackend::Cpu) => Device::Cpu,
            Some(ServeBackend::Cuda) => Device::Cuda,
            None => EngineOptions::default().device,
        },
        context_tokens: args.context,
        threads: args.threads,
        gpu_device: args.gpu_device,
        gpu_memory_limit_bytes: args.gpu_memory_limit_bytes,
        kernel_cache_dir: args.cuda_kernel_cache.clone(),
        prefix_cache: !args.no_prefix_cache,
        ..EngineOptions::default()
    };
    let sampling = args.sampling.sampling();
    sampling.validate()?;
    let model = args.model_name.clone().unwrap_or_else(|| {
        args.model
            .file_stem()
            .map_or("model".into(), |stem| stem.to_string_lossy().into_owned())
    });
    let start = Instant::now();
    let (worker, report) = Worker::start(args.model.clone(), options, args.queue.max(1))?;
    eprintln!(
        "Loaded {model} in {:.1}s ({}, context {} tokens).",
        start.elapsed().as_secs_f64(),
        report.profile,
        report.context_tokens
    );
    let api_key = args.api_key.filter(|key| !key.is_empty());
    if !args.host.is_loopback() && api_key.is_none() {
        eprintln!(
            "warning: listening on {} without --api-key; anyone who can reach this port can use the model",
            args.host
        );
    }
    let state = AppState {
        worker,
        model: model.into(),
        report: Arc::new(report),
        api_key: api_key.map(Into::into),
        created: openai::unix_seconds(),
        sampling,
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(serve(state, args.host, args.port))
}

async fn serve(state: AppState, host: IpAddr, port: u16) -> palito::Result<()> {
    let v1 = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(models))
        .route_layer(middleware::from_fn_with_state(state.clone(), require_key));
    let app = Router::new()
        .route("/health", get(health))
        .merge(v1)
        .layer(DefaultBodyLimit::max(BODY_LIMIT_BYTES))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind((host, port)).await?;
    let address = listener.local_addr()?;
    eprintln!("Serving the OpenAI API at http://{address}/v1 (Ctrl-C stops).");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

async fn require_key(State(state): State<AppState>, request: Request, next: Next) -> Response {
    let Some(key) = &state.api_key else {
        return next.run(request).await;
    };
    let given = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "));
    if given.is_some_and(|given| same_secret(given.as_bytes(), key.as_bytes())) {
        next.run(request).await
    } else {
        failure(StatusCode::UNAUTHORIZED, "invalid or missing API key", "authentication_error")
    }
}

/// Comparison whose time does not depend on where the inputs differ.
fn same_secret(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

async fn health(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "model": &*state.model,
        "profile": state.report.profile,
        "context_tokens": state.report.context_tokens,
    }))
}

async fn models(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": &*state.model,
            "object": "model",
            "created": state.created,
            "owned_by": "palito",
            "context_length": state.report.context_tokens,
        }],
    }))
}

async fn chat_completions(State(state): State<AppState>, body: Bytes) -> Response {
    let request: ChatRequest = match parse(&body) {
        Ok(request) => request,
        Err(response) => return response,
    };
    if !state.report.has_chat_template {
        return invalid("this model has no chat template; use /v1/completions");
    }
    let parsed = request.check_supported().and_then(|()| {
        Ok((
            request.messages()?,
            request.sampling.resolve(&state.sampling)?,
            request.stop()?,
            request.json_schema()?,
        ))
    });
    let (messages, sampling, stop, json_schema) = match parsed {
        Ok(parsed) => parsed,
        Err(message) => return invalid(&message),
    };
    let options = GenerationOptions {
        max_tokens: request.max_tokens(),
        enable_thinking: request.enable_thinking(),
        sampling,
        stop,
        json_schema,
    };
    let reply = Respond {
        kind: Kind::Chat,
        stream: request.stream,
        include_usage: request.stream_options.as_ref().is_some_and(|o| o.include_usage),
        split_reasoning: request.split_reasoning(),
    };
    respond(state, reply, Prompt::Chat(messages), options).await
}

async fn completions(State(state): State<AppState>, body: Bytes) -> Response {
    let request: CompletionRequest = match parse(&body) {
        Ok(request) => request,
        Err(response) => return response,
    };
    let parsed = request.check_supported().and_then(|()| {
        Ok((
            request.prompt()?,
            request.sampling.resolve(&state.sampling)?,
            request.stop()?,
            request.json_schema()?,
        ))
    });
    let (prompt, sampling, stop, json_schema) = match parsed {
        Ok(parsed) => parsed,
        Err(message) => return invalid(&message),
    };
    let options = GenerationOptions {
        max_tokens: Some(request.max_tokens.unwrap_or(openai::COMPLETION_DEFAULT_MAX_TOKENS)),
        enable_thinking: true,
        sampling,
        stop,
        json_schema,
    };
    let reply = Respond {
        kind: Kind::Text,
        stream: request.stream,
        include_usage: request.stream_options.as_ref().is_some_and(|o| o.include_usage),
        split_reasoning: false,
    };
    respond(state, reply, Prompt::Text(prompt), options).await
}

/// How to answer a request.
struct Respond {
    kind: Kind,
    stream: bool,
    include_usage: bool,
    split_reasoning: bool,
}

async fn respond(
    state: AppState,
    reply: Respond,
    prompt: Prompt,
    options: GenerationOptions,
) -> Response {
    let Respond {
        kind,
        stream,
        include_usage,
        split_reasoning,
    } = reply;
    let mut events = match state.worker.submit(prompt, options, split_reasoning) {
        Ok(events) => events,
        Err(SubmitError::Busy) => {
            let mut response = failure(
                StatusCode::SERVICE_UNAVAILABLE,
                "server busy: too many queued requests",
                "server_error",
            );
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, header::HeaderValue::from_static("1"));
            return response;
        }
        Err(SubmitError::Stopped) => {
            return failure(StatusCode::INTERNAL_SERVER_ERROR, "engine stopped", "server_error");
        }
    };
    let id = openai::response_id(kind.id_prefix());
    if stream {
        return stream::sse(kind, id, state.model.clone(), events, include_usage);
    }
    let reply = loop {
        match events.recv().await {
            Some(Event::Text { .. }) => continue,
            Some(Event::Done(result)) => break result,
            None => break Err("engine stopped".into()),
        }
    };
    match reply {
        Ok(reply) => {
            log(kind, &reply.completion);
            Json(final_body(kind, &id, &state.model, &reply)).into_response()
        }
        Err(message) => invalid(&message),
    }
}

fn final_body(kind: Kind, id: &str, model: &str, reply: &Reply) -> Value {
    let completion = &reply.completion;
    let finish = openai::finish_reason(completion.finish_reason);
    let choice = match kind {
        Kind::Chat => {
            let mut message = json!({"role": "assistant", "content": reply.content});
            if let Some(reasoning) = &reply.reasoning {
                message["reasoning_content"] = json!(reasoning);
            }
            json!({
                "index": 0,
                "message": message,
                "finish_reason": finish,
                "logprobs": null,
            })
        }
        Kind::Text => json!({
            "index": 0,
            "text": reply.content,
            "finish_reason": finish,
            "logprobs": null,
        }),
    };
    json!({
        "id": id,
        "object": kind.object(),
        "created": openai::unix_seconds(),
        "model": model,
        "choices": [choice],
        "usage": Usage::new(completion.prompt_tokens, completion.generated_tokens),
        "timings": timings(completion),
    })
}

fn timings(completion: &Completion) -> Value {
    json!({
        "ttft_ms": completion.ttft_ns as f64 / 1e6,
        "tokens_per_second": completion.tokens_per_second,
    })
}

fn log(kind: Kind, completion: &Completion) {
    eprintln!(
        "{}: {} prompt + {} generated tokens, first token {:.0} ms, {:.1} tok/s, {:?}",
        kind.object(),
        completion.prompt_tokens,
        completion.generated_tokens,
        completion.ttft_ns as f64 / 1e6,
        completion.tokens_per_second,
        completion.finish_reason
    );
}

fn parse<T: serde::de::DeserializeOwned>(body: &[u8]) -> Result<T, Response> {
    serde_json::from_slice(body).map_err(|error| invalid(&format!("invalid JSON body: {error}")))
}

fn invalid(message: &str) -> Response {
    failure(StatusCode::BAD_REQUEST, message, "invalid_request_error")
}

fn failure(status: StatusCode, message: &str, kind: &str) -> Response {
    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("application/json"),
    );
    (status, headers, error_body(message, kind).to_string()).into_response()
}
