# Serving real model responses

Nexus keeps two jobs separate: the TypeScript from-scratch model is a research project, while the playground can use a pretrained open-weight instruction model to produce useful responses today. The inference endpoint prefers a configured OpenAI-compatible provider and does not silently fall back to the tiny research checkpoint when a hosted provider fails.

## Option A: Hugging Face Inference Providers

1. Create a Hugging Face user access token with **Inference Providers** permission.
2. In Vercel, open the Nexus project settings, add the secret environment variable `HF_TOKEN` for the target environment, and redeploy after saving it.
3. The default model is `Qwen/Qwen2.5-7B-Instruct-1M:fastest`. To change it, set `NEXUS_LLM_MODEL` to a model available from Hugging Face Inference Providers, such as `Qwen/Qwen2.5-7B-Instruct-1M:fastest`.
4. Visit the playground and ask a complete question. The returned response identifies the selected model and provider.

The router endpoint is `https://router.huggingface.co/v1/chat/completions`; Nexus uses the OpenAI-compatible messages format. The default model's Hub card declares Apache-2.0: [model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-1M). Provider availability, latency, credits, and rate limits depend on the active Hugging Face account and provider. A configured token is required; adding this code does not create or fund a Hugging Face account.

**Keep the token server-side.** Do not prefix it with `NEXT_PUBLIC_`, put it in client code, or commit it to Git. The endpoint has bounded request sizes, bounded conversation history, explicit upstream error handling, and a server-side timeout.

## Option B: self-host a model

Nexus supports servers exposing the OpenAI-compatible `POST /v1/chat/completions` endpoint. Examples include [llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server), [vLLM](https://github.com/vllm-project/vllm), and [SGLang](https://github.com/sgl-project/sglang).

Configure these server-side environment variables:

| Variable | Purpose |
|---|---|
| `NEXUS_LLM_BASE_URL` | Base URL, including `/v1`, e.g. `https://your-private-llm.example/v1` |
| `NEXUS_LLM_MODEL` | Exact model identifier expected by your server |
| `NEXUS_LLM_API_KEY` | Optional API key for the configured endpoint, if it requires one |

A loopback development example is `NEXUS_LLM_BASE_URL=http://127.0.0.1:8080/v1`. Use HTTPS for non-loopback endpoints. Keep private inference servers behind authentication or a private network; do not expose an unauthenticated model server to the public internet. For llama.cpp, the model name can be the name reported by its `/v1/models` route or the identifier expected by that server.

If a custom endpoint is configured, Nexus does not forward `HF_TOKEN` to it. Only `NEXUS_LLM_API_KEY` is considered for custom endpoints, preventing a Hugging Face token from being leaked to another host.

## Experimental from-scratch checkpoint

To test a research checkpoint deliberately, provision a checkpoint and its trainer-generated `.sha256` sidecar, then set:

- `NEXUS_TRANSFORMER_CHECKPOINT`: absolute runtime path to the checkpoint.
- `NEXUS_ENABLE_EXPERIMENTAL_TRANSFORMER=true`: explicit opt-in to serve it when no pretrained provider is configured.

This checkpoint is not used if a pretrained provider is configured. The tiny model has produced incoherent samples in early experiments, so it is disabled by default. Do not enable it as the general assistant path.

## Request format

The playground currently sends `{ "prompt": "...", "maxTokens": 160 }`. The API also accepts a bounded chat history:

```json
{
  "messages": [
    { "role": "user", "content": "Explain photosynthesis simply." },
    { "role": "assistant", "content": "Plants use light energy to make sugars." },
    { "role": "user", "content": "Where does the carbon come from?" }
  ],
  "maxTokens": 256
}
```

Messages are limited to user/assistant roles, at most 12 per request, and the final message must be from the user. The server adds its own system instruction. Output length is capped at 512 tokens. The endpoint returns clear error codes for missing configuration, provider authentication, exhausted budget, rate limiting, timeouts, and malformed responses.

## Research references

Nexus's practical inference path follows established interfaces rather than inventing a model-serving protocol:

- Hugging Face [Inference Providers chat-completion API](https://huggingface.co/docs/inference-providers/tasks/chat-completion)
- Hugging Face Transformers [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [llama.cpp OpenAI-compatible server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
- [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang)

The pretraining/fine-tuning pipeline remains separate from inference. Small from-scratch experiments do not substitute for pretrained weights, and a successful request does not itself prove factual correctness.
