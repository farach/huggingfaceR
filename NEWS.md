# huggingfaceR 2.2.0

This release re-aligns the package with the current Hugging Face Inference
Providers platform. Since mid-2025 the first-party `hf-inference` provider has
narrowed to CPU-friendly classic models, and several models the package shipped
as defaults are no longer served by it.

## Scope of task-based inference

huggingfaceR's task functions (`hf_classify()`, `hf_summarize()`, `hf_embed()`,
`hf_text_to_image()` and friends) speak the Hugging Face **task API contract**:
`POST .../models/{model}` with an `{"inputs": ...}` body. Only the first-party
`hf-inference` provider implements that contract. Other Inference Providers
(Fal AI, nscale, DeepInfra, Together, and so on) expose their own routes,
payloads, and response shapes, and are reachable through the Hugging Face
clients rather than this contract.

This release makes that boundary explicit instead of silently producing
confusing failures. Chat (`hf_chat()`, `hf_generate()`, `hf_describe_image()`)
is unaffected: it uses the OpenAI-compatible router endpoint, which selects a
provider server-side and supports every routed chat model.

## Bug fixes

* **`hf_load_dataset()` no longer multiplies rows for datasets with
  variable-length fields** (#61). Row payloads were converted with
  `tibble::as_tibble()`, which recycles a row to the length of its longest
  field. A dataset such as `openai/gdpval`, whose rows carry fields like
  `reference_files`, therefore returned more rows than requested: asking for 3
  rows returned 8, silently duplicating scalar values. Fields that are ever
  non-scalar are now returned as list-columns, so one source row always yields
  exactly one output row. Scalar fields are unchanged, and a JSON `null` is
  still read as `NA` rather than forcing a list-column.

* **Documentation links updated.** 36 links pointed at
  `huggingface.co/docs/api-inference`, which Hugging Face retired; they now
  point at the corresponding Inference Providers task pages. This also clears an
  `R CMD check --as-cran` NOTE about unreachable URLs.

* **Short dataset names resolve more reliably.** `hf_load_dataset("imdb")`
  expands short names to their full `owner/name` form by querying the Hub. That
  lookup had no retry, so a transient network failure silently fell through to
  the unexpanded name and surfaced as a confusing "dataset has been renamed"
  error. The lookup now retries, and if it still cannot expand the name the
  error says so and suggests passing the full ID.

* **Unroutable models now fail with an actionable error.** Task requests
  previously went to `hf-inference` unconditionally, so a model that provider
  does not serve returned an opaque "not found". The package now checks the
  Hub's provider mapping and raises an error naming the providers that do serve
  the model, and suggesting `endpoint_url` for a dedicated Inference Endpoint.
  Resolved mappings are cached for the session. If the Hub is unreachable the
  historical `hf-inference` route is used unchanged, so Hub metadata is not a
  new hard dependency for models that already worked.

* **`hf_text_to_image()` default replaced.** The previous default,
  `black-forest-labs/FLUX.1-schnell`, is served only by third-party providers
  and so was never reachable through the task contract. The default is now
  `stabilityai/stable-diffusion-3-medium-diffusers`, which `hf-inference`
  serves. It is a gated model: accept the licence on its model page once before
  first use.

* **`hf_text_to_speech()` default replaced, and its limits documented.** The
  previous default, `facebook/mms-tts-eng`, is no longer served by any
  Inference Provider. The default is now `hexgrad/Kokoro-82M`. Note that
  `hf-inference` currently serves **no** text-to-speech model, so serverless
  text-to-speech is not reachable through the task contract; the function now
  says so clearly and points to `endpoint_url`.

* **`HF_TOKEN` is now recognised.** The package previously read only the legacy
  `HUGGING_FACE_HUB_TOKEN`, so a token configured by following current Hugging
  Face documentation (or set up for the `hf` CLI or Python client) was ignored
  and users were told no token was found. `HF_TOKEN` is now checked first, with
  `HUGGING_FACE_HUB_TOKEN` still honoured as a fallback. This also applies to
  the legacy `hf_inference()` and `hf_ez_*()` functions, which previously read
  the legacy variable directly.

* **`hf_list_providers()` now covers non-chat models.** It previously queried
  only the router catalogue, which contains chat-completion models, and so
  returned an empty tibble for embedding, classification, and other task
  models. It now merges the Hub provider mapping (all models, all tasks) with
  router pricing and latency metrics where available, and gained a `task`
  column. Select columns by name rather than position.

* **Router routing policies are no longer mistaken for providers.** Model
  specifications such as `model:cheapest`, `model:fastest`, and
  `model:preferred` were previously treated as literal provider names and would
  have produced invalid task URLs. These policies are resolved by the router for
  **chat completions**, where they are passed through unchanged; for task
  requests the suffix is now recognised and ignored rather than corrupting the
  route.

## New features

* **Organization billing.** Set the `HF_BILL_TO` environment variable to send
  the `X-HF-Bill-To` header so Team and Enterprise usage is billed to the
  organization instead of the individual user.

* `hf_clear_provider_cache()` discards the session's cached provider mappings,
  which is useful after provider availability changes.

## Improvements

* Token errors now link to the fine-grained token page with the required
  "Make calls to Inference Providers" permission, and rate-limit errors mention
  the monthly credit allowance.

* Documentation describes the Inference Providers token permission, `HF_TOKEN`,
  organization billing, and the `hf-inference` task-contract boundary, and no
  longer links to the retired `huggingface.co/docs/api-inference` page.

## Known limitations

* Provider availability is read from the Hub at call time and can change; a
  provider may report `status = "error"` transiently. Successful lookups are
  cached for 15 minutes and failed lookups for 60 seconds; call
  `hf_clear_provider_cache()` to discard them sooner.
* The cache is keyed by model and by whether a token was supplied, not by token
  value. Switching between tokens with different access within one session may
  reuse a mapping; clear the cache when doing so.
* Third-party providers are not used for task requests. Reaching them requires
  provider-specific adapters, which this release does not implement.
* An empty or unavailable Hub provider mapping is treated as "unknown" and the
  historical `hf-inference` route is still attempted, so Hub metadata gaps
  cannot block a request that would otherwise have worked.


# huggingfaceR 2.1.0

## New features

* `hf_extract()` turns unstructured text into tidy columns with chat-model structured JSON output. Pass a lightweight named schema such as `c(name = "string", score = "number")` or a full JSON Schema list, and the function returns one row per input text with one column per schema field (#55).

* `hf_chat()` now supports tool/function calling, streaming callbacks, and image inputs for vision-capable chat models. New helpers `hf_tool()`, `hf_run_tools()`, and `hf_describe_image()` make these capabilities available from R pipelines (#55).

* **Multimodal inference wrappers.** New functions add audio, image, and generation workflows: `hf_transcribe()`, `hf_text_to_image()`, `hf_classify_image()`, `hf_caption_image()`, `hf_detect_objects()`, and `hf_text_to_speech()` (#55). Live verification passed for ASR, text-to-image, image classification, captioning, and object detection; public hosted TTS provider support is currently blocked, so `hf_text_to_speech()` is documented for compatible providers or dedicated Inference Endpoints.

* **Hub files, providers, and guarded writes.** New Hub helpers include `hf_hub_download()`, `hf_list_repo_files()`, `hf_search_spaces()`, `hf_search_papers()`, `hf_list_providers()`, `hf_create_repo()`, `hf_upload_file()`, `hf_push_dataset()`, and guarded `hf_delete_repo()` (#55). Search helpers now follow Hub pagination links, and write/destructive operations require `confirm = TRUE`.

* **First-class text tasks.** New API-first, tidyverse-native wrappers round out
  the text toolkit, each accepting character vectors and returning tibbles:
  `hf_summarize()` (summarization), `hf_translate()` (translation),
  `hf_ner()` (named-entity recognition, one tidy row per entity with character
  offsets), `hf_question_answer()` (extractive QA), and
  `hf_table_question_answer()` (ask a data frame a question in plain language).

## Improvements

* **Centralized default models.** A new exported helper, `hf_default_model()`,
  is the single source of truth for every task's default model. All `hf_*`
  functions now resolve their `model` default through it (no behavior change —
  the resolved values are identical), so defaults can be audited or updated in
  one place. Call `hf_default_model()` to see the whole registry, or
  `hf_default_model("translate")` for a single task's default.

* `hf_whoami()` now returns billing/pro status and token-role metadata so users
  can check whether their token is read-only or write-capable before Hub write
  operations.

* **Beginner-friendly default translation model.** `hf_translate()` now defaults
  to `Helsinki-NLP/opus-mt-en-fr` (English to French) instead of
  `facebook/nllb-200-distilled-600M`. The Helsinki-NLP `opus-mt-*` family encodes
  the translation direction in the model ID, so `hf_translate("Hello")` works
  with no FLORES-200 language codes — a smoother first experience. NLLB remains
  fully supported for multilingual translation via the `model`, `source`, and
  `target` arguments.

* **Unified request engine with inference-provider routing.** Internal request
  construction is consolidated in `R/request.R` (`hf_parse_model()`,
  `hf_inference_url()`, `hf_error_body()`, `hf_is_transient()`,
  `hf_task_request()`). As a result, the `model = "id:provider"` suffix now
  selects an inference provider for *all* serverless tasks — including
  embeddings, classification, and the new text tasks — not just chat. Retries
  now back off only on genuinely transient status codes (429/5xx), and error
  messages are consistent across every inference function.

# huggingfaceR 2.0.0

## Breaking changes

* The package no longer requires Python or reticulate for core functionality.
  All inference is handled through the Hugging Face Inference API via httr2.
  Legacy functions that depend on Python/reticulate remain available but are
  not required for new workflows.

* Default chat and generation model changed from `HuggingFaceTB/SmolLM3-3B`
  to `meta-llama/Llama-3.1-8B-Instruct`, which has broader provider support.

## New features

* **API-first architecture**: All core functions (`hf_classify()`, `hf_embed()`,
  `hf_chat()`, `hf_generate()`, `hf_fill_mask()`) use the Hugging Face
  Inference API directly. No Python installation needed.

* **Text classification**: `hf_classify()` for sentiment analysis and
  `hf_classify_zero_shot()` for custom label classification without training.

* **Embeddings and similarity**: `hf_embed()` generates dense vector
  representations. `hf_similarity()` computes pairwise cosine similarity.
  `hf_nearest_neighbors()`, `hf_cluster_texts()`, and `hf_extract_topics()`
  provide higher-level semantic analysis. `hf_embed_umap()` reduces embeddings
  to 2D for visualization.

* **Chat and generation**: `hf_chat()` for single-turn LLM interaction with
  system prompts. `hf_conversation()` and `chat()` for multi-turn conversations
  with persistent history. `hf_generate()` for text completion.
  `hf_fill_mask()` for BERT-style masked token prediction.

* **Hub discovery**: `hf_search_models()`, `hf_model_info()`,
  `hf_search_datasets()`, `hf_dataset_info()`, and `hf_list_tasks()` for
  exploring the Hugging Face Hub from R.

* **Datasets**: `hf_load_dataset()` loads dataset rows directly into tibbles,
  with support for splits, pagination, and column selection.

* **Batch processing**: `hf_embed_batch()`, `hf_classify_batch()`, and
 `hf_classify_zero_shot_batch()` process large inputs with parallel requests.
  `hf_embed_chunks()` and `hf_classify_chunks()` add disk checkpointing for
  datasets too large to hold in memory.

* **tidymodels integration**: `step_hf_embed()` recipe step embeds text columns
  as part of a tidymodels preprocessing pipeline.

* **tidytext integration**: `hf_embed_text()` works directly with data frame
  text columns for tidytext-style workflows.

* **Model availability checking**: `hf_check_inference()` queries model metadata
  to verify whether a model supports the free serverless Inference API before
  you make inference calls.

* **Dedicated Inference Endpoints**: All inference functions accept an
  `endpoint_url` parameter to route requests to a dedicated Inference Endpoint
  instead of the public serverless API. This supports models not available on
  the free tier and production workloads requiring dedicated capacity.

## Improvements

* All functions return tibbles and accept character vectors, enabling natural
  composition with dplyr, tidyr, and the rest of the tidyverse.

* Improved error messages for 404 responses explain that the model may exist
  on the Hub but not be available for serverless inference, and suggest using
  `hf_check_inference()`.

* Documentation updated to clarify that the Inference API serves a curated
  subset of the Hub's 500,000+ models, not all of them.
