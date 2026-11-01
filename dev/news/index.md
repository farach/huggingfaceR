# Changelog

## huggingfaceR (development version)

### New features

- Functions that send one request per input can now send several at once
  (#72).
  [`hf_summarize()`](https://farach.github.io/huggingfaceR/dev/reference/hf_summarize.md),
  [`hf_translate()`](https://farach.github.io/huggingfaceR/dev/reference/hf_translate.md),
  [`hf_ner()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ner.md),
  [`hf_question_answer()`](https://farach.github.io/huggingfaceR/dev/reference/hf_question_answer.md),
  [`hf_table_question_answer()`](https://farach.github.io/huggingfaceR/dev/reference/hf_table_question_answer.md),
  [`hf_fill_mask()`](https://farach.github.io/huggingfaceR/dev/reference/hf_fill_mask.md),
  [`hf_classify_zero_shot()`](https://farach.github.io/huggingfaceR/dev/reference/hf_classify_zero_shot.md),
  [`hf_generate()`](https://farach.github.io/huggingfaceR/dev/reference/hf_generate.md),
  [`hf_extract()`](https://farach.github.io/huggingfaceR/dev/reference/hf_extract.md),
  [`hf_classify_image()`](https://farach.github.io/huggingfaceR/dev/reference/hf_classify_image.md),
  [`hf_caption_image()`](https://farach.github.io/huggingfaceR/dev/reference/hf_caption_image.md),
  [`hf_detect_objects()`](https://farach.github.io/huggingfaceR/dev/reference/hf_detect_objects.md),
  [`hf_describe_image()`](https://farach.github.io/huggingfaceR/dev/reference/hf_describe_image.md),
  [`hf_text_to_image()`](https://farach.github.io/huggingfaceR/dev/reference/hf_text_to_image.md),
  [`hf_text_to_speech()`](https://farach.github.io/huggingfaceR/dev/reference/hf_text_to_speech.md),
  and
  [`hf_transcribe()`](https://farach.github.io/huggingfaceR/dev/reference/hf_transcribe.md)
  gain a `max_active` argument. It defaults to
  `getOption("huggingfaceR.max_active", 1L)`, so requests stay
  sequential unless you opt in. Results keep their input order, and the
  first failing input still stops the call with its error. Each request
  is billed, so a higher `max_active` spends credits faster but not in
  larger amounts.
- [`hf_load_dataset()`](https://farach.github.io/huggingfaceR/dev/reference/hf_load_dataset.md)
  requests every page after the first together, up to `max_active` at a
  time (default 4). Loading 3,000 rows took about 2 seconds instead of
  3.6.
- `hf_load_dataset(format = "parquet")` reads a split from its Parquet
  export with the arrow package. A complete split often needs one
  request instead of one per 100 rows, which is faster and stays well
  inside rate limits.
- New dataset viewer helpers query a split without loading it.
  [`hf_dataset_search()`](https://farach.github.io/huggingfaceR/dev/reference/hf_dataset_search.md)
  runs full-text search,
  [`hf_dataset_filter()`](https://farach.github.io/huggingfaceR/dev/reference/hf_dataset_filter.md)
  selects rows with a SQL-style condition,
  [`hf_dataset_statistics()`](https://farach.github.io/huggingfaceR/dev/reference/hf_dataset_statistics.md)
  returns the per-column summaries and histograms the Hub computes, and
  [`hf_dataset_validity()`](https://farach.github.io/huggingfaceR/dev/reference/hf_dataset_validity.md)
  reports which of these features work for a dataset.
- [`hf_search_models()`](https://farach.github.io/huggingfaceR/dev/reference/hf_search_models.md)
  gains `inference_provider`, which keeps only models served by the
  named Inference Providers. Use `"all"` for any provider.

### Rate limits and billing

- Requests that hit a Hugging Face rate limit (HTTP 429) now wait as
  long as the server asks, read from the `Retry-After` or `RateLimit`
  header, for up to 5 minutes. Before, retries used a short backoff that
  ended long before the 5-minute rate-limit window reset.
- A 429 now explains the 5-minute limit window, notes that requests with
  a token get higher limits, and links the rate-limit documentation.
  Dataset calls also suggest `format = "parquet"` for large loads.
- An exhausted credit balance (HTTP 402) now has its own message. Free
  accounts no longer include monthly Inference Providers credits, so it
  points to the billing page instead of suggesting a wait.
- The README and vignettes describe current Inference Providers billing
  and routing. The first-party `hf-inference` provider has served CPU
  tasks such as classification and embeddings since July 2025, so chat
  and generation go to third-party providers through the router.

### Bug fixes

- Batch functions no longer retry a failing request forever. Repeated
  429 or 503 responses are retried a bounded number of times and then
  reported as errors.
- Dropped connections and timeouts are now retried like other transient
  failures.
- Images and audio already written by
  [`hf_text_to_image()`](https://farach.github.io/huggingfaceR/dev/reference/hf_text_to_image.md)
  and
  [`hf_text_to_speech()`](https://farach.github.io/huggingfaceR/dev/reference/hf_text_to_speech.md)
  are kept when a later input fails.
- [`hf_describe_image()`](https://farach.github.io/huggingfaceR/dev/reference/hf_describe_image.md)
  streams one image at a time instead of failing when `max_active` is
  greater than 1.
- [`hf_embed_umap()`](https://farach.github.io/huggingfaceR/dev/reference/hf_embed_umap.md)
  no longer fails when some `text` is `NA`. It returns one row per input
  with `NA` coordinates for the missing ones.
- [`hf_embed_umap()`](https://farach.github.io/huggingfaceR/dev/reference/hf_embed_umap.md),
  [`hf_cluster_texts()`](https://farach.github.io/huggingfaceR/dev/reference/hf_cluster_texts.md),
  and
  [`step_hf_embed()`](https://farach.github.io/huggingfaceR/dev/reference/step_hf_embed.md)
  stop with a clear error when embeddings have different lengths,
  instead of recycling the shorter ones.

## huggingfaceR 2.4.0

CRAN release: 2026-10-31

### Performance

These changes remove interpreted loops and repeated work from the R code
that runs around each request. Hosted calls remain bounded by network
latency and server-side inference, which a client written in any
language must wait for. Functions that send one request per input, such
as
[`hf_summarize()`](https://farach.github.io/huggingfaceR/dev/reference/hf_summarize.md),
[`hf_translate()`](https://farach.github.io/huggingfaceR/dev/reference/hf_translate.md),
and
[`hf_ner()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ner.md),
still send those requests one at a time. An opt-in way to run them
concurrently is tracked in \#72.

- [`hf_similarity()`](https://farach.github.io/huggingfaceR/dev/reference/hf_similarity.md)
  builds its pairs with vectorized primitives instead of
  [`utils::combn()`](https://rdrr.io/r/utils/combn.html) and scores each
  embedding dimension with one BLAS cross-product. Inputs with mixed
  dimensions no longer fall back to an R loop over every pair, and
  embeddings with missing or infinite values now give `NA` similarities
  instead of an error.
- [`hf_nearest_neighbors()`](https://farach.github.io/huggingfaceR/dev/reference/hf_nearest_neighbors.md)
  scores the corpus with blocked matrix-vector products instead of one R
  function call per row. Embeddings that are non-numeric, zero-norm, or
  of a different dimension than the query now score `NA`. Previously
  they raised an error, returned `NaN`, or were silently recycled with a
  warning.
- [`hf_embed_batch()`](https://farach.github.io/huggingfaceR/dev/reference/hf_embed_batch.md),
  [`hf_classify_batch()`](https://farach.github.io/huggingfaceR/dev/reference/hf_classify_batch.md),
  and
  [`hf_classify_zero_shot_batch()`](https://farach.github.io/huggingfaceR/dev/reference/hf_classify_zero_shot_batch.md)
  resolve the provider and build the request once per call, then write
  results straight into columns instead of building, binding, and
  sorting a tibble for every batch or label. A response with more
  embeddings than texts is now truncated with a warning instead of
  failing.
- Embedding responses are parsed with the optional ‘RcppSimdJson’
  package when it is installed. Well-formed numeric arrays go straight
  into a matrix instead of one R object per number, and any other shape
  falls back to ‘jsonlite’, so the embeddings returned are the same. Set
  `options(huggingfaceR.simdjson = FALSE)` to always use ‘jsonlite’.
- [`hf_load_dataset()`](https://farach.github.io/huggingfaceR/dev/reference/hf_load_dataset.md)
  and the paginated Hub searches combine pages once rather than growing
  a list on every request, and dataset rows are converted to columns
  with primitive field access.
- [`hf_search_models()`](https://farach.github.io/huggingfaceR/dev/reference/hf_search_models.md),
  [`hf_search_datasets()`](https://farach.github.io/huggingfaceR/dev/reference/hf_search_datasets.md),
  [`hf_search_spaces()`](https://farach.github.io/huggingfaceR/dev/reference/hf_search_spaces.md),
  [`hf_search_papers()`](https://farach.github.io/huggingfaceR/dev/reference/hf_search_papers.md),
  [`hf_list_repo_files()`](https://farach.github.io/huggingfaceR/dev/reference/hf_list_repo_files.md),
  and
  [`hf_list_providers()`](https://farach.github.io/huggingfaceR/dev/reference/hf_list_providers.md)
  bind records by column instead of building a tibble for every record.
- [`step_hf_embed()`](https://farach.github.io/huggingfaceR/dev/reference/step_hf_embed.md)
  adds all embedding columns in one assignment when baked.
- A developer workflow profiles these paths against the 2.3.0 release
  with HTTP mocked, and fails if any output changes.

### Dependencies

- huggingfaceR now requires httr2 1.1.1 or later. Embedding and
  classification functions call
  [`httr2::req_perform_parallel()`](https://httr2.r-lib.org/reference/req_perform_parallel.html)
  with its `max_active` argument, and streaming chat uses
  [`httr2::req_perform_connection()`](https://httr2.r-lib.org/reference/req_perform_connection.html).
  Neither exists in earlier httr2 releases, so those functions failed
  with them.
- The declared minimum versions of cli (3.0.0), dplyr (1.0.0), recipes
  (1.0.4), and testthat (3.1.7) now match the functions the package and
  its tests use.
- ‘RcppSimdJson’ is a new suggested package, used only when installed.

## huggingfaceR 2.3.0

CRAN release: 2026-09-18

### Optional local models

- [`hf_load_local_model()`](https://farach.github.io/huggingfaceR/dev/reference/hf_load_local_model.md)
  uses `processor_kwargs` on newer Sentence Transformers interfaces,
  avoiding the deprecated `tokenizer_kwargs` argument, while retaining
  compatibility with older constructors and preserving local-only
  loading flags.
- [`hf_local_setup()`](https://farach.github.io/huggingfaceR/dev/reference/hf_local_setup.md),
  [`hf_download_model()`](https://farach.github.io/huggingfaceR/dev/reference/hf_download_model.md),
  and
  [`hf_load_local_model()`](https://farach.github.io/huggingfaceR/dev/reference/hf_load_local_model.md)
  provide an explicit setup, revision-aware download, and reusable
  local-model workflow. Python dependencies are optional and are not
  initialized at package load.
- [`hf_embed_local()`](https://farach.github.io/huggingfaceR/dev/reference/hf_embed_local.md)
  and
  [`hf_classify_local()`](https://farach.github.io/huggingfaceR/dev/reference/hf_classify_local.md)
  run standard safetensors-based embedding and text-classification
  models on the CPU by default, returning the same tidy columns as their
  API counterparts. Input order, duplicates, and missing values are
  preserved.
- Local loading keeps tokenizer and weights in the same snapshot,
  refuses remote custom code, supports cached offline use, and never
  falls back to hosted inference. Existing API-first functions and
  defaults are unchanged. Embedding snapshots require Sentence
  Transformers module metadata so a partial cache cannot silently change
  the model’s pooling configuration.
- A new local-model vignette and standalone online/offline validation
  script cover real downloads, predictions, and reuse across fresh R
  sessions.
- Local internals are separated into public operations, Python
  integration, and validation/error handling. Loading a Hub model
  initializes its Python environment and validates its snapshot once,
  without introducing a session-wide validation cache.
- Developer profiling compares the installed pre-refactor and current
  versions on the same native Linux runner, including cached loading,
  Python-native inference, R wrapper allocations, and unchanged
  prediction results.

## huggingfaceR 2.2.0

CRAN release: 2026-08-29

This release re-aligns the package with the current Hugging Face
Inference Providers platform. Since mid-2025 the first-party
`hf-inference` provider has narrowed to CPU-friendly classic models, and
several models the package shipped as defaults are no longer served by
it.

### Scope of task-based inference

huggingfaceR’s task functions
([`hf_classify()`](https://farach.github.io/huggingfaceR/dev/reference/hf_classify.md),
[`hf_summarize()`](https://farach.github.io/huggingfaceR/dev/reference/hf_summarize.md),
[`hf_embed()`](https://farach.github.io/huggingfaceR/dev/reference/hf_embed.md),
[`hf_text_to_image()`](https://farach.github.io/huggingfaceR/dev/reference/hf_text_to_image.md)
and friends) speak the Hugging Face **task API contract**:
`POST .../models/{model}` with an `{"inputs": ...}` body. Only the
first-party `hf-inference` provider implements that contract. Other
Inference Providers (Fal AI, nscale, DeepInfra, Together, and so on)
expose their own routes, payloads, and response shapes, and are
reachable through the Hugging Face clients rather than this contract.

This release makes that boundary explicit instead of silently producing
confusing failures. Chat
([`hf_chat()`](https://farach.github.io/huggingfaceR/dev/reference/hf_chat.md),
[`hf_generate()`](https://farach.github.io/huggingfaceR/dev/reference/hf_generate.md),
[`hf_describe_image()`](https://farach.github.io/huggingfaceR/dev/reference/hf_describe_image.md))
is unaffected: it uses the OpenAI-compatible router endpoint, which
selects a provider server-side and supports every routed chat model.

### Bug fixes

- **[`hf_load_dataset()`](https://farach.github.io/huggingfaceR/dev/reference/hf_load_dataset.md)
  no longer multiplies rows for datasets with variable-length fields**
  (#61). Row payloads were converted with
  [`tibble::as_tibble()`](https://tibble.tidyverse.org/reference/as_tibble.html),
  which recycles a row to the length of its longest field. A dataset
  such as `openai/gdpval`, whose rows carry fields like
  `reference_files`, therefore returned more rows than requested: asking
  for 3 rows returned 8, silently duplicating scalar values. Fields that
  are ever non-scalar are now returned as list-columns, so one source
  row always yields exactly one output row. Scalar fields are unchanged,
  and a JSON `null` is still read as `NA` rather than forcing a
  list-column.

- **Documentation links updated.** 36 links pointed at
  `huggingface.co/docs/api-inference`, which Hugging Face retired; they
  now point at the corresponding Inference Providers task pages. This
  also clears an `R CMD check --as-cran` NOTE about unreachable URLs.

- **Short dataset names resolve more reliably.**
  `hf_load_dataset("imdb")` expands short names to their full
  `owner/name` form by querying the Hub. That lookup had no retry, so a
  transient network failure silently fell through to the unexpanded name
  and surfaced as a confusing “dataset has been renamed” error. The
  lookup now retries, and if it still cannot expand the name the error
  says so and suggests passing the full ID.

- **Unroutable models now fail with an actionable error.** Task requests
  previously went to `hf-inference` unconditionally, so a model that
  provider does not serve returned an opaque “not found”. The package
  now checks the Hub’s provider mapping and raises an error naming the
  providers that do serve the model, and suggesting `endpoint_url` for a
  dedicated Inference Endpoint. Resolved mappings are cached for the
  session. If the Hub is unreachable the historical `hf-inference` route
  is used unchanged, so Hub metadata is not a new hard dependency for
  models that already worked.

- **[`hf_text_to_image()`](https://farach.github.io/huggingfaceR/dev/reference/hf_text_to_image.md)
  default replaced.** The previous default,
  `black-forest-labs/FLUX.1-schnell`, is served only by third-party
  providers and so was never reachable through the task contract. The
  default is now `stabilityai/stable-diffusion-3-medium-diffusers`,
  which `hf-inference` serves. It is a gated model: accept the licence
  on its model page once before first use.

- **[`hf_text_to_speech()`](https://farach.github.io/huggingfaceR/dev/reference/hf_text_to_speech.md)
  default replaced, and its limits documented.** The previous default,
  `facebook/mms-tts-eng`, is no longer served by any Inference Provider.
  The default is now `hexgrad/Kokoro-82M`. Note that `hf-inference`
  currently serves **no** text-to-speech model, so serverless
  text-to-speech is not reachable through the task contract; the
  function now says so clearly and points to `endpoint_url`.

- **`HF_TOKEN` is now recognised.** The package previously read only the
  legacy `HUGGING_FACE_HUB_TOKEN`, so a token configured by following
  current Hugging Face documentation (or set up for the `hf` CLI or
  Python client) was ignored and users were told no token was found.
  `HF_TOKEN` is now checked first, with `HUGGING_FACE_HUB_TOKEN` still
  honoured as a fallback. This also applies to the legacy
  [`hf_inference()`](https://farach.github.io/huggingfaceR/dev/reference/hf_inference.md)
  and `hf_ez_*()` functions, which previously read the legacy variable
  directly.

- **[`hf_list_providers()`](https://farach.github.io/huggingfaceR/dev/reference/hf_list_providers.md)
  now covers non-chat models.** It previously queried only the router
  catalogue, which contains chat-completion models, and so returned an
  empty tibble for embedding, classification, and other task models. It
  now merges the Hub provider mapping (all models, all tasks) with
  router pricing and latency metrics where available, and gained a
  `task` column. Select columns by name rather than position.

- **Router routing policies are no longer mistaken for providers.**
  Model specifications such as `model:cheapest`, `model:fastest`, and
  `model:preferred` were previously treated as literal provider names
  and would have produced invalid task URLs. These policies are resolved
  by the router for **chat completions**, where they are passed through
  unchanged; for task requests the suffix is now recognised and ignored
  rather than corrupting the route.

### New features

- **Organization billing.** Set the `HF_BILL_TO` environment variable to
  send the `X-HF-Bill-To` header so Team and Enterprise usage is billed
  to the organization instead of the individual user.

- [`hf_clear_provider_cache()`](https://farach.github.io/huggingfaceR/dev/reference/hf_clear_provider_cache.md)
  discards the session’s cached provider mappings, which is useful after
  provider availability changes.

### Improvements

- Token errors now link to the fine-grained token page with the required
  “Make calls to Inference Providers” permission, and rate-limit errors
  mention the monthly credit allowance.

- Documentation describes the Inference Providers token permission,
  `HF_TOKEN`, organization billing, and the `hf-inference` task-contract
  boundary, and no longer links to the retired
  `huggingface.co/docs/api-inference` page.

### Known limitations

- Provider availability is read from the Hub at call time and can
  change; a provider may report `status = "error"` transiently.
  Successful lookups are cached for 15 minutes and failed lookups for 60
  seconds; call
  [`hf_clear_provider_cache()`](https://farach.github.io/huggingfaceR/dev/reference/hf_clear_provider_cache.md)
  to discard them sooner.
- The cache is keyed by model and by whether a token was supplied, not
  by token value. Switching between tokens with different access within
  one session may reuse a mapping; clear the cache when doing so.
- Third-party providers are not used for task requests. Reaching them
  requires provider-specific adapters, which this release does not
  implement.
- An empty or unavailable Hub provider mapping is treated as “unknown”
  and the historical `hf-inference` route is still attempted, so Hub
  metadata gaps cannot block a request that would otherwise have worked.

## huggingfaceR 2.1.0

CRAN release: 2026-06-30

### New features

- [`hf_extract()`](https://farach.github.io/huggingfaceR/dev/reference/hf_extract.md)
  turns unstructured text into tidy columns with chat-model structured
  JSON output. Pass a lightweight named schema such as
  `c(name = "string", score = "number")` or a full JSON Schema list, and
  the function returns one row per input text with one column per schema
  field (#55).

- [`hf_chat()`](https://farach.github.io/huggingfaceR/dev/reference/hf_chat.md)
  now supports tool/function calling, streaming callbacks, and image
  inputs for vision-capable chat models. New helpers
  [`hf_tool()`](https://farach.github.io/huggingfaceR/dev/reference/hf_tool.md),
  [`hf_run_tools()`](https://farach.github.io/huggingfaceR/dev/reference/hf_run_tools.md),
  and
  [`hf_describe_image()`](https://farach.github.io/huggingfaceR/dev/reference/hf_describe_image.md)
  make these capabilities available from R pipelines (#55).

- **Multimodal inference wrappers.** New functions add audio, image, and
  generation workflows:
  [`hf_transcribe()`](https://farach.github.io/huggingfaceR/dev/reference/hf_transcribe.md),
  [`hf_text_to_image()`](https://farach.github.io/huggingfaceR/dev/reference/hf_text_to_image.md),
  [`hf_classify_image()`](https://farach.github.io/huggingfaceR/dev/reference/hf_classify_image.md),
  [`hf_caption_image()`](https://farach.github.io/huggingfaceR/dev/reference/hf_caption_image.md),
  [`hf_detect_objects()`](https://farach.github.io/huggingfaceR/dev/reference/hf_detect_objects.md),
  and
  [`hf_text_to_speech()`](https://farach.github.io/huggingfaceR/dev/reference/hf_text_to_speech.md)
  (#55). Live verification passed for ASR, text-to-image, image
  classification, captioning, and object detection; public hosted TTS
  provider support is currently blocked, so
  [`hf_text_to_speech()`](https://farach.github.io/huggingfaceR/dev/reference/hf_text_to_speech.md)
  is documented for compatible providers or dedicated Inference
  Endpoints.

- **Hub files, providers, and guarded writes.** New Hub helpers include
  [`hf_hub_download()`](https://farach.github.io/huggingfaceR/dev/reference/hf_hub_download.md),
  [`hf_list_repo_files()`](https://farach.github.io/huggingfaceR/dev/reference/hf_list_repo_files.md),
  [`hf_search_spaces()`](https://farach.github.io/huggingfaceR/dev/reference/hf_search_spaces.md),
  [`hf_search_papers()`](https://farach.github.io/huggingfaceR/dev/reference/hf_search_papers.md),
  [`hf_list_providers()`](https://farach.github.io/huggingfaceR/dev/reference/hf_list_providers.md),
  [`hf_create_repo()`](https://farach.github.io/huggingfaceR/dev/reference/hf_create_repo.md),
  [`hf_upload_file()`](https://farach.github.io/huggingfaceR/dev/reference/hf_upload_file.md),
  [`hf_push_dataset()`](https://farach.github.io/huggingfaceR/dev/reference/hf_push_dataset.md),
  and guarded
  [`hf_delete_repo()`](https://farach.github.io/huggingfaceR/dev/reference/hf_delete_repo.md)
  (#55). Search helpers now follow Hub pagination links, and
  write/destructive operations require `confirm = TRUE`.

- **First-class text tasks.** New API-first, tidyverse-native wrappers
  round out the text toolkit, each accepting character vectors and
  returning tibbles:
  [`hf_summarize()`](https://farach.github.io/huggingfaceR/dev/reference/hf_summarize.md)
  (summarization),
  [`hf_translate()`](https://farach.github.io/huggingfaceR/dev/reference/hf_translate.md)
  (translation),
  [`hf_ner()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ner.md)
  (named-entity recognition, one tidy row per entity with character
  offsets),
  [`hf_question_answer()`](https://farach.github.io/huggingfaceR/dev/reference/hf_question_answer.md)
  (extractive QA), and
  [`hf_table_question_answer()`](https://farach.github.io/huggingfaceR/dev/reference/hf_table_question_answer.md)
  (ask a data frame a question in plain language).

### Improvements

- **Centralized default models.** A new exported helper,
  [`hf_default_model()`](https://farach.github.io/huggingfaceR/dev/reference/hf_default_model.md),
  is the single source of truth for every task’s default model. All
  `hf_*` functions now resolve their `model` default through it (no
  behavior change — the resolved values are identical), so defaults can
  be audited or updated in one place. Call
  [`hf_default_model()`](https://farach.github.io/huggingfaceR/dev/reference/hf_default_model.md)
  to see the whole registry, or `hf_default_model("translate")` for a
  single task’s default.

- [`hf_whoami()`](https://farach.github.io/huggingfaceR/dev/reference/hf_whoami.md)
  now returns billing/pro status and token-role metadata so users can
  check whether their token is read-only or write-capable before Hub
  write operations.

- **Beginner-friendly default translation model.**
  [`hf_translate()`](https://farach.github.io/huggingfaceR/dev/reference/hf_translate.md)
  now defaults to `Helsinki-NLP/opus-mt-en-fr` (English to French)
  instead of `facebook/nllb-200-distilled-600M`. The Helsinki-NLP
  `opus-mt-*` family encodes the translation direction in the model ID,
  so `hf_translate("Hello")` works with no FLORES-200 language codes — a
  smoother first experience. NLLB remains fully supported for
  multilingual translation via the `model`, `source`, and `target`
  arguments.

- **Unified request engine with inference-provider routing.** Internal
  request construction is consolidated in `R/request.R`
  (`hf_parse_model()`, `hf_inference_url()`, `hf_error_body()`,
  `hf_is_transient()`, `hf_task_request()`). As a result, the
  `model = "id:provider"` suffix now selects an inference provider for
  *all* serverless tasks — including embeddings, classification, and the
  new text tasks — not just chat. Retries now back off only on genuinely
  transient status codes (429/5xx), and error messages are consistent
  across every inference function.

## huggingfaceR 2.0.0

### Breaking changes

- The package no longer requires Python or reticulate for core
  functionality. All inference is handled through the Hugging Face
  Inference API via httr2. Legacy functions that depend on
  Python/reticulate remain available but are not required for new
  workflows.

- Default chat and generation model changed from
  `HuggingFaceTB/SmolLM3-3B` to `meta-llama/Llama-3.1-8B-Instruct`,
  which has broader provider support.

### New features

- **API-first architecture**: All core functions
  ([`hf_classify()`](https://farach.github.io/huggingfaceR/dev/reference/hf_classify.md),
  [`hf_embed()`](https://farach.github.io/huggingfaceR/dev/reference/hf_embed.md),
  [`hf_chat()`](https://farach.github.io/huggingfaceR/dev/reference/hf_chat.md),
  [`hf_generate()`](https://farach.github.io/huggingfaceR/dev/reference/hf_generate.md),
  [`hf_fill_mask()`](https://farach.github.io/huggingfaceR/dev/reference/hf_fill_mask.md))
  use the Hugging Face Inference API directly. No Python installation
  needed.

- **Text classification**:
  [`hf_classify()`](https://farach.github.io/huggingfaceR/dev/reference/hf_classify.md)
  for sentiment analysis and
  [`hf_classify_zero_shot()`](https://farach.github.io/huggingfaceR/dev/reference/hf_classify_zero_shot.md)
  for custom label classification without training.

- **Embeddings and similarity**:
  [`hf_embed()`](https://farach.github.io/huggingfaceR/dev/reference/hf_embed.md)
  generates dense vector representations.
  [`hf_similarity()`](https://farach.github.io/huggingfaceR/dev/reference/hf_similarity.md)
  computes pairwise cosine similarity.
  [`hf_nearest_neighbors()`](https://farach.github.io/huggingfaceR/dev/reference/hf_nearest_neighbors.md),
  [`hf_cluster_texts()`](https://farach.github.io/huggingfaceR/dev/reference/hf_cluster_texts.md),
  and
  [`hf_extract_topics()`](https://farach.github.io/huggingfaceR/dev/reference/hf_extract_topics.md)
  provide higher-level semantic analysis.
  [`hf_embed_umap()`](https://farach.github.io/huggingfaceR/dev/reference/hf_embed_umap.md)
  reduces embeddings to 2D for visualization.

- **Chat and generation**:
  [`hf_chat()`](https://farach.github.io/huggingfaceR/dev/reference/hf_chat.md)
  for single-turn LLM interaction with system prompts.
  [`hf_conversation()`](https://farach.github.io/huggingfaceR/dev/reference/hf_conversation.md)
  and
  [`chat()`](https://farach.github.io/huggingfaceR/dev/reference/chat.md)
  for multi-turn conversations with persistent history.
  [`hf_generate()`](https://farach.github.io/huggingfaceR/dev/reference/hf_generate.md)
  for text completion.
  [`hf_fill_mask()`](https://farach.github.io/huggingfaceR/dev/reference/hf_fill_mask.md)
  for BERT-style masked token prediction.

- **Hub discovery**:
  [`hf_search_models()`](https://farach.github.io/huggingfaceR/dev/reference/hf_search_models.md),
  [`hf_model_info()`](https://farach.github.io/huggingfaceR/dev/reference/hf_model_info.md),
  [`hf_search_datasets()`](https://farach.github.io/huggingfaceR/dev/reference/hf_search_datasets.md),
  [`hf_dataset_info()`](https://farach.github.io/huggingfaceR/dev/reference/hf_dataset_info.md),
  and
  [`hf_list_tasks()`](https://farach.github.io/huggingfaceR/dev/reference/hf_list_tasks.md)
  for exploring the Hugging Face Hub from R.

- **Datasets**:
  [`hf_load_dataset()`](https://farach.github.io/huggingfaceR/dev/reference/hf_load_dataset.md)
  loads dataset rows directly into tibbles, with support for splits,
  pagination, and column selection.

- **Batch processing**:
  [`hf_embed_batch()`](https://farach.github.io/huggingfaceR/dev/reference/hf_embed_batch.md),
  [`hf_classify_batch()`](https://farach.github.io/huggingfaceR/dev/reference/hf_classify_batch.md),
  and
  [`hf_classify_zero_shot_batch()`](https://farach.github.io/huggingfaceR/dev/reference/hf_classify_zero_shot_batch.md)
  process large inputs with parallel requests.
  [`hf_embed_chunks()`](https://farach.github.io/huggingfaceR/dev/reference/hf_embed_chunks.md)
  and
  [`hf_classify_chunks()`](https://farach.github.io/huggingfaceR/dev/reference/hf_classify_chunks.md)
  add disk checkpointing for datasets too large to hold in memory.

- **tidymodels integration**:
  [`step_hf_embed()`](https://farach.github.io/huggingfaceR/dev/reference/step_hf_embed.md)
  recipe step embeds text columns as part of a tidymodels preprocessing
  pipeline.

- **tidytext integration**:
  [`hf_embed_text()`](https://farach.github.io/huggingfaceR/dev/reference/hf_embed_text.md)
  works directly with data frame text columns for tidytext-style
  workflows.

- **Model availability checking**:
  [`hf_check_inference()`](https://farach.github.io/huggingfaceR/dev/reference/hf_check_inference.md)
  queries model metadata to verify whether a model supports the free
  serverless Inference API before you make inference calls.

- **Dedicated Inference Endpoints**: All inference functions accept an
  `endpoint_url` parameter to route requests to a dedicated Inference
  Endpoint instead of the public serverless API. This supports models
  not available on the free tier and production workloads requiring
  dedicated capacity.

### Improvements

- All functions return tibbles and accept character vectors, enabling
  natural composition with dplyr, tidyr, and the rest of the tidyverse.

- Improved error messages for 404 responses explain that the model may
  exist on the Hub but not be available for serverless inference, and
  suggest using
  [`hf_check_inference()`](https://farach.github.io/huggingfaceR/dev/reference/hf_check_inference.md).

- Documentation updated to clarify that the Inference API serves a
  curated subset of the Hub’s 500,000+ models, not all of them.
