# huggingfaceR — Implementation Plan: Expanded Hugging Face Capabilities

> Status: proposal / design doc. Lives in `dev/` so it can be edited or deleted
> freely; it is not part of the built package or site.
>
> Target: extend the v2 API-first architecture with the modern Hugging Face
> Inference Providers surface (structured output, tools, streaming, vision,
> speech, image generation), the missing first-class text tasks, and Hub
> read/write operations — without breaking existing functions.

---

## 1. Guiding principles (do not violate)

These are the invariants the current package already honors. Every new function
must follow them so the package stays coherent.

1. **API-first, no Python required.** Everything goes through `httr2` against
   `huggingface.co` / `router.huggingface.co` / `datasets-server.huggingface.co`.
   Python/reticulate functions stay quarantined as "Legacy".
2. **Tidyverse-native.** Inputs accept character vectors (or data frames where
   natural); outputs are tibbles with stable, documented column names and types.
   One row per logical result (e.g. one row per entity for NER, one row per
   bounding box for detection).
3. **Vectorized + order-preserving.** Multi-element input returns multi-row
   output in input order; `NA` in → `NA`/`NULL` placeholder out (never silently
   dropped), matching `hf_embed()`/`hf_classify()` behavior.
4. **Pipe-friendly & composable.** Results `unnest()` cleanly and join back to
   source data frames.
5. **Backward compatible.** No existing exported signature or output shape
   changes. New parameters are added with defaults that reproduce current
   behavior. The Phase 0 refactor is covered by characterization tests first.
6. **Consistent auth + errors + provider routing** through one internal request
   engine (see Phase 0).

---

## 2. Confirmed current Hugging Face API surface (June 2026)

Verified against `huggingface.co/docs/inference-providers` and `.../hub`:

- **Chat (OpenAI-compatible):** `POST https://router.huggingface.co/v1/chat/completions`
  - Supports LLM **and** VLM (vision) via `image_url` content blocks
    (task `image-text-to-text`).
  - `tools` + `tool_choice` (function calling), `response_format` with
    `json_schema` (`strict`) or `json_object` (structured output), `stream`,
    `seed`, `stop`.
- **Model/provider catalog:** `GET https://router.huggingface.co/v1/models`
  returns per-provider metadata: `supports_tools`, `supports_structured_output`,
  `pricing.{input,output}`, `context_length`, `first_token_latency_ms`,
  `throughput`, `status`. This is the basis for smart provider selection.
- **Task routes (provider-routed):** `https://router.huggingface.co/{provider}/...`;
  the default HF provider uses `hf-inference/models/{model}`. Tasks include
  `automatic-speech-recognition` (returns `{text, chunks[]}`), `text-to-speech`,
  `text-to-image`, `image-classification`, `object-detection`,
  `image-segmentation`, `feature-extraction`, `summarization`, `translation`,
  `token-classification`, `question-answering`, `table-question-answering`,
  `fill-mask`, `zero-shot-classification`.
- **Hub read:** `GET /api/{models,datasets,spaces}` (search, supports cursor
  pagination), `/api/{type}/{id}` (info), `/api/{type}/{id}/tree/{rev}` (file
  list), `https://huggingface.co/{id}/resolve/{rev}/{path}` (file download;
  datasets/spaces are prefixed).
- **Hub write:** `POST /api/repos/create`, the commit API
  (`POST /api/{type}s/{id}/commit/{rev}`), `POST /api/repos/delete`.

Key architectural finding: `hf_chat`/`hf_generate` already use the modern
multi-provider `/v1/chat/completions` route, but `hf_embed`/`hf_classify`/
`hf_fill_mask` use `hf_api_request()` / `hf_build_request()`, both hardcoded to
the **legacy single-provider** `hf-inference/models/{model}` route and still
sending the deprecated `options={wait_for_model,use_cache}` body. Provider
selection (`model:provider`) therefore silently does nothing for those tasks.
Fixing this is Phase 0 and unblocks everything else.

---

## 3. Phase 0 — Foundation: one request engine (no user-facing change)

**Goal:** collapse the three duplicated request/error code paths (`utils.R`,
`batch.R`, `chat.R`/`generate.R`/`datasets_api.R`) into a single internal engine,
add provider routing for *all* tasks, and remove the dead `options` body.

### Deliverables
- `R/request.R` (new):
  - `hf_parse_model(model)` → `list(model, provider)` from `"id:provider"` suffix.
  - `hf_route(task, model, provider, endpoint_url)` → base URL. Handles:
    serverless task route, provider route, and dedicated `endpoint_url`.
  - `hf_req(url, token, ...)` → builds an `httr2` request with auth, retry
    (`req_retry`, honoring `Retry-After` on 429/503), throttle, and a single
    shared `req_error` translator (`hf_error_body()`), reused everywhere.
  - `hf_perform_json(req)` and `hf_perform_raw(req)` for text vs binary.
  - Binary I/O helpers: `hf_as_input_bytes(x)` (path | URL | raw → raw +
    content-type) and `hf_write_output(raw, output, ext)` (write file / return
    raw).
- Migrate `hf_api_request()`, `hf_build_request()`, `hf_chat`, `hf_generate`,
  `chat.hf_conversation`, `hf_fill_mask` onto the engine. Drop the deprecated
  `options` block (or keep `wait_for_model` as a no-op with a one-time
  deprecation note).
- Centralize default model IDs in `R/defaults.R` (`hf_default_model(task)`), so
  model-availability churn is fixed in one place.

### Second/third-order effects & mitigations
- **Risk: behavior drift during refactor.** Mitigation: write *characterization
  tests* first (Section 8.0) that lock today's request bodies and output tibbles,
  then refactor until green.
- **Provider routing now active for embeddings/classification** → users can do
  `hf_embed(x, model = "...:nebius")`. Document; default stays `hf-inference`.
- **`req_retry` on POST**: HF inference POSTs are safe to retry; keep
  `is_transient` on 429/500/503 only.

---

## 4. Phase 1 — Modern chat capabilities (build on existing `/v1/chat/completions`)

Highest value-to-risk ratio: the route is already in use.

### 4.1 Structured extraction — `hf_extract()`  ⭐ flagship
The single most valuable addition for R users: reliably turn unstructured text
into tidy columns.

- Signature: `hf_extract(text, schema, model = hf_default_model("chat"),
  strict = TRUE, token = NULL, ...)`.
- `schema` accepts either a JSON Schema (list) or a lightweight field spec, e.g.
  `c(sentiment = "string", score = "number", is_complaint = "boolean")`, which we
  expand into a strict JSON Schema internally.
- Sends `response_format = list(type = "json_schema", json_schema = ...)`,
  parses the guaranteed-valid JSON with `jsonlite`, binds to a tibble with one
  column per schema field, vectorized over `text` (one row per input).
- **Use cases (R):** survey free-text → coded variables; clinical/inspection
  notes → structured fields; making *any* LLM output safe to `mutate()` on.
- **Second-order:** not every provider supports structured output
  (`supports_structured_output`). Pre-check via `/v1/models`; if unsupported,
  fall back to `json_object` + validate, or error with a suggested provider.

### 4.2 Tool / function calling — `hf_tool()`, `hf_chat(tools=)`, `hf_run_tools()`
- `hf_tool(name, description, parameters)` builds a tool definition (parameters
  as a JSON-schema list).
- `hf_chat(..., tools, tool_choice)` surfaces `tool_calls` in the returned tibble
  (list-column) when the model requests them.
- `hf_run_tools(conversation, tools = list(fn_name = r_function))` — optional
  MVP agent loop: dispatch tool calls to R functions, feed results back, repeat
  until the model returns a final message. Mark **experimental**.
- **Use cases (R):** let an LLM query a local `data.frame`, call a plotting
  function, or hit another API as a step in a pipeline.
- **Scope control:** ship `hf_tool()` + `tool_calls` exposure first; the
  auto-dispatch loop is a clearly-labeled experimental extra.

### 4.3 Streaming — `hf_chat(stream = TRUE, callback = NULL)`
- Use `httr2::req_perform_connection()` + `resp_stream_sse()` to read deltas;
  `callback(delta)` is invoked per chunk (default prints to console), full text
  reassembled and returned as the usual tibble.
- **Use cases:** responsive console/Shiny UX, long generations.
- **Second-order:** harder to test (Section 8 covers a mocked SSE stream);
  encapsulate SSE parsing in the engine so only one place handles it.

### 4.4 Vision chat — `hf_chat(image=)` and `hf_describe_image()`
- Accept image as local path, URL, or raw vector; local files become a base64
  `data:` URI in an `image_url` content block.
- `hf_describe_image(image, prompt = "Describe this image.", model = VLM default)`
  → tibble(image, description).
- **Use cases (R):** extract numbers/text from chart images or scanned tables,
  caption figures, screenshot QA.

---

## 5. Phase 2 — First-class text tasks (close the v2 gap)

These tasks exist only as legacy `hf_*_payload()` builders. Wrap them in modern,
vectorized, tibble-returning functions that mirror `hf_classify()` exactly. Low
risk, high utility, reuse the Phase 0 engine.

| Function | Returns (one row per …) | Default model (verify serverless) | Headline R use case |
|---|---|---|---|
| `hf_summarize(text, min_length, max_length, ...)` | input | `facebook/bart-large-cnn` | Condense long docs before analysis |
| `hf_translate(text, source, target, model)` | input | NLLB / opus-mt; or via instruct chat | Normalize multilingual survey text |
| `hf_ner(text, aggregation_strategy)` | **entity** (text, entity, label, score, start, end) | `dslim/bert-base-NER` | Pull people/orgs/places into tidy data |
| `hf_question_answer(question, context)` | question | `deepset/roberta-base-squad2` | Extractive QA over documents |
| `hf_table_question_answer(query, table)` | query | `google/tapas-base-finetuned-wtq` | Ask a `data.frame` questions in English |

- `hf_table_question_answer()` accepts a native R `data.frame` and converts it to
  the required dict-of-lists payload (all values coerced to character per API).
- `hf_ner()` returning character offsets enables downstream highlighting and
  joins — emphasize in docs.
- **Second-order:** translation models are often language-pair-specific; document
  the tradeoff and offer an instruct-model path (`via = "chat"`) for arbitrary
  pairs. Centralize defaults (Phase 0) and guard with `hf_check_inference()`.

---

## 6. Phase 3 — Multimodal inference (new modalities)

Introduces binary I/O; relies on Phase 0 binary helpers and optional Suggests.

| Function | Input | Returns | Default model | R use case |
|---|---|---|---|---|
| `hf_transcribe(audio, return_timestamps)` | path/URL/raw | tibble(audio, text, [chunks list-col]) | `openai/whisper-large-v3` | Interview/podcast audio → text → topic models |
| `hf_text_to_speech(text, output)` | text | file path (+ raw) | a TTS model | Generate narration/audio |
| `hf_text_to_image(prompt, output, seed)` | text | file path (+ optional `magick`) | FLUX/SD | Generate figures/assets |
| `hf_classify_image(image, top_k)` | path/URL/raw | tibble(image, label, score) | ViT | Tag/triage image collections |
| `hf_caption_image(image)` | path/URL/raw | tibble(image, caption) | BLIP | Caption figures |
| `hf_detect_objects(image)` | path/URL/raw | tibble(image, label, score, xmin, ymin, xmax, ymax) | DETR | Camera-trap counts; **ggplot bbox overlay** |

### Second/third-order effects
- **Binary handling:** task routes take raw bytes with a content-type header;
  vision *chat* takes base64 data URIs. Two code paths, both in the engine, both
  unit-tested with tiny fixtures.
- **File outputs:** TTS/image generation write files. Default to `tempfile()`
  with the right extension; never overwrite without an explicit path; return the
  path (invisibly print where it went).
- **Optional deps (Suggests, `requireNamespace`-gated like `uwot`/`arrow`):**
  `magick` (image return/decode), optionally `tuneR`/`av` for audio. No new hard
  dependency.
- **Reproducibility:** expose `seed` for generation; document non-determinism.
- **Cost/latency:** these calls are heavier; document and keep batch variants out
  of scope initially.

---

## 7. Phase 4 — Hub read + write, providers, Spaces

Moves the package from read-only discovery to read/write. Requires care.

### 7.1 Read (safe)
- `hf_hub_download(repo_id, filename, repo_type = "model", revision = "main", dest = NULL)`
  via `/resolve/`. Streams to disk; returns path.
- `hf_list_repo_files(repo_id, repo_type, revision)` via the tree API → tibble.
- Pagination for `hf_search_models()`/`hf_search_datasets()` (cursor / `Link`
  header) so users can enumerate beyond one page; add `hf_search_spaces()` and
  `hf_search_papers()`.
- `hf_list_providers(model_id)` / enrich `hf_check_inference()` using
  `/v1/models` capability + pricing metadata → tibble. **Use case:** pick the
  cheapest/fastest provider programmatically before a batch job.
- Enrich `hf_whoami()` (auth scopes, plan).

### 7.2 Write (token-gated, guarded)
- `hf_create_repo(repo_id, repo_type, private)`,
  `hf_upload_file(path, repo_id, path_in_repo, repo_type, commit_message)`,
  `hf_push_dataset(data, repo_id, ...)` (write a `data.frame` as a parquet/CSV
  dataset — reproducible-research win), `hf_delete_repo()` (**guarded**: requires
  explicit confirmation / `confirm = TRUE`, never callable in CI).
- **Use cases (R):** publish a dataset or model card straight from an analysis
  pipeline; share results reproducibly.

### 7.3 Spaces / Gradio (stretch)
- `hf_call_space(space_id, ...)` to call a hosted Gradio app's API as if it were
  an R function — large surface, schedule after 7.1/7.2 prove out.

### Second/third-order effects
- **Security posture shift.** Read-only → read-write means write-scope tokens.
  Document token scopes prominently; default everything to least privilege.
- **Destructive ops** (`hf_delete_repo`, overwriting files) need confirmation
  guards and must be excluded from automated/live CI.
- **Idempotency / commit semantics**: surface `commit_message`, sensible defaults.

---

## 8. Testing strategy (must exercise arguments + behavior, not just "it runs")

Two layers. **Unit tests run offline in CI with no token**; **live tests are
gated** and assert real semantics.

### 8.0 Characterization tests (precede Phase 0)
Lock current request bodies and output shapes of `hf_chat`/`embed`/`classify`/
`fill_mask` so the refactor provably changes nothing.

### 8.1 Unit tests (offline, deterministic)
Use `httr2::local_mocked_responses()` to inject fixture JSON/binary, and
`testthat::local_mocked_bindings()` on the request engine (extends the existing
pattern in `test-embeddings.R`/`test-classify.R`). For each new function assert:

1. **Request construction reflects arguments** (the core of the user's ask):
   - `hf_extract(schema=)` puts a correct `response_format.json_schema` in the body.
   - `hf_chat(image=)` emits a well-formed `image_url` content block.
   - `hf_chat(tools=)` serializes tool definitions; `tool_choice` is honored.
   - `hf_translate(target="fr")` / `hf_summarize(max_length=)` /
     `hf_ner(aggregation_strategy=)` / `hf_transcribe(return_timestamps=TRUE)`
     each change the outgoing request as expected.
   - Provider routing: `model="x:together"` hits the `together` route; binary
     tasks set the correct `Content-Type`.
2. **Response parsing → exact tibble contract** from representative fixtures:
   column names, types, and **one row per entity / per box / per input**; chunks
   list-column present only when requested; bounding-box columns numeric.
3. **Argument-driven output shape**, e.g. `top_k = 3` → 3 rows; multi-input → N
   rows in input order.
4. **Edge cases**: empty vector, `NA` (placeholder row, not dropped), missing
   `[MASK]`, schema/parse mismatch, provider lacking a capability → informative
   error (test the message).
5. **Vectorization & ordering** preserved across multi-input.
6. **Streaming**: feed a canned SSE byte stream; assert deltas reach the callback
   and the reassembled text matches.

Fixtures live in `tests/testthat/fixtures/` (JSON + a tiny PNG and short WAV for
binary path/encoding tests — kept small for the repo).

### 8.2 Live integration tests (gated)
Extend `scripts/test-integration.R` and add token-gated `testthat` tests
(`skip_if(Sys.getenv("HUGGING_FACE_HUB_TOKEN") == "")`) with *semantic*
assertions, in the spirit of the existing `cat-kitten > cat-car` check:
- `hf_ner("Barack Obama was born in Hawaii.")` → a PER and a LOC entity.
- `hf_translate("Hello, how are you?", target = "fr")` → contains "Bonjour"/"Salut".
- `hf_summarize(<long text>)` → shorter than input, non-empty.
- `hf_extract("Amélie is a chef in Paris.", c(name="string", city="string"))`
  → `name == "Amélie"`, `city == "Paris"`.
- `hf_classify_image(<cat.png>)` → "cat"/"tabby" in top labels.
- `hf_detect_objects(<street.png>)` → ≥1 box, numeric coords within image bounds.
- `hf_transcribe(<hello.wav>)` → expected keyword present.
- Hub write tests create a temp private repo and **clean it up**; never run in CI.

`Config/testthat/edition: 3` already set; keep `skip_on_cran()`.

---

## 9. Documentation strategy

### 9.1 Function docs (roxygen2)
Every new function: full `@param`/`@return` (spell out exact tibble columns and
types), `@seealso` to the relevant HF task doc URL, `@examples` in `\dontrun{}`
(token required), and `@family` tags to drive grouping
(`text-tasks`, `chat`, `multimodal`, `hub`). Run `devtools::document()`.

### 9.2 NAMESPACE / NEWS / version
Export new functions; add a NEWS.md section. Version: additive, no breaks →
**2.1.0** (bump per phase: 2.1, 2.2, … or one 2.1.0 at the end). Consider a
later `3.0.0` only if legacy reticulate functions are formally deprecated.

### 9.3 pkgdown site (`_pkgdown.yml` + Articles)
- New reference sections: **Structured Output & Tools**, **Multimodal**,
  **Speech & Audio**, **Vision**, **Hub: Download & Share**, **Providers**.
- New vignettes (eval = FALSE, matching house style), added to the navbar:
  1. **"From Text to Tidy Data: Structured Extraction & Tool Calling"** — the
     `hf_extract()` flagship; survey-text → columns; an agent example.
  2. **"Multimodal: Images, Audio & Speech"** — transcribe interviews → topic
     model; object detection → ggplot bbox overlay; image generation.
  3. **"Working with the Hub: Download, Upload & Share"** — publish a dataset
     from an R pipeline; provider/cost selection with `hf_list_providers()`.
  4. Update **getting-started** with a capabilities matrix and the new routes.
- Update **README.Rmd**, then re-knit `README.md` (+ `README.html`).
- A new **case study** vignette (e.g., qualitative-interview transcription →
  embeddings → clustering) to show end-to-end value.
- Rebuild site: `devtools::document()` → `pkgdown::build_site()`; commit the
  regenerated `docs/` (consistent with commit `0684dcf`).

---

## 10. Sequencing (suggested PR breakdown)

Keep PRs reviewable; each ships code + tests + docs together.

1. **PR1 — Phase 0** engine + characterization tests (no user-facing change).
2. **PR2 — Phase 2** text tasks (`summarize`/`translate`/`ner`/`question_answer`/
   `table_question_answer`) + unit/live tests + docs + getting-started update.
3. **PR3 — Phase 1** chat capabilities (`extract`, tools, streaming, vision) +
   tests + new "Structured Extraction & Tools" vignette.
4. **PR4 — Phase 3** multimodal + tests + "Multimodal" vignette + Suggests deps.
5. **PR5 — Phase 4** Hub read/write + providers + tests + "Hub" vignette.
6. **PR6 — Polish**: `hf_list_tasks()` live fetch, rate-limit backoff, pkgdown
   rebuild, NEWS, version bump, README re-knit.

A reasonable first slice to implement immediately: **PR1 + PR2** (foundation +
the four/five text tasks) — pure wins that also de-risk every later phase.

---

## 11. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Refactor changes behavior | breaks users | Characterization tests before Phase 0 |
| Provider feature variance (tools/structured/tasks) | runtime errors | Capability check via `/v1/models`; graceful fallback + clear errors; document |
| Default model availability churn | 404s | Centralize defaults; `hf_check_inference()` guard; document |
| Binary encoding bugs (audio/image) | corrupt I/O | Normalization helper + fixture-based unit tests (raw + base64 paths) |
| CI has no token | tests can't call API | All unit tests mocked; live tests gated/skipped |
| Tool-calling agent scope creep | unbounded work | Ship `hf_tool()` + `tool_calls` first; loop is experimental |
| Write ops are destructive | data loss | Token-gated, `confirm=`, never in CI; document token scopes |
| httr2 streaming API edges | flaky stream | Encapsulate SSE in the engine; mocked-stream test |
