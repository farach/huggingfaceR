# Book Outline — *huggingfaceR*

> Status: planning document. Lives in `dev/book/` so it can be edited or deleted
> freely; it is not part of the built package, the pkgdown site, or CRAN.
>
> Companion book to the `huggingfaceR` R package (v2.x): an API-first,
> tidyverse-native interface to the Hugging Face Inference Providers API and the
> Hugging Face Hub. No Python required — everything goes through `httr2`, every
> function takes character vectors / data frames and returns tibbles.

---

## 1. Title & positioning

**Working title:** *Production Machine Learning with huggingfaceR: From the Hugging Face Hub to Enterprise Deployment in R*

**Alternate titles:**
1. *huggingfaceR in Action: Tidy AI Workflows from Inference to Operations*
2. *The Tidyverse Guide to Hugging Face: Embeddings, LLMs, and Production Pipelines in R*
3. *State-of-the-Art Models, Tidy R: Accessing, Operationalizing, and Deploying Hugging Face Models*

**Positioning statement.** Most machine-learning books for R either stop at
modeling on a laptop or assume the reader will switch to Python the moment a
transformer is involved. This book takes the opposite stance: it shows R
practitioners how to reach 500,000+ state-of-the-art models on the Hugging Face
Hub directly from R — through a single, tidyverse-native package — and then how
to take a chosen model all the way to a governed, monitored, enterprise
deployment on Microsoft Foundry that real employees and customers use. It is
organized as a deliberate learning arc, from a two-line "hello world" auth call,
through core text tasks, embeddings and semantic search, chat/structured
extraction/tools, and multimodal inference, to Hub read/write, fine-tuning, and
a full production pipeline. Throughout, it treats the concerns that decide
whether an AI project survives contact with reality — reproducibility, cost,
rate limits, serverless-vs-dedicated-vs-managed tradeoffs, and governance — as
first-class material rather than footnotes.

> The full positioning argument — the thesis that R users turn AI into a
> *research instrument* rather than build AI *systems*, and the ranked catalog of
> use cases that follows from it — lives in **`use-cases-and-audience.md`**, the
> canonical source for "why / who." This outline states positioning only in
> brief and does not duplicate that catalog.

---

## 2. Audience, prerequisites, outcomes

> Canonical source: **`use-cases-and-audience.md`** (§2 audiences, §3 use cases).
> The audience summary below is the short version, kept here so the structure is
> self-contained; deepen or revise audiences in that file, not this one.

**Primary audience**
- R data scientists / analysts comfortable with the tidyverse who want to use
  modern (transformer / LLM) models without leaving R or learning Python.
- Analytics engineers and ML practitioners in enterprises who must move a model
  from "works in a notebook" to "deployed, monitored, and governed."

**Secondary audience**
- Data-science managers and architects evaluating an R-centric path to
  generative AI / LLM features.
- Researchers (social science, health, qualitative) who need reproducible,
  scriptable access to NLP and multimodal models.

**Prerequisites**
- Working knowledge of R and the tidyverse (`dplyr`, `tidyr`, `purrr`,
  pipes, tibbles). Intermediate, not expert.
- Basic command line + a code editor (RStudio / Positron / VS Code).
- Helpful but not required: exposure to `httr2`/APIs, git/GitHub, and a
  conceptual grasp of embeddings and classification.
- **No Python and no GPU required** for the bulk of the book; the fine-tuning
  and Foundry chapters note where managed compute enters.

**By the end, the reader will be able to:**
1. Authenticate to Hugging Face from R and manage tokens/scopes safely.
2. Run every core inference task — classification, embeddings, chat, generation,
   structured extraction, tool calling, multimodal — and get tidy tibbles back.
3. Build a semantic search / clustering / topic pipeline over their own data.
4. Use an LLM to turn unstructured text into analysis-ready columns reliably.
5. Discover, evaluate, download, and publish models and datasets on the Hub.
6. Choose between serverless inference, a dedicated Inference Endpoint, and a
   managed platform on cost, latency, and governance grounds.
7. Fine-tune (or select an already-fine-tuned) model and register it.
8. Deploy that model to Microsoft Foundry with security, monitoring, and access
   control, and wire it into an employee/customer-facing application.
9. Reason about reproducibility, cost, rate limits, and governance at every step.

---

## 3. The recurring case study (threads through the whole book)

**"Northwind Support Intelligence" — an enterprise customer-support analytics &
assistant program.** A fictional mid-size company, *Northwind*, has years of
multilingual customer-support tickets, call recordings, product images from
return requests, and a knowledge base. Each part of the book advances one
realistic mission against this single dataset, so the reader builds one coherent
system rather than disconnected demos.

- **Why this case study:** support data is text-heavy, multilingual, multimodal
  (audio calls, return-photo images), has obvious business value, and naturally
  exercises every package capability — classification (triage), embeddings
  (dedup / search), extraction (structured ticket fields), chat (drafted
  replies), multimodal (call transcription, return-photo triage), Hub (publish a
  curated dataset / register a fine-tune), and production (a governed assistant).
- **Data provenance:** built from public, license-clean sources so it is
  reproducible — a public support/ticket or review dataset loaded via
  `hf_load_dataset()`, augmented with a small synthetic, shippable sample stored
  with the book. A short "about the data" section documents licensing and PII
  handling.
- **Mission arc (mirrors the parts):**
  - Triage and route incoming tickets (classification / zero-shot).
  - Find duplicate and related tickets; build a searchable knowledge base
    (embeddings + nearest neighbors + clustering + topics).
  - Extract structured fields and draft replies (chat, extraction, tools).
  - Handle calls and return photos (transcription, image classification/caption).
  - Curate and publish a clean dataset; select/fine-tune a triage model (Hub).
  - Deploy the triage + assistant system to Foundry for agents and customers,
    with monitoring and governance.

Mini side-cases reuse the existing vignette domains (the Anthropic Economic Index
and OpenAI GDPval analyses) as **standalone "applied interludes"** so the
package's published examples stay reachable from the book without duplicating
them verbatim.

---

## 4. Part / chapter structure

Eight parts, ~24 chapters. Each chapter lists: **Objectives**, **Functions
featured**, **Running example** (Northwind unless noted), and **Exercises**.
Most chapters end with a short **"Second-order concerns"** box
(reproducibility / cost / rate limits / governance) so those themes recur
explicitly rather than living only in the appendices.

---

### Part I — Foundations: Hello, Hugging Face from R

#### Chapter 1 — Why huggingfaceR (and why API-first)
- **Objectives:** Understand the package's philosophy (API-first, no Python,
  tidyverse-native, tibbles in/out); place it in the R ML landscape; understand
  serverless inference vs. dedicated endpoints vs. managed platforms at a high
  level; see the book's destination (the Foundry pipeline).
- **Functions:** none yet — conceptual; preview of the function families.
- **Running example:** a 10-line end-to-end taste (classify → embed → chat) on a
  handful of Northwind tickets to show the whole shape of the package.
- **Exercises:** install the package; sketch which capability maps to each
  Northwind mission; identify one task in the reader's own work that fits.

#### Chapter 2 — Setup, authentication, and your first call
- **Objectives:** Create and store a token securely; verify identity and scopes;
  understand environment variables vs. keyring; make the canonical first request.
- **Functions:** `hf_set_token`, `hf_whoami`, `hf_check_inference`.
- **Running example:** authenticate, confirm whoami, check that the default
  triage model is available before relying on it.
- **Exercises:** configure a token via `.Renviron`; run `hf_whoami()`; check
  inference availability for two models; intentionally use a bad token and read
  the error.
- **Second-order:** never commit tokens; least-privilege scopes; what
  `hf_check_inference()` saves you from later.

#### Chapter 3 — The tidy data contract
- **Objectives:** Internalize the invariants that make the package composable —
  vectorized, order-preserving, one row per logical result, `NA` in → placeholder
  out, tibbles that `unnest()` and join cleanly; how requests flow through
  `httr2` (auth, retry, `Retry-After`) under the hood.
- **Functions:** `hf_classify` (as the worked archetype), `%>%` / native pipe.
- **Running example:** classify a vector of tickets, join results back to the
  source data frame, observe ordering and `NA` handling.
- **Exercises:** predict output shape before running; feed a vector with an `NA`;
  pipe a classification into a `dplyr` summary.
- **Second-order:** why stable column contracts matter for reproducible pipelines.

---

### Part II — Core Text Tasks

#### Chapter 4 — Text classification and triage
- **Objectives:** Run sentiment / topic classification and zero-shot
  classification with custom labels; choose models; interpret scores.
- **Functions:** `hf_classify`, `hf_classify_zero_shot`,
  `hf_search_models`, `hf_model_info`.
- **Running example:** triage Northwind tickets into urgency × category using
  zero-shot labels; route negatives to a priority queue.
- **Exercises:** compare a fine-tuned sentiment model vs. zero-shot; design a
  label set for the reader's own domain; measure agreement with a small hand-
  labeled sample.
- **Second-order:** zero-shot cost vs. a small fine-tuned classifier; when to
  switch.

#### Chapter 5 — Generation, completion, and fill-mask
- **Objectives:** Generate and complete text; understand decoding parameters;
  use fill-mask for cloze / data-augmentation tasks.
- **Functions:** `hf_generate`, `hf_fill_mask`.
- **Running example:** generate canned-response templates; use fill-mask to
  probe a model's domain knowledge of Northwind product terms.
- **Exercises:** sweep temperature / max tokens; compare two generation models;
  build a templated reply generator.
- **Second-order:** non-determinism and `seed`; reproducibility of generations.

#### Chapter 6 — The text-task toolbelt: summarize, translate, NER, QA
- **Objectives:** Apply the first-class text tasks; understand which return one
  row per input vs. one row per entity; handle multilingual data.
- **Functions (planned):** `hf_summarize`, `hf_translate`, `hf_ner`,
  `hf_question_answer`, `hf_table_question_answer`.
- **Running example:** normalize multilingual Northwind tickets with
  `hf_translate`; pull people/orgs/products with `hf_ner`; summarize long
  threads; ask a tickets `data.frame` questions in English with
  `hf_table_question_answer`.
- **Exercises:** build a tidy entity table and join offsets back for highlighting;
  compare a language-pair model vs. an instruct model for translation; QA over a
  product manual.
- **Second-order:** language-pair-specific models vs. instruct fallback; latency
  of summarization on long inputs.

#### Chapter 7 — Batch processing and scale
- **Objectives:** Process large datasets with parallel requests and disk
  checkpointing; chunk inputs; resume safely after failures.
- **Functions:** `hf_classify_batch`, `hf_classify_chunks`,
  `hf_classify_zero_shot_batch`, `hf_embed_batch`, `hf_embed_chunks`,
  `hf_read_chunks`.
- **Running example:** classify the full Northwind ticket history with
  checkpointing; resume after a simulated interruption.
- **Exercises:** tune chunk size; benchmark serial vs. batched; recover a partial
  run from disk.
- **Second-order:** rate limits and `Retry-After`; cost of reprocessing; why
  checkpointing is a reproducibility and cost control, not just convenience.

---

### Part III — Embeddings & Semantic Understanding

#### Chapter 8 — Embeddings and similarity
- **Objectives:** Turn text into vectors; compute similarity; understand model
  choice and dimensionality.
- **Functions:** `hf_embed`, `hf_similarity`, `hf_embed_text`.
- **Running example:** embed Northwind tickets; find near-duplicate submissions
  via pairwise similarity.
- **Exercises:** compare two embedding models on a dedup task; build a similarity
  matrix; threshold for duplicates.
- **Second-order:** embedding cost at scale; caching embeddings for reuse.

#### Chapter 9 — Semantic search and nearest neighbors
- **Objectives:** Build a retrieval workflow over a corpus; rank by semantic
  relevance; lay groundwork for retrieval-augmented generation (RAG).
- **Functions:** `hf_embed`, `hf_nearest_neighbors`.
- **Running example:** a knowledge-base search — given a new ticket, retrieve the
  most relevant past resolutions.
- **Exercises:** build a small semantic search function; evaluate retrieval with
  a handful of labeled query/answer pairs; persist the index.
- **Second-order:** when to precompute/store vectors vs. embed on the fly; cost.

#### Chapter 10 — Clustering, topics, and visualization
- **Objectives:** Discover structure in unlabeled text; reduce dimensionality;
  visualize and name clusters.
- **Functions:** `hf_cluster_texts`, `hf_extract_topics`, `hf_embed_umap`.
- **Running example:** cluster Northwind tickets to discover emergent issue
  themes; label topics; plot a UMAP map of the ticket space.
- **Exercises:** choose k / inspect cluster quality; name topics; produce a
  ggplot of the embedding space colored by category.
- **Second-order:** reproducibility of UMAP (`seed`); stability of cluster labels
  across runs.

#### Chapter 11 — Embeddings in modeling (tidymodels & tidytext)
- **Objectives:** Use embeddings as features in a predictive model; integrate
  with `recipes`/`tidymodels` and `tidytext` workflows.
- **Functions:** `step_hf_embed`, `hf_embed_text`, tidytext integration.
- **Running example:** predict ticket escalation from embedding features inside a
  tidymodels workflow with proper resampling.
- **Exercises:** add `step_hf_embed` to a recipe; compare embeddings vs. bag-of-
  words features; tune and evaluate.
- **Second-order:** baking embeddings into a recipe means inference at predict
  time — cost/latency implications for deployment (foreshadows Part VIII).

---

### Part IV — Chat, Structured Extraction, and Tools

#### Chapter 12 — Chat and multi-turn conversations
- **Objectives:** Hold single- and multi-turn conversations; manage system
  prompts and conversation state; understand the OpenAI-compatible chat route and
  provider selection.
- **Functions:** `hf_chat`, `hf_conversation`, `chat` (S3 method).
- **Running example:** a Northwind support copilot that drafts replies given a
  ticket and retrieved context (RAG using Part III).
- **Exercises:** build a conversation object and carry context across turns;
  craft a system prompt for tone/policy compliance; stream a long response
  (planned `stream = TRUE`).
- **Second-order:** token/cost accounting per turn; provider capability variance.

#### Chapter 13 — Structured extraction: from text to tidy columns
- **Objectives:** Reliably turn unstructured text into typed columns with JSON-
  schema-constrained output — the flagship "make any LLM output safe to
  `mutate()` on" workflow.
- **Functions (planned):** `hf_extract` (schema as JSON Schema or lightweight
  field spec), with `hf_check_inference` / provider capability pre-check.
- **Running example:** extract `{product, issue_type, sentiment, refund_requested,
  severity}` from free-text Northwind tickets into one tidy row per ticket.
- **Exercises:** design a schema for the reader's own free text; handle a provider
  that lacks structured output; validate extracted fields against ground truth.
- **Second-order:** `supports_structured_output` varies by provider; fallback to
  `json_object` + validation; reproducibility via `seed` + `strict`.

#### Chapter 14 — Tool / function calling and lightweight agents
- **Objectives:** Define tools, expose `tool_calls`, and (experimentally) run an
  R-function dispatch loop; understand the boundary between assistive and
  autonomous behavior.
- **Functions (planned):** `hf_tool`, `hf_chat(tools=, tool_choice=)`,
  `hf_run_tools` (experimental).
- **Running example:** let the support copilot call R functions — query the
  tickets `data.frame`, look up an order, compute a refund — as pipeline steps.
- **Exercises:** define a tool schema; expose and inspect `tool_calls`; wire one
  tool to a real R function; add a guardrail that requires confirmation.
- **Second-order:** governance of tool execution (what an agent is allowed to
  do); cost/looping limits; testing agent loops with mocked responses.

---

### Part V — Multimodal Inference

#### Chapter 15 — Vision chat and image understanding
- **Objectives:** Send images to a vision-language model; extract text/numbers
  from images; caption and QA over figures.
- **Functions (planned):** `hf_chat(image=)`, `hf_describe_image`,
  `hf_caption_image`.
- **Running example:** triage Northwind return-request photos — describe the
  product condition and flag damage from customer-submitted images.
- **Exercises:** read a value off a chart image; caption a batch of figures;
  compare two VLMs.
- **Second-order:** base64 data-URI vs. raw-byte routes; image size → cost.

#### Chapter 16 — Image tasks: classification, captioning, detection
- **Objectives:** Run dedicated vision tasks returning tidy tibbles; overlay
  detections on plots.
- **Functions (planned):** `hf_classify_image`, `hf_caption_image`,
  `hf_detect_objects` (one row per bounding box, numeric coords).
- **Running example:** auto-tag return photos by product category; detect objects
  and overlay bounding boxes with ggplot.
- **Exercises:** build a ggplot bbox overlay; tag an image folder; evaluate top-k
  labels against a small labeled set.
- **Second-order:** heavier/slower calls; when to batch vs. on-demand.

#### Chapter 17 — Speech and audio
- **Objectives:** Transcribe audio with timestamps; generate speech; feed
  transcripts into downstream text pipelines.
- **Functions (planned):** `hf_transcribe` (with `return_timestamps`),
  `hf_text_to_speech`.
- **Running example:** transcribe Northwind call recordings, then run the
  Part II/III pipeline (classify → embed → topic) over the transcripts; generate
  a spoken summary.
- **Exercises:** transcribe with and without timestamps; build a topic model over
  transcripts; produce an audio digest.
- **Second-order:** file I/O hygiene (`tempfile`, never overwrite without an
  explicit path); audio length → cost/latency; PII in recordings → governance.

#### Chapter 18 — Image generation
- **Objectives:** Generate images from prompts; control reproducibility with
  seeds; manage file outputs.
- **Functions (planned):** `hf_text_to_image` (with `seed`, `output`).
- **Running example:** generate illustrative assets for a Northwind knowledge-base
  article.
- **Exercises:** fix a seed and reproduce; sweep prompts; save to a chosen path.
- **Second-order:** non-determinism without a seed; cost of generation; usage
  rights / governance of generated assets.

---

### Part VI — The Hugging Face Hub: Discover, Download, Share

#### Chapter 19 — Discovering models and datasets
- **Objectives:** Search and evaluate models and datasets; read model cards and
  metadata; verify inference availability and supported tasks before committing.
- **Functions:** `hf_search_models`, `hf_model_info`, `hf_check_inference`,
  `hf_list_tasks`, `hf_search_datasets`, `hf_dataset_info`, `hf_load_dataset`.
- **Running example:** find a strong multilingual classification model and a
  public support-ticket dataset for Northwind; load rows into a tibble.
- **Exercises:** compare candidate models on size/license/downloads; load a
  dataset slice; check a model's serverless availability.
- **Second-order:** license and model-card due diligence as governance; dataset
  provenance and PII.

#### Chapter 20 — Reading from and writing to the Hub
- **Objectives:** Download files/revisions; list repo contents; create repos and
  publish datasets/files reproducibly from an R pipeline; understand write scopes
  and destructive-op guards.
- **Functions (planned):** `hf_hub_download`, `hf_list_repo_files`,
  `hf_create_repo`, `hf_upload_file`, `hf_push_dataset`,
  `hf_delete_repo` (guarded), `hf_list_providers`.
- **Running example:** curate the cleaned/labeled Northwind triage dataset and
  publish it (private) with a data card straight from the analysis script.
- **Exercises:** download a specific revision; push a `data.frame` as a dataset;
  list providers and pick the cheapest/fastest for a batch job.
- **Second-order:** write-scope tokens & least privilege; commit semantics /
  idempotency; never run destructive ops in CI; pinning revisions for
  reproducibility.

---

### Part VII — Selecting and Fine-Tuning a Model

#### Chapter 21 — Choosing: prompt, fine-tune, or pick an existing fine-tune
- **Objectives:** Decide when zero-shot/prompting suffices vs. when fine-tuning
  pays off; find existing fine-tuned models; frame the cost/benefit.
- **Functions:** `hf_search_models`, `hf_model_info`, `hf_classify` /
  `hf_classify_zero_shot` (as baselines), `hf_check_inference`.
- **Running example:** establish a zero-shot triage baseline on Northwind, then
  decide whether a fine-tune is justified by error analysis.
- **Exercises:** build an evaluation harness; compute the break-even point where a
  fine-tune beats per-call zero-shot cost.
- **Second-order:** total cost of ownership: data labeling + training + serving.

#### Chapter 22 — Fine-tuning a model for the task
- **Objectives:** Prepare a labeled dataset, run a fine-tune, evaluate, and
  register the resulting model on the Hub. (This chapter is explicit that
  training itself uses managed/AutoTrain or GPU compute outside serverless
  inference; the R package's role is data prep, dataset publishing, evaluation,
  and registration.)
- **Functions:** `hf_push_dataset`, `hf_create_repo`/`hf_upload_file`,
  `hf_load_dataset`, `hf_classify` / `hf_extract` for evaluation; plus a clearly
  labeled note on the legacy reticulate path for local training.
- **Running example:** fine-tune a compact triage classifier on the curated
  Northwind dataset (via AutoTrain / managed compute), evaluate against the
  zero-shot baseline from Ch. 21, and push it to the Hub.
- **Exercises:** split/prepare training data reproducibly; run an evaluation and
  produce a model card; register the fine-tune.
- **Second-order:** training reproducibility (seeds, data versioning); the
  serverless-vs-dedicated decision now that you own a model; governance of
  training data.

---

### Part VIII — Operationalization: The Enterprise Pipeline

> The arc's destination. Hugging Face → find/fine-tune model → Microsoft Foundry
> → secure deployment + monitoring → employee/customer application. R remains the
> orchestration and analytics layer; Foundry is the managed serving/governance
> layer.

#### Chapter 23 — Architecting for production
- **Objectives:** Compare serverless inference, a dedicated Hugging Face
  Inference Endpoint, and a managed platform (Microsoft Foundry) on cost,
  latency, scaling, isolation, compliance, and operational burden; choose per
  workload; design the Northwind architecture.
- **Functions:** `endpoint_url` usage across `hf_*` functions (dedicated
  endpoints), `hf_list_providers`, `hf_check_inference`.
- **Running example:** point the existing Northwind code at a dedicated endpoint
  by changing only `endpoint_url`; draw the target Foundry architecture.
- **Exercises:** build a decision matrix for three workloads (batch triage,
  real-time copilot, occasional image triage); estimate monthly cost for each
  serving option.
- **Second-order:** vendor lock-in vs. portability; the cost cliff between
  per-call and always-on serving.

#### Chapter 24 — Deploying to Microsoft Foundry
- **Objectives:** Move the chosen/fine-tuned model into Microsoft Foundry; stand
  up a secure, scalable endpoint; configure networking, secrets, and identity.
- **Functions:** R-side client code (`httr2`) calling the Foundry endpoint, with
  `huggingfaceR` patterns reused where the API surface is compatible; token/
  secret management mirroring Ch. 2.
- **Running example:** deploy the Northwind triage model + support copilot to a
  Foundry endpoint; call it from R end to end.
- **Exercises:** deploy a model; secure the endpoint (private networking +
  managed identity); call it from an R script with managed secrets.
- **Second-order:** secret management and least privilege at production scale;
  region/data-residency choices.

#### Chapter 25 — Monitoring, evaluation, and governance in production
- **Objectives:** Instrument the deployed system — latency, cost, error rates,
  drift, and output quality; set up evaluation and alerting; establish governance
  (access control, audit, content safety, PII, model/data lineage).
- **Functions:** R analytics over collected logs/metrics (the tidyverse + the
  `hf_*` evaluation patterns from Ch. 21–22), scheduled re-evaluation jobs.
- **Running example:** a monitoring dashboard and a scheduled drift check that
  re-scores a sample of live Northwind traffic and flags degradation.
- **Exercises:** define SLOs and alert thresholds; build a drift detector; write
  a governance checklist for the deployment.
- **Second-order:** the full convergence of cost / reproducibility / rate limits
  / governance — this chapter is where every recurring theme is paid off.

#### Chapter 26 — The application: serving employees and customers
- **Objectives:** Wire the deployed model into a real application — an internal
  agent-assist tool and a customer-facing assistant; close the loop with feedback
  that feeds re-training.
- **Functions:** Shiny / Plumber front-ends calling the Foundry endpoint;
  `huggingfaceR` for any auxiliary inference; the full Northwind pipeline as a
  scheduled + interactive system.
- **Running example:** an internal Shiny "agent copilot" (suggested triage + draft
  reply with retrieval) and a customer-facing assistant, both backed by the
  governed endpoint; feedback captured for the next fine-tune cycle.
- **Exercises:** build the Shiny copilot; add human-in-the-loop review; design the
  feedback-to-retraining loop.
- **Second-order:** human oversight and escalation; responsible-AI UX; closing
  the lifecycle (deploy → observe → improve).

---

## 5. Appendices

- **Appendix A — Token & auth setup, deep dive.** Creating tokens, fine-grained
  vs. classic scopes, `.Renviron` vs. keyring vs. CI secrets, rotating tokens,
  org tokens, and the difference between read, inference, and write scopes.
- **Appendix B — Provider selection & cost.** Reading `/v1/models` capability and
  pricing metadata via `hf_list_providers`; choosing on
  `supports_tools` / `supports_structured_output` / `pricing` / `context_length` /
  latency / throughput; worked cost estimates for serverless vs. dedicated vs.
  Foundry; cost-control patterns (caching, batching, checkpointing, model
  right-sizing).
- **Appendix C — Troubleshooting inference availability & errors.** Using
  `hf_check_inference`; reading HF error bodies; handling 429/503 and
  `Retry-After`; cold starts / model loading; default-model churn and how to pin
  models; provider-capability mismatches and fallbacks.
- **Appendix D — Reproducibility cookbook.** Seeds and determinism limits;
  pinning model and dataset revisions; caching embeddings/results; recording the
  exact request; `renv` for package versions; environment capture.
- **Appendix E — Glossary.** Inference provider, serverless vs. dedicated
  endpoint, embedding, zero-shot, fine-tuning, JSON Schema / structured output,
  tool/function calling, VLM, RAG, drift, token (auth) vs. token (LLM), context
  length, throughput, model card, governance.
- **Appendix F — Legacy / Python interop.** When and how to use the legacy
  reticulate-backed functions (`hf_load_*`, `hf_ez_*`), and why the book defaults
  to the API-first path.
- **Appendix G — Migration & install.** Installing from GitHub/CRAN, version
  notes, and migrating from earlier package versions.

---

## 6. How the book complements the pkgdown site & vignettes

The book, the pkgdown reference, and the vignettes have **different, non-
overlapping jobs**, and the book should explicitly point at the other two rather
than restate them.

- **pkgdown function reference** = the *dictionary*: authoritative per-function
  `@param`/`@return`/argument contracts and exact tibble columns. The book never
  reproduces full argument tables; it links to them ("see `?hf_extract`").
- **Vignettes** = *task recipes*: short, self-contained "how do I do X" articles
  (`getting-started`, `text-classification`, `embeddings-and-similarity`,
  `llm-chat-and-generation`, `hub-datasets-and-modeling`) plus applied case
  studies (Anthropic Economic Index, OpenAI GDPval). They are copy-pasteable and
  scoped to one capability.
- **The book** = the *curriculum / narrative*: a sequenced learning arc with one
  end-to-end case study, the conceptual "why," the tradeoff reasoning
  (serverless vs. dedicated vs. managed; cost; governance), and the production
  pipeline that no single vignette covers. It teaches judgment, not just syntax.

**Boundary rules to avoid duplication:**
1. When a vignette already nails a recipe, the book *links* to it and adds the
   surrounding decision context instead of re-deriving the code.
2. The book's running example (Northwind) is **distinct** from the vignette
   datasets so material does not collide; the existing case studies appear only
   as short "applied interludes" with links.
3. Anything that is purely reference (signatures, full column lists) lives in
   pkgdown and is cited, never copied.
4. New capabilities should land first as a vignette (recipe) and roxygen
   reference; the book then weaves them into the arc — keeping the three layers
   in sync.

---

## 7. Tooling recommendation: build with Quarto Book

**Recommendation: Quarto Book** (`quarto-book` project type), not bookdown.

**Rationale**
- **Active direction of travel.** Quarto is the actively developed successor to
  R Markdown / bookdown from the same team (Posit); new R technical books are
  standardizing on it. Choosing Quarto future-proofs the book.
- **Native to this package's style.** The package's vignettes are knitr/R
  Markdown with `eval = FALSE` (token-gated) code; Quarto runs the same
  `knitr` engine, so existing chunks and the house style port over directly, and
  freeze/cache features (`execute: freeze`) let token-dependent chapters render
  reproducibly in CI without live API calls.
- **Multi-format from one source.** HTML book (primary), PDF, and EPUB from the
  same `.qmd` sources, with cross-references, callouts (ideal for the recurring
  "Second-order concerns" boxes), and code annotations.
- **Cohesion with the docs site.** A Quarto book deploys cleanly to GitHub Pages
  alongside the existing pkgdown site (e.g., under a `/book/` path), keeping one
  publishing pipeline and visual identity.
- **Tooling fit.** First-class in RStudio/Positron and VS Code; easy CI via
  `quarto render` + GitHub Actions.

**When bookdown would still make sense:** an existing all-bookdown toolchain or a
hard dependency on a bookdown-only output feature — neither applies here. Default
to Quarto.

**Suggested project layout (under `dev/book/`):**
```
dev/book/
  _quarto.yml          # book config: parts, chapters, formats, freeze
  index.qmd            # preface / how to read this book
  part-1-foundations/  # one .qmd per chapter, grouped by part
  ...
  appendices/
  data/                # shippable Northwind sample + "about the data"
  _freeze/             # cached executed chunks (committed for reproducible CI)
```
Pin package versions with `renv`, freeze token-dependent chunks, and render in CI
without secrets so the book is reproducible by readers and contributors alike.
