# huggingfaceR 2.1.0

## New features

* `hf_extract()` turns unstructured text into tidy columns with chat-model structured JSON output. Pass a lightweight named schema such as `c(name = "string", score = "number")` or a full JSON Schema list, and the function returns one row per input text with one column per schema field (#55).

* `hf_chat()` now supports tool/function calling, streaming callbacks, and image inputs for vision-capable chat models. New helpers `hf_tool()`, `hf_run_tools()`, and `hf_describe_image()` make these capabilities available from R pipelines (#55).

* **Multimodal inference wrappers.** New functions add audio, image, and generation workflows: `hf_transcribe()`, `hf_text_to_image()`, `hf_classify_image()`, `hf_caption_image()`, `hf_detect_objects()`, and `hf_text_to_speech()` (#55). Live verification passed for ASR, text-to-image, image classification, captioning, and object detection; public hosted TTS provider support is currently blocked, so `hf_text_to_speech()` is documented for compatible providers or dedicated Inference Endpoints.

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
