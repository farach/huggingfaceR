# Getting Started with huggingfaceR

## What is huggingfaceR?

huggingfaceR provides R users with access to machine learning models
hosted on the [Hugging Face Hub](https://huggingface.co) via the Hugging
Face Inference API. You can perform natural language processing tasks –
classification, embeddings, chat, text generation, and more – without
installing Python or managing model weights locally. Note that the
Inference API serves a curated subset of the 500,000+ models on the Hub;
not every model is available for serverless inference.

Key design principles:

- **No Python required.** Authentication and a network connection are
  all you need.
- **Tidyverse-native.** Every function accepts character vectors and
  returns tibbles.
- **Pipe-friendly.** Functions compose naturally with dplyr, tidyr, and
  the rest of the tidyverse.

## Capability matrix

| Workflow | Start with | Returns |
|----|----|----|
| Sentiment and labels | [`hf_classify()`](https://farach.github.io/huggingfaceR/dev/reference/hf_classify.md), [`hf_classify_zero_shot()`](https://farach.github.io/huggingfaceR/dev/reference/hf_classify_zero_shot.md) | one tibble row per text or label |
| Embeddings and search | [`hf_embed()`](https://farach.github.io/huggingfaceR/dev/reference/hf_embed.md), [`hf_similarity()`](https://farach.github.io/huggingfaceR/dev/reference/hf_similarity.md), [`hf_nearest_neighbors()`](https://farach.github.io/huggingfaceR/dev/reference/hf_nearest_neighbors.md) | vectors, similarities, nearest rows |
| Chat and agents | [`hf_chat()`](https://farach.github.io/huggingfaceR/dev/reference/hf_chat.md), [`hf_conversation()`](https://farach.github.io/huggingfaceR/dev/reference/hf_conversation.md), [`hf_tool()`](https://farach.github.io/huggingfaceR/dev/reference/hf_tool.md) | assistant messages and tool calls |
| Structured extraction | [`hf_extract()`](https://farach.github.io/huggingfaceR/dev/reference/hf_extract.md) | one tidy row per input with schema columns |
| Text tasks | [`hf_summarize()`](https://farach.github.io/huggingfaceR/dev/reference/hf_summarize.md), [`hf_translate()`](https://farach.github.io/huggingfaceR/dev/reference/hf_translate.md), [`hf_ner()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ner.md), [`hf_question_answer()`](https://farach.github.io/huggingfaceR/dev/reference/hf_question_answer.md) | task-specific tidy columns |
| Multimodal | [`hf_transcribe()`](https://farach.github.io/huggingfaceR/dev/reference/hf_transcribe.md), [`hf_text_to_image()`](https://farach.github.io/huggingfaceR/dev/reference/hf_text_to_image.md), [`hf_classify_image()`](https://farach.github.io/huggingfaceR/dev/reference/hf_classify_image.md) | transcripts, files/raw bytes, image labels |
| Hub workflows | [`hf_hub_download()`](https://farach.github.io/huggingfaceR/dev/reference/hf_hub_download.md), [`hf_list_providers()`](https://farach.github.io/huggingfaceR/dev/reference/hf_list_providers.md), [`hf_push_dataset()`](https://farach.github.io/huggingfaceR/dev/reference/hf_push_dataset.md) | files, provider metadata, guarded uploads |

## Installation

Install the released version from CRAN or the development version from
GitHub:

``` r

# From CRAN
install.packages("huggingfaceR")

# Development version
# install.packages("devtools")
devtools::install_github("farach/huggingfaceR")
```

## Authentication

Hugging Face requires an API token for inference requests. To obtain
one:

1.  Create a free account at [huggingface.co](https://huggingface.co).
2.  Follow the [Hugging Face access tokens
    documentation](https://huggingface.co/docs/hub/security-tokens).
3.  Generate a token with the **Make calls to Inference Providers**
    permission. A read-only token is enough for Hub browsing, but
    inference requests are rejected without the Inference Providers
    permission.

Inference is billed per request. PRO, Team, and Enterprise plans include
a small monthly credit, while Free accounts include none and must buy
credits first. See [Inference Providers
pricing](https://huggingface.co/docs/inference-providers/pricing). Hub
search and dataset loading do not use credits.

Then configure the token in R:

``` r

library(huggingfaceR)

# Store your token persistently (writes HF_TOKEN to .Renviron)
hf_set_token("hf_your_token_here", store = TRUE)

# Verify authentication
hf_whoami()
```

After storing the token, it is loaded automatically in future sessions.
huggingfaceR reads `HF_TOKEN` and falls back to the legacy
`HUGGING_FACE_HUB_TOKEN`, so tokens configured for the `hf` CLI or the
Python client are picked up automatically.

If you belong to a Team or Enterprise organization and want usage billed
to the organization rather than to you personally, set `HF_BILL_TO`:

``` r

Sys.setenv(HF_BILL_TO = "my-org")
```

## Quick Tour

### Classify Text

Assign labels to text using pre-trained classifiers. The default model
performs sentiment analysis, but you can supply any classification model
available on the Hugging Face Inference API. Not all models on the Hub
support serverless inference — use `hf_check_inference(model_id)` to
verify.

``` r

# Sentiment analysis
hf_classify("I love using R for data science!")

# Zero-shot classification with custom labels (no training needed)
hf_classify_zero_shot(
  "NASA launches new Mars rover",
  labels = c("science", "politics", "sports", "entertainment")
)
```

### Generate Embeddings

Convert text into dense numeric vectors that capture semantic meaning.
Similar texts produce similar vectors.

``` r

sentences <- c(
  "The cat sat on the mat",
  "A feline rested on the rug",
  "The dog played in the park"
)

embeddings <- hf_embed(sentences)
embeddings

# Compute pairwise cosine similarity
hf_similarity(embeddings)
```

### Chat with a Language Model

Interact with open-source large language models through a simple
interface.

``` r

# Single question
hf_chat("What is the tidyverse?", max_tokens = 60)

# Guide the model with a system prompt
hf_chat(
  "Explain logistic regression in two sentences.",
  system = "You are a statistics instructor. Use plain language.",
  max_tokens = 80
)
```

### Explore the Hub

Search for models and load datasets directly into R without leaving your
session.

``` r

# Find popular text classification models
hf_search_models(task = "text-classification", limit = 5)

# Load dataset rows into a tibble
imdb <- hf_load_dataset("imdb", split = "train", limit = 100)
head(imdb)
```

### Extract Structured Data

Turn messy prose into analysis-ready columns.

``` r

hf_extract(
  "Amelie is a chef in Paris who mentions burnout.",
  c(name = "string", occupation = "string", city = "string", theme = "string")
)
```

### Work with Images and Audio

``` r

audio <- "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"
image <- "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"

transcript <- hf_transcribe(audio, return_timestamps = "word")
substr(transcript$text, 1, 120)

hf_classify_image(image, top_k = 3)
hf_detect_objects(image, threshold = 0.5) |>
  filter(label == "cat")
```

## Working with Data Frames

All huggingfaceR functions accept character vectors and return tibbles,
so they integrate naturally into tidyverse pipelines.

``` r

reviews <- tibble(

  product_id = 1:5,
  review = c(
    "Excellent quality, highly recommend!",
    "Broke after one week of use",
    "Good value for the price",
    "Disappointing, not as advertised",
    "Love it! Will buy again"
  )
)

# Add sentiment scores
reviews |>
  mutate(sentiment = hf_classify(review)) |>
  unnest(sentiment) |>
  select(product_id, review, label, score)
```

## Next Steps

For deeper coverage of each capability, see the following vignettes:

- [Text Classification and Zero-Shot
  Labeling](https://farach.github.io/huggingfaceR/dev/articles/text-classification.md)
  – sentiment analysis, custom categories, and data frame workflows.
- [Embeddings, Similarity, and Semantic
  Search](https://farach.github.io/huggingfaceR/dev/articles/embeddings-and-similarity.md)
  – vector representations, clustering, topic modeling, and
  visualization.
- [Chat, Conversations, and Text
  Generation](https://farach.github.io/huggingfaceR/dev/articles/llm-chat-and-generation.md)
  – LLM interaction patterns, multi-turn conversations, and fill-mask.
- [Structured Extraction and Tool
  Calling](https://farach.github.io/huggingfaceR/dev/articles/structured-extraction-and-tools.md)
  – JSON-schema extraction, streaming, and R-backed tool calls.
- [Multimodal: Images, Audio, and
  Speech](https://farach.github.io/huggingfaceR/dev/articles/multimodal-images-audio-speech.md)
  – transcription, image generation, captioning, and object detection.
- [Hub Discovery, Datasets, and Tidymodels
  Integration](https://farach.github.io/huggingfaceR/dev/articles/hub-datasets-and-modeling.md)
  – searching models, loading data, and building ML pipelines with
  embeddings.
- [Working with the Hub: Download, Upload, and
  Share](https://farach.github.io/huggingfaceR/dev/articles/hub-download-upload-share.md)
  – file downloads, provider metadata, and guarded repository writes.
- [Analyzing the Anthropic Economic
  Index](https://farach.github.io/huggingfaceR/dev/articles/anthropic-economic-index.md)
  – a research-oriented case study using embeddings, clustering, and
  zero-shot classification on real-world AI adoption data.

**For production workloads:** The classification and embeddings
vignettes cover batch processing functions
([`hf_embed_batch()`](https://farach.github.io/huggingfaceR/dev/reference/hf_embed_batch.md),
[`hf_classify_batch()`](https://farach.github.io/huggingfaceR/dev/reference/hf_classify_batch.md),
etc.) that use parallel requests and disk checkpointing for processing
thousands of texts efficiently.

## Using Dedicated Inference Endpoints

By default, huggingfaceR sends requests through the Hugging Face router
to Inference Providers. Task functions such as
[`hf_classify()`](https://farach.github.io/huggingfaceR/dev/reference/hf_classify.md),
[`hf_embed()`](https://farach.github.io/huggingfaceR/dev/reference/hf_embed.md),
and
[`hf_summarize()`](https://farach.github.io/huggingfaceR/dev/reference/hf_summarize.md)
use the first-party `hf-inference` provider, which has focused on
CPU-friendly models since July 2025. Chat and generation go to
third-party providers instead. If you need a model that no provider
serves, or you need dedicated capacity for production workloads, you can
deploy a [Dedicated Inference
Endpoint](https://huggingface.co/docs/inference-endpoints) and point
huggingfaceR at it with the `endpoint_url` parameter.

``` r

# Check whether an Inference Provider serves the model
hf_check_inference("my-org/my-custom-model")

# If not, deploy a Dedicated Endpoint on huggingface.co/inference-endpoints,
# then pass its URL to any huggingfaceR function:
hf_embed(
  "Embed this with my dedicated endpoint",
  model = "my-org/my-custom-model",
  endpoint_url = "https://my-endpoint-id.us-east-1.aws.endpoints.huggingface.cloud"
)

hf_classify(
  "Classify with a private model",
  model = "my-org/my-classifier",
  endpoint_url = "https://my-endpoint-id.us-east-1.aws.endpoints.huggingface.cloud"
)

# Chat and generate also support endpoint_url
hf_chat(
  "Hello from my dedicated endpoint!",
  model = "my-org/my-llm",
  endpoint_url = "https://my-endpoint-id.us-east-1.aws.endpoints.huggingface.cloud"
)
```

The `endpoint_url` parameter is available on all inference functions,
including batch variants
([`hf_embed_batch()`](https://farach.github.io/huggingfaceR/dev/reference/hf_embed_batch.md),
[`hf_classify_batch()`](https://farach.github.io/huggingfaceR/dev/reference/hf_classify_batch.md),
etc.).

## Additional Resources

- [Hugging Face Inference Providers
  documentation](https://huggingface.co/docs/inference-providers)
- [Hugging Face Inference
  Endpoints](https://huggingface.co/docs/inference-endpoints)
- [Hugging Face Model Hub](https://huggingface.co/models)
- [Hugging Face Datasets](https://huggingface.co/datasets)
- [Report issues on
  GitHub](https://github.com/farach/huggingfaceR/issues)
