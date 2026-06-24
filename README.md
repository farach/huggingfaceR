
<!-- README.md is generated from README.Rmd. Please edit that file -->

# huggingfaceR <a href="https://farach.github.io/huggingfaceR/"><img src="https://raw.githubusercontent.com/farach/huggingfaceR/main/man/figures/logo.svg" align="right" height="139" alt="huggingfaceR logo" /></a>

<!-- badges: start -->
<!-- badges: end -->

An API-first R package for accessing machine learning models,
embeddings, and datasets on the [Hugging Face
Hub](https://huggingface.co) via the Hugging Face Inference API. No
Python required. The Inference API serves a curated subset of the Hub’s
500,000+ models – use `hf_check_inference()` to verify model
availability.

## Installation

``` r
install.packages("huggingfaceR")

# install.packages("devtools")
devtools::install_github("farach/huggingfaceR")
```

## Setup

Get a free API token by following the [Hugging Face access tokens
documentation](https://huggingface.co/docs/hub/security-tokens), then
configure it in R:

``` r
library(huggingfaceR)

hf_set_token("hf_your_token_here", store = TRUE)
hf_whoami()
```

## Text Classification

``` r
# Sentiment analysis
hf_classify("I love using R for data science!")
#> # A tibble: 1 × 3
#>   text                             label    score
#>   <chr>                            <chr>    <dbl>
#> 1 I love using R for data science! POSITIVE 1.000

# Zero-shot classification with custom labels
hf_classify_zero_shot(
  "I just bought a new laptop for coding",
  labels = c("technology", "sports", "politics", "food")
)
#> # A tibble: 4 × 3
#>   text                                  label        score
#>   <chr>                                 <chr>        <dbl>
#> 1 I just bought a new laptop for coding technology 0.985  
#> 2 I just bought a new laptop for coding sports     0.00661
#> 3 I just bought a new laptop for coding food       0.00485
#> 4 I just bought a new laptop for coding politics   0.00391
```

## Embeddings and Similarity

``` r
sentences <- c(
  "The cat sat on the mat",
  "A feline rested on the rug",
  "The dog played in the park"
)

embeddings <- hf_embed(sentences)
embeddings
#> # A tibble: 3 × 3
#>   text                       embedding   n_dims
#>   <chr>                      <list>       <int>
#> 1 The cat sat on the mat     <dbl [384]>    384
#> 2 A feline rested on the rug <dbl [384]>    384
#> 3 The dog played in the park <dbl [384]>    384

hf_similarity(embeddings)
#> # A tibble: 3 × 3
#>   text_1                     text_2                     similarity
#>   <chr>                      <chr>                           <dbl>
#> 1 The cat sat on the mat     A feline rested on the rug      0.748
#> 2 The cat sat on the mat     The dog played in the park      0.516
#> 3 A feline rested on the rug The dog played in the park      0.555
```

## Chat with Open-Source LLMs

``` r
hf_chat("What is the tidyverse?", max_tokens = 60)
#> # A tibble: 1 × 5
#>   role      content                                 model tokens_used tool_calls
#>   <chr>     <chr>                                   <chr>       <int> <list>    
#> 1 assistant "**What is the Tidyverse?**\n\nThe Tid… meta…          60 <list [0]>

# With a system prompt
hf_chat(
  "Explain logistic regression in two sentences.",
  system = "You are a statistics instructor. Use plain language.",
  max_tokens = 80
)
#> # A tibble: 1 × 5
#>   role      content                                 model tokens_used tool_calls
#>   <chr>     <chr>                                   <chr>       <int> <list>    
#> 1 assistant Logistic regression is a type of stati… meta…          80 <list [0]>

# Multi-turn conversation
convo <- hf_conversation(system = "You are a helpful R tutor.")
convo <- chat(convo, "How do I read a CSV file?", max_tokens = 60, temperature = 0)
convo <- chat(
  convo,
  "What about Excel files? Reply in one short sentence.",
  max_tokens = 40,
  temperature = 0
)
convo$history[[length(convo$history)]]$content
#> [1] "You can use the `readxl` package in R to read Excel files, specifically using the `read_excel()` function."
```

## Text Generation

``` r
hf_generate("Once upon a time in a land far away,", max_new_tokens = 40)
#> # A tibble: 1 × 2
#>   prompt                               generated_text                           
#>   <chr>                                <chr>                                    
#> 1 Once upon a time in a land far away, ...there was a beautiful and magical kin…

hf_fill_mask("The capital of France is [MASK].")
#> # A tibble: 5 × 4
#>   text                             token      score filled                      
#>   <chr>                            <chr>      <dbl> <chr>                       
#> 1 The capital of France is [MASK]. paris     0.417  The capital of France is pa…
#> 2 The capital of France is [MASK]. lille     0.0714 The capital of France is li…
#> 3 The capital of France is [MASK]. lyon      0.0634 The capital of France is ly…
#> 4 The capital of France is [MASK]. marseille 0.0444 The capital of France is ma…
#> 5 The capital of France is [MASK]. tours     0.0303 The capital of France is to…
```

## Structured Extraction and Tools

``` r
# Turn unstructured text into typed columns
hf_extract(
  "Amelie is a chef in Paris who mentions burnout.",
  c(name = "string", occupation = "string", city = "string", theme = "string")
)
#> # A tibble: 1 × 4
#>   name   occupation city  theme  
#>   <chr>  <chr>      <chr> <chr>  
#> 1 Amelie chef       Paris burnout

# Tool calling with a tool-capable model
add_tool <- hf_tool("add", "Add two numbers.", c(x = "number", y = "number"))

convo <- hf_conversation(model = "Qwen/Qwen2.5-72B-Instruct")
convo <- chat(
  convo,
  "Use the add tool to add x=2 and y=3, then tell me the answer.",
  tools = list(add_tool),
  tool_choice = "auto"
)
convo <- hf_run_tools(convo, list(add = function(x, y) x + y))
convo$history[[length(convo$history)]]$content
#> [1] "The answer to adding \\( x = 2 \\) and \\( y = 3 \\) is \\( 5 \\)."
```

## Multimodal

``` r
# Audio transcription
audio <- "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"
transcript <- hf_transcribe(audio, return_timestamps = "word")
substr(transcript$text, 1, 120)
#> [1] " I have a dream that one day this nation will rise up and live out the true meaning of its creed."
head(transcript$chunks[[1]], 3)
#> [[1]]
#> [[1]]$text
#> [1] " I"
#> 
#> [[1]]$timestamp
#> [[1]]$timestamp[[1]]
#> [1] 0
#> 
#> [[1]]$timestamp[[2]]
#> [1] 1.1
#> 
#> 
#> 
#> [[2]]
#> [[2]]$text
#> [1] " have"
#> 
#> [[2]]$timestamp
#> [[2]]$timestamp[[1]]
#> [1] 1.1
#> 
#> [[2]]$timestamp[[2]]
#> [1] 1.44
#> 
#> 
#> 
#> [[3]]
#> [[3]]$text
#> [1] " a"
#> 
#> [[3]]$timestamp
#> [[3]]$timestamp[[1]]
#> [1] 1.44
#> 
#> [[3]]$timestamp[[2]]
#> [1] 1.62

# Image generation writes a file and returns raw bytes
img <- hf_text_to_image(
  "a small red cube on a white background",
  output = "man/figures/README-red-cube.jpg",
  seed = 1,
  num_inference_steps = 2,
  guidance_scale = 0,
  overwrite = TRUE
)
tibble::tibble(
  content_type = img$content_type,
  bytes = length(img$image[[1]]),
  file_exists = file.exists(img$path)
)
#> # A tibble: 1 × 3
#>   content_type bytes file_exists
#>   <chr>        <int> <lgl>      
#> 1 image/jpeg   26383 TRUE
knitr::include_graphics("man/figures/README-red-cube.jpg")
```

<img src="man/figures/README-red-cube.jpg" alt="A generated image of a small red cube on a white background." width="100%" />

``` r

# Image understanding
image <- "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"
hf_classify_image(image, top_k = 3)
#> # A tibble: 3 × 3
#>   image                                                              label score
#>   <chr>                                                              <chr> <dbl>
#> 1 https://huggingface.co/datasets/huggingface/documentation-images/… tabb… 0.277
#> 2 https://huggingface.co/datasets/huggingface/documentation-images/… tige… 0.276
#> 3 https://huggingface.co/datasets/huggingface/documentation-images/… Egyp… 0.140
hf_detect_objects(image, threshold = 0.5) |>
  dplyr::filter(label == "cat")
#> # A tibble: 2 × 7
#>   image                                      label score  xmin  ymin  xmax  ymax
#>   <chr>                                      <chr> <dbl> <dbl> <dbl> <dbl> <dbl>
#> 1 https://huggingface.co/datasets/huggingfa… cat   0.997   156    31   385   146
#> 2 https://huggingface.co/datasets/huggingfa… cat   0.999   145   132   429   341
```

## Tidyverse Integration

All functions accept character vectors and return tibbles.

``` r
library(dplyr)
library(tidyr)

reviews <- tibble(
  id = 1:3,
  text = c(
    "This product is amazing!",
    "Terrible experience.",
    "It's okay, nothing special."
  )
)

reviews |>
  mutate(sentiment = hf_classify(text)) |>
  unnest(sentiment, names_sep = "_") |>
  select(id, text, sentiment_label, sentiment_score)
#> # A tibble: 3 × 4
#>      id text                        sentiment_label sentiment_score
#>   <int> <chr>                       <chr>                     <dbl>
#> 1     1 This product is amazing!    POSITIVE                  1.000
#> 2     2 Terrible experience.        NEGATIVE                  1.000
#> 3     3 It's okay, nothing special. NEGATIVE                  0.819
```

### Tidymodels

Use embeddings as features in machine learning workflows:

``` r
library(tidymodels)

rec <- recipe(sentiment ~ text, data = train_data) |>
  step_hf_embed(text)

wf <- workflow() |>
  add_recipe(rec) |>
  add_model(logistic_reg()) |>
  fit(data = train_data)
```

### Tidytext

Semantic search and document clustering:

``` r
docs |>
  hf_embed_text(text) |>
  hf_nearest_neighbors("machine learning", k = 5)

docs |>
  hf_embed_text(text) |>
  hf_cluster_texts(k = 3) |>
  hf_extract_topics(text_col = "text", k = 3)
```

## Hub and Datasets

``` r
# Search models
hf_search_models(task = "text-classification", limit = 5)
#> # A tibble: 5 × 7
#>   model_id                            author task  downloads likes tags  library
#>   <chr>                               <chr>  <chr>     <int> <int> <lis> <chr>  
#> 1 BAAI/bge-reranker-v2-m3             <NA>   text…  16443234  1053 <chr> senten…
#> 2 ProsusAI/finbert                    <NA>   text…   7648889  1184 <chr> transf…
#> 3 BAAI/bge-reranker-base              <NA>   text…   4167279   238 <chr> senten…
#> 4 cardiffnlp/twitter-roberta-base-se… <NA>   text…   3953164   813 <chr> transf…
#> 5 distilbert/distilbert-base-uncased… <NA>   text…   3644729   910 <chr> transf…

# Load datasets into tibbles (no Python needed)
imdb <- hf_load_dataset("imdb", split = "train", limit = 5)
imdb |>
  dplyr::mutate(text = substr(text, 1, 80))
#> # A tibble: 5 × 4
#>   text                                                     label .dataset .split
#>   <chr>                                                    <int> <chr>    <chr> 
#> 1 "I rented I AM CURIOUS-YELLOW from my video store becau…     0 stanfor… train 
#> 2 "\"I Am Curious: Yellow\" is a risible and pretentious …     0 stanfor… train 
#> 3 "If only to avoid making this type of film in the futur…     0 stanfor… train 
#> 4 "This film was probably inspired by Godard's Masculin, …     0 stanfor… train 
#> 5 "Oh, brother...after hearing about this ridiculous film…     0 stanfor… train

# Hugging Face split slices work too
imdb_sample <- hf_load_dataset("imdb", split = "train[:10%]", limit = 5)
imdb_sample |>
  dplyr::mutate(text = substr(text, 1, 80))
#> # A tibble: 5 × 4
#>   text                                                     label .dataset .split
#>   <chr>                                                    <int> <chr>    <chr> 
#> 1 "I rented I AM CURIOUS-YELLOW from my video store becau…     0 stanfor… train…
#> 2 "\"I Am Curious: Yellow\" is a risible and pretentious …     0 stanfor… train…
#> 3 "If only to avoid making this type of film in the futur…     0 stanfor… train…
#> 4 "This film was probably inspired by Godard's Masculin, …     0 stanfor… train…
#> 5 "Oh, brother...after hearing about this ridiculous film…     0 stanfor… train…

# Download files and inspect provider options
hf_list_repo_files("BAAI/bge-small-en-v1.5", recursive = FALSE)
#> # A tibble: 14 × 4
#>    path                              type           size oid                    
#>    <chr>                             <chr>         <int> <chr>                  
#>  1 1_Pooling                         directory         0 b9d00290ba7577bd1709db…
#>  2 onnx                              directory         0 8e8db8abdebe9ba4820469…
#>  3 .gitattributes                    file           1519 a6344aac8c09253b3b630f…
#>  4 README.md                         file          94783 8b8567d75ffa619486d959…
#>  5 config.json                       file            743 3992bf890728a92476c700…
#>  6 config_sentence_transformers.json file            124 dcb0c0d97d09b930d13600…
#>  7 model.safetensors                 file      133466304 a4fef68c99b10468206a9b…
#>  8 modules.json                      file            349 952a9b81c0bfd99800fabf…
#>  9 pytorch_model.bin                 file      133508397 b6e2a796fcfd4513c609fc…
#> 10 sentence_bert_config.json         file             52 ea85692bff64b0d1917833…
#> 11 special_tokens_map.json           file            125 a8b3208c2884c4efb86e49…
#> 12 tokenizer.json                    file         711396 688882a79f44442ddc1f60…
#> 13 tokenizer_config.json             file            366 37fca74771bc76a8e01178…
#> 14 vocab.txt                         file         231508 fb140275c155a9c7c5a3b3…
readme <- hf_hub_download("BAAI/bge-small-en-v1.5", "README.md")
readLines(readme, n = 3)
#> [1] "---"                     "tags:"                  
#> [3] "- sentence-transformers"
hf_list_providers("Qwen/Qwen2.5-72B-Instruct")
#> # A tibble: 2 × 12
#>   model_id       provider status context_length input_price output_price is_free
#>   <chr>          <chr>    <chr>           <int>       <dbl>        <dbl> <lgl>  
#> 1 Qwen/Qwen2.5-… novita   live            32000        0.38          0.4 FALSE  
#> 2 Qwen/Qwen2.5-… feather… live               NA       NA            NA   FALSE  
#> # ℹ 5 more variables: supports_tools <lgl>, supports_structured_output <lgl>,
#> #   first_token_latency_ms <dbl>, throughput <dbl>, is_model_author <lgl>
```

``` r
# Guarded dataset publishing requires a write-scoped token and confirm = TRUE
hf_push_dataset(mtcars, "your-username/mtcars-small", confirm = TRUE)
```

## Learn More

- `vignette("getting-started")` – setup and first examples
- `vignette("text-classification")` – sentiment analysis and zero-shot
  labeling
- `vignette("embeddings-and-similarity")` – semantic search, clustering,
  visualization
- `vignette("llm-chat-and-generation")` – conversations and text
  generation
- `vignette("structured-extraction-and-tools")` – typed extraction,
  streaming, and tool calls
- `vignette("multimodal-images-audio-speech")` – transcription, image
  generation, and vision
- `vignette("hub-datasets-and-modeling")` – Hub discovery and tidymodels
  pipelines
- `vignette("hub-download-upload-share")` – downloads, providers, and
  guarded Hub writes
- `vignette("anthropic-economic-index")` – AI productivity research with
  the Anthropic Economic Index

## License

MIT
