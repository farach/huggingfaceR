# Integration Test Script for huggingfaceR v2
#
# This script exercises every v2 API function with real API calls.
# Run manually to verify the package works end-to-end.
#
# Prerequisites:
#   - A valid Hugging Face token in your .Renviron (HUGGING_FACE_HUB_TOKEN)
#   - Package dependencies installed
#
# Usage:
#   Rscript scripts/test-integration.R
#   -- or from R console --
#   source("scripts/test-integration.R")

devtools::load_all()
library(dplyr)

# Ensure the token is loaded from .Renviron
# R may not auto-load .Renviron when run via Rscript from non-standard shells
if (Sys.getenv("HUGGING_FACE_HUB_TOKEN") == "") {
  renviron_paths <- c(
    file.path(Sys.getenv("HOME"), ".Renviron"),
    file.path(Sys.getenv("USERPROFILE"), "Documents", ".Renviron"),
    file.path(Sys.getenv("USERPROFILE"),
              "OneDrive - Microsoft", "Documents", ".Renviron"),
    file.path(path.expand("~"), ".Renviron")
  )
  for (p in renviron_paths) {
    if (file.exists(p)) {
      readRenviron(p)
      if (Sys.getenv("HUGGING_FACE_HUB_TOKEN") != "") {
        cat(sprintf("Loaded token from: %s\n", p))
        break
      }
    }
  }
  if (Sys.getenv("HUGGING_FACE_HUB_TOKEN") == "") {
    stop("No HUGGING_FACE_HUB_TOKEN found. Set it in .Renviron or run hf_set_token() first.")
  }
}

# --- Test infrastructure ---
results <- list()

test <- function(name, expr) {
  cat(sprintf("  %-50s", name))
  result <- tryCatch(
    {
      val <- eval(expr)
      cat("[PASS]\n")
      results[[name]] <<- list(status = "PASS", value = val)
      val
    },
    error = function(e) {
      cat(sprintf("[FAIL] %s\n", conditionMessage(e)))
      results[[name]] <<- list(status = "FAIL", error = conditionMessage(e))
      NULL
    }
  )
  invisible(result)
}

check <- function(condition, msg = "assertion failed") {

  if (!isTRUE(condition)) stop(msg, call. = FALSE)
}

cat("=== huggingfaceR v2 Integration Tests ===\n\n")

# --- 1. Authentication ---
cat("Authentication\n")

test("hf_whoami returns a list", {
  info <- hf_whoami()
  check(is.list(info), "expected list")
  check(!is.null(info$name), "expected name field")
  info
})

# --- 2. Text Classification ---
cat("\nText Classification\n")

test("hf_classify single text", {
  result <- hf_classify("I love R programming!")
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 1, "expected 1 row")
  check(all(c("text", "label", "score") %in% names(result)))
  check(result$score[1] > 0.5, "expected high confidence")
  result
})

test("hf_classify multiple texts", {
  result <- hf_classify(c("Great product!", "Terrible service."))
  check(nrow(result) == 2, "expected 2 rows")
  result
})

test("hf_classify_zero_shot basic", {
  result <- hf_classify_zero_shot(
    "NASA launches new Mars rover",
    labels = c("science", "politics", "sports")
  )
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 3, "expected 3 rows (one per label)")
  check(all(c("text", "label", "score") %in% names(result)))
  result
})

test("hf_classify_zero_shot multi_label", {
  result <- hf_classify_zero_shot(
    "This laptop is great for gaming and programming",
    labels = c("technology", "gaming", "business"),
    multi_label = TRUE
  )
  check(nrow(result) == 3, "expected 3 rows")
  result
})

# --- 3. Embeddings ---
cat("\nEmbeddings\n")

test("hf_embed basic", {
  result <- hf_embed(c("Hello world", "Goodbye world"))
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 2, "expected 2 rows")
  check(all(c("text", "embedding", "n_dims") %in% names(result)))
  check(is.list(result$embedding), "expected list column")
  check(result$n_dims[1] == 384, "expected 384 dimensions")
  check(length(result$embedding[[1]]) == 384, "expected 384-element vector")
  result
})

test("hf_embed handles NA", {
  result <- hf_embed(c("Hello", NA, "World"))
  check(nrow(result) == 3, "expected 3 rows")
  check(is.null(result$embedding[[2]]), "expected NULL for NA input")
  check(is.na(result$n_dims[2]), "expected NA n_dims for NA input")
  result
})

test("hf_similarity pairwise", {
  emb <- hf_embed(c("cat", "kitten", "car"))
  result <- hf_similarity(emb)
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 3, "expected 3 pairs")
  check(all(c("text_1", "text_2", "similarity") %in% names(result)))
  # cat and kitten should be more similar than cat and car
  cat_kitten <- result$similarity[result$text_1 == "cat" & result$text_2 == "kitten"]
  cat_car <- result$similarity[result$text_1 == "cat" & result$text_2 == "car"]
  check(cat_kitten > cat_car, "expected cat-kitten > cat-car similarity")
  result
})

# --- 4. Tidytext Integration ---
cat("\nTidytext Integration\n")

test("hf_embed_text adds embedding column", {
  docs <- tibble::tibble(
    id = 1:3,
    text = c("Machine learning", "Deep learning", "Cooking recipes")
  )
  result <- hf_embed_text(docs, text)
  check(tibble::is_tibble(result), "expected tibble")
  check("embedding" %in% names(result), "expected embedding column")
  check("n_dims" %in% names(result), "expected n_dims column")
  check(nrow(result) == 3, "expected 3 rows")
  result
})

test("hf_nearest_neighbors", {
  docs <- tibble::tibble(
    id = 1:4,
    text = c("Neural networks", "Linear regression", "Pasta recipe", "Deep learning")
  )
  embedded <- hf_embed_text(docs, text)
  result <- hf_nearest_neighbors(embedded, "artificial intelligence", k = 2)
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 2, "expected 2 neighbors")
  check("similarity" %in% names(result), "expected similarity column")
  result
})

test("hf_cluster_texts", {
  docs <- tibble::tibble(
    text = c("cat", "dog", "kitten", "puppy", "car", "truck")
  )
  embedded <- hf_embed_text(docs, text)
  result <- hf_cluster_texts(embedded, k = 2)
  check("cluster" %in% names(result), "expected cluster column")
  check(length(unique(result$cluster)) == 2, "expected 2 clusters")
  result
})

# --- 5. Chat and Conversations ---
cat("\nChat and Conversations\n")

test("hf_chat basic", {

  result <- hf_chat("What is 2 + 2? Reply with just the number.")
  check(tibble::is_tibble(result), "expected tibble")
  check(all(c("role", "content", "model", "tokens_used") %in% names(result)))
  check(nchar(result$content[1]) > 0, "expected non-empty response")
  result
})

test("hf_chat with system prompt", {
  result <- hf_chat(
    "What is R?",
    system = "Reply in exactly one sentence.",
    max_tokens = 100
  )
  check(nchar(result$content[1]) > 0, "expected non-empty response")
  result
})

test("hf_conversation multi-turn", {
  convo <- hf_conversation(system = "You are a helpful assistant.")
  convo <- chat(convo, "My name is Alex.")
  check(length(convo$history) == 2, "expected 2 messages after first turn")
  convo <- chat(convo, "What is my name?")
  check(length(convo$history) == 4, "expected 4 messages after second turn")
  convo
})

test("hf_chat streaming", {
  deltas <- character()
  result <- hf_chat(
    "Reply with exactly: OK",
    max_tokens = 8,
    temperature = 0,
    stream = TRUE,
    callback = function(delta) deltas <<- c(deltas, delta)
  )
  check(tibble::is_tibble(result), "expected tibble")
  check(nchar(result$content[1]) > 0, "expected non-empty streamed response")
  check(nchar(paste0(deltas, collapse = "")) > 0, "expected streamed deltas")
  result
})

test("hf_chat tool call and hf_run_tools", {
  tool <- hf_tool("add", "Add two numbers.", c(x = "number", y = "number"))
  convo <- hf_conversation(model = "Qwen/Qwen2.5-72B-Instruct")
  convo <- chat(
    convo,
    "Use the add tool to add x=2 and y=3, then tell me the answer.",
    tools = list(tool),
    tool_choice = "auto",
    max_tokens = 120,
    temperature = 0
  )
  check(length(convo$history[[2]]$tool_calls) > 0, "expected a tool call")
  convo <- hf_run_tools(
    convo,
    list(add = function(x, y) x + y),
    max_tokens = 120,
    temperature = 0
  )
  check(length(convo$history) >= 4, "expected tool and final messages")
  check(grepl("5", convo$history[[length(convo$history)]]$content),
        "expected final answer to mention 5")
  convo
})

test("hf_describe_image", {
  result <- hf_describe_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cat.png",
    max_tokens = 50,
    temperature = 0
  )
  check(tibble::is_tibble(result), "expected tibble")
  check(all(c("image", "description") %in% names(result)))
  check(nchar(result$description[1]) > 0, "expected non-empty description")
  result
})

# --- 6. Text Generation ---
cat("\nText Generation\n")

test("hf_generate basic", {
  result <- hf_generate("The tidyverse is", max_new_tokens = 30)
  check(tibble::is_tibble(result), "expected tibble")
  check(all(c("prompt", "generated_text") %in% names(result)))
  check(nchar(result$generated_text[1]) > 0, "expected non-empty generation")
  result
})

test("hf_fill_mask basic", {
  result <- hf_fill_mask("The capital of France is [MASK].", top_k = 3)
  check(tibble::is_tibble(result), "expected tibble")
  check(all(c("text", "token", "score", "filled") %in% names(result)))
  check(nrow(result) == 3, "expected 3 predictions")
  check(any(grepl("paris", result$token, ignore.case = TRUE)),
        "expected 'paris' in predictions")
  result
})

test("hf_fill_mask multiple texts", {
  result <- hf_fill_mask(
    c("R is a [MASK] language.", "Python is [MASK]."),
    top_k = 2
  )
  check(nrow(result) == 4, "expected 4 rows (2 texts x 2 predictions)")
  result
})

# --- 7. Hub Discovery ---
cat("\nHub Discovery\n")

test("hf_search_models by task", {
  result <- hf_search_models(task = "text-classification", limit = 5)
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 5, "expected 5 models")
  check(all(c("model_id", "author", "task", "downloads") %in% names(result)))
  result
})

test("hf_search_models by search term", {
  result <- hf_search_models(search = "sentiment", limit = 3)
  check(nrow(result) == 3, "expected 3 results")
  result
})

test("hf_model_info", {
  result <- hf_model_info("BAAI/bge-small-en-v1.5")
  check(is.list(result), "expected list")
  check(!is.null(result$id), "expected id field")
  result
})

test("hf_list_tasks", {
  result <- hf_list_tasks()
  check(is.character(result), "expected character vector")
  check(length(result) > 10, "expected many tasks")
  check("text-classification" %in% result)
  result
})

test("hf_list_tasks with pattern", {
  result <- hf_list_tasks(pattern = "classification")
  check(all(grepl("classification", result)))
  result
})

test("hf_search_datasets", {
  result <- hf_search_datasets(search = "sentiment", limit = 3)
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 3, "expected 3 datasets")
  check("dataset_id" %in% names(result))
  result
})

# --- 8. Datasets ---
cat("\nDatasets\n")

test("hf_load_dataset with full name", {
  result <- hf_load_dataset("stanfordnlp/imdb", split = "train", limit = 5)
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 5, "expected 5 rows")
  check(all(c("text", "label", ".dataset", ".split") %in% names(result)))
  result
})

test("hf_load_dataset with short name resolution", {
  result <- hf_load_dataset("imdb", split = "test", limit = 3)
  check(nrow(result) == 3, "expected 3 rows")
  result
})

test("hf_load_dataset with explicit config", {
  result <- hf_load_dataset("stanfordnlp/imdb", split = "train",
                            config = "plain_text", limit = 2)
  check(nrow(result) == 2, "expected 2 rows")
  result
})

test("hf_dataset_info", {
  result <- hf_dataset_info("imdb")
  check(is.list(result), "expected list")
  result
})

# --- 9. Text Tasks ---
cat("\nText Tasks\n")

test("hf_summarize shortens text", {
  long_text <- paste(
    "The R programming language is widely used for statistical computing and",
    "graphics. It was created by Ross Ihaka and Robert Gentleman at the",
    "University of Auckland, New Zealand, and is currently developed by the R",
    "Core Team. R provides a wide variety of statistical and graphical",
    "techniques and is highly extensible through packages."
  )
  result <- hf_summarize(long_text, max_length = 40)
  check(tibble::is_tibble(result), "expected tibble")
  check(all(c("text", "summary") %in% names(result)))
  check(nchar(result$summary[1]) > 0, "expected non-empty summary")
  check(nchar(result$summary[1]) < nchar(long_text), "summary should be shorter")
  result
})

test("hf_translate produces non-empty output", {
  result <- hf_translate(
    "Hello, how are you?",
    source = "eng_Latn", target = "fra_Latn"
  )
  check(tibble::is_tibble(result), "expected tibble")
  check(all(c("text", "translation") %in% names(result)))
  check(nchar(result$translation[1]) > 0, "expected non-empty translation")
  result
})

test("hf_ner extracts a person and a location", {
  result <- hf_ner("Barack Obama was born in Hawaii.")
  check(tibble::is_tibble(result), "expected tibble")
  check(all(c("text", "word", "entity_group", "score", "start", "end") %in% names(result)))
  check(any(grepl("Obama", result$word, ignore.case = TRUE)),
        "expected 'Obama' among detected entities")
  check("PER" %in% result$entity_group, "expected a PER entity")
  check("LOC" %in% result$entity_group, "expected a LOC entity")
  result
})

test("hf_question_answer answers from context", {
  result <- hf_question_answer(
    question = "Where was Obama born?",
    context = "Barack Obama was born in Honolulu, Hawaii."
  )
  check(tibble::is_tibble(result), "expected tibble")
  check(all(c("question", "answer", "score", "start", "end") %in% names(result)))
  check(grepl("Honolulu|Hawaii", result$answer[1]),
        "expected answer to mention Honolulu or Hawaii")
  result
})

test("hf_table_question_answer queries a data frame", {
  sales <- data.frame(
    product = c("Widgets", "Gadgets", "Gizmos"),
    revenue = c("120", "80", "50")
  )
  result <- hf_table_question_answer(
    "Which product had the highest revenue?", sales
  )
  check(tibble::is_tibble(result), "expected tibble")
  check(all(c("query", "answer", "aggregator", "cells") %in% names(result)))
  check(nchar(result$answer[1]) > 0, "expected a non-empty answer")
  result
})

test("hf_extract returns structured fields", {
  result <- hf_extract(
    "Amelie is a chef in Paris.",
    c(name = "string", occupation = "string", city = "string"),
    max_tokens = 80
  )
  check(tibble::is_tibble(result), "expected tibble")
  check(all(c("name", "occupation", "city") %in% names(result)))
  check(result$name[1] == "Amelie", "expected name to be Amelie")
  check(result$city[1] == "Paris", "expected city to be Paris")
  result
})

# --- Summary ---
cat("\n=== Results ===\n\n")

pass <- sum(sapply(results, function(r) r$status == "PASS"))
fail <- sum(sapply(results, function(r) r$status == "FAIL"))
total <- length(results)

cat(sprintf("  PASS: %d / %d\n", pass, total))
cat(sprintf("  FAIL: %d / %d\n", fail, total))

if (fail > 0) {
  cat("\nFailed tests:\n")
  for (name in names(results)) {
    if (results[[name]]$status == "FAIL") {
      cat(sprintf("  - %s: %s\n", name, results[[name]]$error))
    }
  }
  cat("\n")
  quit(status = 1)
} else {
  cat("\nAll tests passed.\n")
}
