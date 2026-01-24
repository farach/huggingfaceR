# Test Script for getting-started.Rmd Vignette
#
# Executes all code from the vignette to verify correctness.
#
# Usage:
#   Rscript scripts/test-getting-started.R
#
# Prerequisites:
#   - A valid Hugging Face token in .Renviron (HUGGING_FACE_HUB_TOKEN)
#   - Packages: huggingfaceR, dplyr, tidyr

devtools::load_all()
library(dplyr)
library(tidyr)

# --- Token discovery ---
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
      if (Sys.getenv("HUGGING_FACE_HUB_TOKEN") != "") break
    }
  }
  if (Sys.getenv("HUGGING_FACE_HUB_TOKEN") == "") {
    stop("No HUGGING_FACE_HUB_TOKEN found.")
  }
}

# --- Test infrastructure ---
results <- list()
test <- function(name, expr) {
  cat(sprintf("  %-55s", name))
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

cat("=== Getting Started Vignette Test ===\n\n")

# ============================================================
# Section: Authentication
# ============================================================
cat("Authentication\n")

test("hf_whoami returns user info", {
  result <- hf_whoami()
  check(!is.null(result), "expected non-NULL result")
  result
})

# ============================================================
# Section: Classify Text
# ============================================================
cat("\nClassify Text\n")

test("hf_classify single text", {
  result <- hf_classify("I love using R for data science!")
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) >= 1, "expected at least 1 row")
  check(all(c("text", "label", "score") %in% names(result)))
  check(result$label[1] == "POSITIVE", "expected POSITIVE label")
  check(result$score[1] > 0.9, "expected high confidence")
  result
})

test("hf_classify_zero_shot with custom labels", {
  result <- hf_classify_zero_shot(
    "NASA launches new Mars rover",
    labels = c("science", "politics", "sports", "entertainment")
  )
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 4, "expected 4 rows (one per label)")
  check(all(c("text", "label", "score") %in% names(result)))
  top_label <- result |> slice_max(score, n = 1)
  check(top_label$label[1] == "science",
        sprintf("expected 'science' as top label, got '%s'", top_label$label[1]))
  result
})

# ============================================================
# Section: Generate Embeddings
# ============================================================
cat("\nGenerate Embeddings\n")

test("hf_embed on 3 sentences", {
  sentences <- c(
    "The cat sat on the mat",
    "A feline rested on the rug",
    "The dog played in the park"
  )
  embeddings <<- hf_embed(sentences)
  check(tibble::is_tibble(embeddings), "expected tibble")
  check(nrow(embeddings) == 3, "expected 3 rows")
  check(all(c("text", "embedding", "n_dims") %in% names(embeddings)))
  check(embeddings$n_dims[1] == 384, "expected 384 dimensions")
  embeddings
})

test("hf_similarity on embeddings", {
  result <- hf_similarity(embeddings)
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 3, "expected 3 pairs (3 choose 2)")
  check(all(c("text_1", "text_2", "similarity") %in% names(result)))
  # Cat/feline pair should be most similar
  cat_feline <- result |>
    filter(grepl("cat", text_1) & grepl("feline", text_2) |
           grepl("feline", text_1) & grepl("cat", text_2))
  check(nrow(cat_feline) == 1, "expected cat-feline pair")
  check(cat_feline$similarity[1] > 0.5, "expected high similarity for cat-feline")
  result
})

# ============================================================
# Section: Chat with a Language Model
# ============================================================
cat("\nChat with LLM\n")

test("hf_chat single question", {
  result <- hf_chat("What is the tidyverse?", max_tokens = 50)
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 1, "expected 1 row")
  check("content" %in% names(result), "expected content column")
  check(nchar(result$content[1]) > 0, "expected non-empty response")
  result
})

test("hf_chat with system prompt", {
  result <- hf_chat(
    "Explain logistic regression in two sentences.",
    system = "You are a statistics instructor. Use plain language.",
    max_tokens = 100
  )
  check(tibble::is_tibble(result), "expected tibble")
  check(nchar(result$content[1]) > 0, "expected non-empty response")
  result
})

# ============================================================
# Section: Explore the Hub
# ============================================================
cat("\nExplore the Hub\n")

test("hf_search_models for text-classification", {
  result <- hf_search_models(task = "text-classification", limit = 5)
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 5, "expected 5 rows")
  check("model_id" %in% names(result), "expected model_id column")
  result
})

test("hf_load_dataset imdb", {
  imdb <- hf_load_dataset("imdb", split = "train", limit = 10)
  check(tibble::is_tibble(imdb), "expected tibble")
  check(nrow(imdb) == 10, "expected 10 rows")
  check("text" %in% names(imdb), "expected text column")
  check("label" %in% names(imdb), "expected label column")
  imdb
})

# ============================================================
# Section: Working with Data Frames
# ============================================================
cat("\nData Frame Workflows\n")

test("Classify in mutate pipeline", {
  reviews <- tibble(
    product_id = 1:3,
    review = c(
      "Excellent quality, highly recommend!",
      "Broke after one week of use",
      "Love it! Will buy again"
    )
  )

  result <- reviews |>
    mutate(sentiment = hf_classify(review)) |>
    unnest(sentiment) |>
    select(product_id, review, label, score)

  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 3, "expected 3 rows")
  check(all(c("product_id", "review", "label", "score") %in% names(result)))
  result
})

# ============================================================
# Summary
# ============================================================
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
