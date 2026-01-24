# Test Script for README.Rmd
#
# Executes all code examples from the README to verify correctness.
#
# Usage:
#   Rscript scripts/test-readme.R
#
# Prerequisites:
#   - A valid Hugging Face token in .Renviron (HUGGING_FACE_HUB_TOKEN)
#   - Packages: huggingfaceR, dplyr, tidyr, tidymodels

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

cat("=== README.Rmd Test ===\n\n")

# ============================================================
# Section: Setup
# ============================================================
cat("Setup\n")

test("hf_whoami returns user info", {
  result <- hf_whoami()
  check(!is.null(result), "expected non-NULL result")
  result
})

# ============================================================
# Section: Text Classification
# ============================================================
cat("\nText Classification\n")

test("hf_classify sentiment", {
  result <- hf_classify("I love using R for data science!")
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) >= 1, "expected at least 1 row")
  check(result$label[1] == "POSITIVE", "expected POSITIVE")
  check(result$score[1] > 0.9, "expected high confidence")
  result
})

test("hf_classify_zero_shot", {
  result <- hf_classify_zero_shot(
    "I just bought a new laptop for coding",
    labels = c("technology", "sports", "politics", "food")
  )
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 4, "expected 4 rows")
  top <- result |> slice_max(score, n = 1)
  check(top$label[1] == "technology",
        sprintf("expected technology, got %s", top$label[1]))
  result
})

# ============================================================
# Section: Embeddings and Similarity
# ============================================================
cat("\nEmbeddings and Similarity\n")

test("hf_embed on 3 sentences", {
  sentences <- c(
    "The cat sat on the mat",
    "A feline rested on the rug",
    "The dog played in the park"
  )
  embeddings <<- hf_embed(sentences)
  check(tibble::is_tibble(embeddings), "expected tibble")
  check(nrow(embeddings) == 3, "expected 3 rows")
  check(embeddings$n_dims[1] == 384, "expected 384 dims")
  embeddings
})

test("hf_similarity", {
  result <- hf_similarity(embeddings)
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 3, "expected 3 pairs")
  check(all(c("text_1", "text_2", "similarity") %in% names(result)))
  # Cat-feline should be most similar
  best <- result |> slice_max(similarity, n = 1)
  check(grepl("cat|feline", best$text_1[1]) && grepl("cat|feline", best$text_2[1]),
        "expected cat-feline as most similar pair")
  result
})

# ============================================================
# Section: Chat with Open-Source LLMs
# ============================================================
cat("\nChat\n")

test("hf_chat basic", {
  result <- hf_chat("What is the tidyverse?", max_tokens = 50)
  check(tibble::is_tibble(result), "expected tibble")
  check(nchar(result$content[1]) > 0, "expected non-empty response")
  result
})

test("hf_chat with system prompt", {
  result <- hf_chat(
    "Explain logistic regression in two sentences.",
    system = "You are a statistics instructor. Use plain language.",
    max_tokens = 80
  )
  check(tibble::is_tibble(result), "expected tibble")
  check(nchar(result$content[1]) > 0, "expected non-empty response")
  result
})

test("Multi-turn conversation", {
  convo <- hf_conversation(system = "You are a helpful R tutor.")
  convo <- chat(convo, "How do I read a CSV file?")
  check(inherits(convo, "hf_conversation"), "expected hf_conversation")
  convo <- chat(convo, "What about Excel files?")
  check(inherits(convo, "hf_conversation"), "expected hf_conversation")
  convo
})

# ============================================================
# Section: Text Generation
# ============================================================
cat("\nText Generation\n")

test("hf_generate", {
  result <- hf_generate(
    "Once upon a time in a land far away,",
    max_new_tokens = 50
  )
  check(tibble::is_tibble(result), "expected tibble")
  check("generated_text" %in% names(result))
  check(nchar(result$generated_text[1]) > 0, "expected non-empty text")
  result
})

test("hf_fill_mask", {
  result <- hf_fill_mask("The capital of France is [MASK].")
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) >= 1, "expected predictions")
  check(any(grepl("paris", tolower(result$token))),
        "expected paris in predictions")
  result
})

# ============================================================
# Section: Tidyverse Integration
# ============================================================
cat("\nTidyverse Integration\n")

test("Classify in mutate pipeline", {
  reviews <- tibble(
    id = 1:3,
    text = c(
      "This product is amazing!",
      "Terrible experience.",
      "It's okay, nothing special."
    )
  )

  result <- reviews |>
    mutate(sentiment = hf_classify(text)) |>
    unnest(sentiment, names_sep = "_") |>
    select(id, text, sentiment_label, sentiment_score)

  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 3, "expected 3 rows")
  check(all(c("id", "text", "sentiment_label", "sentiment_score") %in% names(result)))
  result
})

# ============================================================
# Section: Tidymodels
# ============================================================
cat("\nTidymodels\n")

test("step_hf_embed in recipe", {
  library(tidymodels)

  train_data <- tibble(
    text = c(
      "Great product, love it",
      "Terrible quality, broke immediately",
      "Excellent value for money",
      "Worst purchase ever"
    ),
    sentiment = factor(c("pos", "neg", "pos", "neg"))
  )

  rec <- recipe(sentiment ~ text, data = train_data) |>
    step_hf_embed(text)

  wf <- workflow() |>
    add_recipe(rec) |>
    add_model(logistic_reg()) |>
    fit(data = train_data)

  check(inherits(wf, "workflow"), "expected fitted workflow")
  wf
})

# ============================================================
# Section: Tidytext
# ============================================================
cat("\nTidytext\n")

test("hf_embed_text + hf_nearest_neighbors", {
  docs <- tibble(
    text = c(
      "Machine learning algorithms",
      "Deep neural networks",
      "Cooking pasta recipes",
      "Italian food traditions",
      "Statistical modeling techniques"
    )
  )

  result <- docs |>
    hf_embed_text(text) |>
    hf_nearest_neighbors("machine learning", k = 3)

  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 3, "expected 3 neighbors")
  check("similarity" %in% names(result))
  result
})

test("hf_embed_text + hf_cluster_texts + hf_extract_topics", {
  docs <- tibble(
    text = c(
      "Machine learning algorithms",
      "Deep neural networks",
      "Cooking pasta recipes",
      "Italian food traditions",
      "Statistical modeling techniques",
      "Baking bread at home"
    )
  )

  result <- docs |>
    hf_embed_text(text) |>
    hf_cluster_texts(k = 2)
  check("cluster" %in% names(result), "expected cluster column")
  check(length(unique(result$cluster)) == 2, "expected 2 clusters")

  topics <- docs |>
    hf_embed_text(text) |>
    hf_extract_topics(text_col = "text", k = 2)
  check(tibble::is_tibble(topics), "expected tibble")
  check(nrow(topics) >= 2, "expected at least 2 topic rows")
  topics
})

# ============================================================
# Section: Hub and Datasets
# ============================================================
cat("\nHub and Datasets\n")

test("hf_search_models", {
  result <- hf_search_models(task = "text-classification", limit = 5)
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 5, "expected 5 results")
  result
})

test("hf_load_dataset imdb", {
  imdb <- hf_load_dataset("imdb", split = "train", limit = 10)
  check(tibble::is_tibble(imdb), "expected tibble")
  check(nrow(imdb) == 10, "expected 10 rows")
  check("text" %in% names(imdb))
  imdb
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
