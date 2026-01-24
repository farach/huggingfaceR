# Test Script for text-classification.Rmd Vignette
#
# Executes all code from the vignette to verify correctness.
#
# Usage:
#   Rscript scripts/test-text-classification.R
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

cat("=== Text Classification Vignette Test ===\n\n")

# ============================================================
# Section: Sentiment Analysis with hf_classify()
# ============================================================
cat("Sentiment Analysis\n")

test("hf_classify single text", {
  result <- hf_classify("I love using R for data science!")
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 1, "expected 1 row")
  check(all(c("text", "label", "score") %in% names(result)))
  check(result$label[1] == "POSITIVE", "expected POSITIVE")
  check(result$score[1] > 0.9, "expected score > 0.9")
  result
})

test("hf_classify batch (5 texts)", {
  reviews <- c(
    "This product exceeded my expectations",
    "Terrible customer service, never again",
    "It works fine, nothing remarkable",
    "Absolutely brilliant design",
    "Waste of money"
  )
  result <- hf_classify(reviews)
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 5, "expected 5 rows")
  check(all(c("text", "label", "score") %in% names(result)))
  result
})

test("hf_classify with alternative model (emotion)", {
  result <- hf_classify(
    "I can't believe we won the championship!",
    model = "j-hartmann/emotion-english-distilroberta-base"
  )
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) >= 1, "expected at least 1 row")
  check("label" %in% names(result))
  result
})

# ============================================================
# Section: Zero-Shot Classification
# ============================================================
cat("\nZero-Shot Classification\n")

test("hf_classify_zero_shot custom categories", {
  result <- hf_classify_zero_shot(
    "The Federal Reserve raised interest rates by 25 basis points",
    labels = c("economics", "politics", "technology", "sports")
  )
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 4, "expected 4 rows (one per label)")
  check(all(c("text", "label", "score") %in% names(result)))
  top <- result |> slice_max(score, n = 1)
  check(top$label[1] == "economics",
        sprintf("expected 'economics' top, got '%s'", top$label[1]))
  result
})

test("hf_classify_zero_shot multi-label", {
  result <- hf_classify_zero_shot(
    "This laptop has amazing graphics and runs all my games smoothly",
    labels = c("technology", "gaming", "business", "entertainment"),
    multi_label = TRUE
  )
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 4, "expected 4 rows")
  # In multi-label mode, scores are independent (can all be high)
  check(all(result$score >= 0 & result$score <= 1), "scores in [0,1]")
  result
})

test("hf_classify_zero_shot batch (3 texts)", {
  headlines <- c(
    "Stock markets reach all-time highs",
    "New vaccine shows 95% efficacy in trials",
    "Championship finals draw record viewership"
  )
  result <- hf_classify_zero_shot(
    headlines,
    labels = c("finance", "health", "sports", "politics")
  )
  check(tibble::is_tibble(result), "expected tibble")
  # 3 texts x 4 labels = 12 rows
  check(nrow(result) == 12,
        sprintf("expected 12 rows, got %d", nrow(result)))
  result
})

# ============================================================
# Section: Data Frame Workflows
# ============================================================
cat("\nData Frame Workflows\n")

test("Sentiment in mutate + unnest pipeline", {
  customer_reviews <- tibble(
    review_id = 1:4,
    product = c("Widget A", "Widget A", "Widget B", "Widget B"),
    text = c(
      "Works perfectly, great build quality",
      "Stopped working after a month",
      "Good value for the price",
      "Flimsy materials, disappointed"
    )
  )

  result <- customer_reviews |>
    mutate(sentiment = hf_classify(text)) |>
    unnest(sentiment, names_sep = "_") |>
    select(review_id, product, text, sentiment_label, sentiment_score)

  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 4, "expected 4 rows")
  check(all(c("sentiment_label", "sentiment_score") %in% names(result)))
  result
})

test("Zero-shot categorization of support tickets", {
  tickets <- tibble(
    ticket_id = 101:104,
    message = c(
      "I can't log into my account",
      "Please cancel my subscription",
      "The app crashes when I open settings",
      "How do I update my payment method?"
    )
  )

  # Classify all messages against the label set
  category_results <- hf_classify_zero_shot(
    tickets$message,
    labels = c("account access", "billing", "bug report",
               "cancellation", "feedback")
  )
  check(nrow(category_results) == 20, "expected 4 texts x 5 labels = 20 rows")

  # Keep the top category for each ticket
  categorized <<- category_results |>
    group_by(text) |>
    slice_max(score, n = 1) |>
    ungroup() |>
    left_join(tickets, by = c("text" = "message")) |>
    select(ticket_id, message = text, category = label, confidence = score)

  check(tibble::is_tibble(categorized), "expected tibble")
  check(nrow(categorized) == 4, "expected 4 rows (one per ticket)")
  check(all(c("category", "confidence") %in% names(categorized)))
  categorized
})

test("Summarize by category", {
  # Uses `categorized` from previous test
  check(!is.null(categorized), "categorized should exist from prior test")

  count_result <- categorized |> count(category, sort = TRUE)
  check(nrow(count_result) > 0, "expected category counts")

  summary_result <- categorized |>
    group_by(category) |>
    summarise(
      n = n(),
      avg_confidence = mean(confidence)
    )
  check(nrow(summary_result) > 0, "expected summary rows")
  check(all(summary_result$avg_confidence > 0), "expected positive confidence")
  summary_result
})

# ============================================================
# Section: Finding Models
# ============================================================
cat("\nFinding Models\n")

test("hf_search_models for text-classification", {
  result <- hf_search_models(task = "text-classification", limit = 5)
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 5, "expected 5 rows")
  check("model_id" %in% names(result))
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
