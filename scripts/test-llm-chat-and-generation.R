# Test Script for llm-chat-and-generation.Rmd Vignette
#
# Executes all code from the vignette to verify correctness.
#
# Usage:
#   Rscript scripts/test-llm-chat-and-generation.R
#
# Prerequisites:
#   - A valid Hugging Face token in .Renviron (HUGGING_FACE_HUB_TOKEN)
#   - Packages: huggingfaceR, dplyr

devtools::load_all()
library(dplyr)

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

cat("=== LLM Chat and Generation Vignette Test ===\n\n")

# ============================================================
# Section: Single-Turn Chat
# ============================================================
cat("Single-Turn Chat\n")

test("hf_chat basic question", {
  result <- hf_chat(
    "What are the main differences between R and Python for data analysis?",
    max_tokens = 50
  )
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 1, "expected 1 row")
  check(all(c("role", "content") %in% names(result)))
  check(result$role[1] == "assistant", "expected assistant role")
  check(nchar(result$content[1]) > 0, "expected non-empty content")
  result
})

test("hf_chat with system prompt", {
  result <- hf_chat(
    "What is p-hacking?",
    system = "You are a statistics professor. Explain concepts precisely.",
    max_tokens = 80
  )
  check(tibble::is_tibble(result), "expected tibble")
  check(nchar(result$content[1]) > 0, "expected non-empty response")
  result
})

test("hf_chat with temperature and max_tokens", {
  # Short, focused answer
  result <- hf_chat(
    "Define overfitting in one sentence.",
    max_tokens = 50,
    temperature = 0.1
  )
  check(tibble::is_tibble(result), "expected tibble")
  check(nchar(result$content[1]) > 0, "expected non-empty response")
  result
})

# ============================================================
# Section: Multi-Turn Conversations
# ============================================================
cat("\nMulti-Turn Conversations\n")

test("hf_conversation creation", {
  convo <<- hf_conversation(
    system = "You are a helpful R programming tutor. Give concise answers."
  )
  check(!is.null(convo), "expected non-NULL conversation")
  check(inherits(convo, "hf_conversation"), "expected hf_conversation class")
  convo
})

test("chat() first turn", {
  convo <<- chat(convo, "How do I read a CSV file in R?")
  check(!is.null(convo), "expected non-NULL after chat")
  check(inherits(convo, "hf_conversation"), "expected hf_conversation class")
  convo
})

test("chat() second turn (context-aware)", {
  convo <<- chat(convo, "What if the file uses semicolons as delimiters?")
  check(!is.null(convo), "expected non-NULL after second chat")
  convo
})

test("chat() third turn", {
  convo <<- chat(convo, "How do I handle missing values during import?")
  check(!is.null(convo), "expected non-NULL after third chat")
  convo
})

test("Print conversation", {
  # Just verify print doesn't error
  capture.output(print(convo))
  TRUE
})

# ============================================================
# Section: Text Generation
# ============================================================
cat("\nText Generation\n")

test("hf_generate basic prompt", {
  result <- hf_generate(
    "The three most important principles of tidy data are",
    max_new_tokens = 50
  )
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 1, "expected 1 row")
  check("generated_text" %in% names(result), "expected generated_text column")
  check(nchar(result$generated_text[1]) > 0, "expected non-empty generation")
  result
})

test("hf_generate with temperature", {
  result <- hf_generate(
    "Once upon a time in a small village nestled in the mountains,",
    max_new_tokens = 50,
    temperature = 0.8
  )
  check(tibble::is_tibble(result), "expected tibble")
  check(nchar(result$generated_text[1]) > 0, "expected non-empty")
  result
})

test("hf_generate with top_p", {
  result <- hf_generate(
    "The best way to learn R programming is",
    top_p = 0.5,
    temperature = 0.7,
    max_new_tokens = 40
  )
  check(tibble::is_tibble(result), "expected tibble")
  check(nchar(result$generated_text[1]) > 0, "expected non-empty")
  result
})

test("hf_generate batch (3 prompts)", {
  prompts <- c(
    "The advantages of functional programming include",
    "Reproducible research requires",
    "The tidyverse philosophy emphasizes"
  )
  result <- hf_generate(prompts, max_new_tokens = 30)
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 3, "expected 3 rows")
  check(all(nchar(result$generated_text) > 0), "all non-empty")
  result
})

# ============================================================
# Section: Fill-in-the-Blank (Fill Mask)
# ============================================================
cat("\nFill Mask\n")

test("hf_fill_mask basic (BERT)", {
  result <- hf_fill_mask("The capital of France is [MASK].")
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) >= 1, "expected at least 1 prediction")
  check(all(c("token", "score", "filled") %in% names(result)))
  # "paris" should be among top predictions
  check(any(grepl("paris", tolower(result$token))),
        "expected 'paris' among predictions")
  result
})

test("hf_fill_mask with top_k=3", {
  result <- hf_fill_mask("R is a [MASK] for statistical computing.", top_k = 3)
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 3, sprintf("expected 3 rows, got %d", nrow(result)))
  result
})

test("hf_fill_mask with RoBERTa (<mask>)", {
  result <- hf_fill_mask(
    "Data science is a <mask> field.",
    model = "FacebookAI/roberta-base",
    mask_token = "<mask>"
  )
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) >= 1, "expected at least 1 prediction")
  check(all(c("token", "score") %in% names(result)))
  result
})

test("hf_fill_mask use cases", {
  # Word associations
  result1 <- hf_fill_mask("In machine learning, the opposite of overfitting is [MASK].")
  check(nrow(result1) >= 1, "expected results for overfitting")

  # Linguistic expectations
  result2 <- hf_fill_mask("After the storm, the sky became [MASK].")
  check(nrow(result2) >= 1, "expected results for sky")
  result2
})

# ============================================================
# Section: Using Different Models
# ============================================================
cat("\nDifferent Models\n")

test("hf_search_models for text-generation", {
  result <- hf_search_models(task = "text-generation", sort = "downloads", limit = 5)
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 5, "expected 5 rows")
  check("model_id" %in% names(result))
  result
})

test("hf_search_models for fill-mask", {
  result <- hf_search_models(task = "fill-mask", sort = "downloads", limit = 5)
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 5, "expected 5 rows")
  result
})

# ============================================================
# Section: Data Frame Integration
# ============================================================
cat("\nData Frame Integration\n")

test("Generate descriptions in pipeline", {
  products <- tibble(
    name = c("Ergonomic Keyboard", "Standing Desk"),
    features = c(
      "split layout, mechanical switches, wrist rest",
      "electric height adjustment, memory presets"
    )
  )

  result <- products |>
    mutate(
      description = purrr::map_chr(paste(name, "-", features), function(prompt) {
        res <- hf_chat(
          paste("Write a one-sentence product description for:", prompt),
          max_tokens = 50,
          temperature = 0.7
        )
        res$content[1]
      })
    )
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 2, "expected 2 rows")
  check("description" %in% names(result), "expected description column")
  check(all(nchar(result$description) > 0), "all descriptions non-empty")
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
