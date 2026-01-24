# Test Script for hub-datasets-and-modeling.Rmd Vignette
#
# Executes all code from the vignette to verify correctness.
#
# Usage:
#   Rscript scripts/test-hub-datasets-and-modeling.R
#
# Prerequisites:
#   - A valid Hugging Face token in .Renviron (HUGGING_FACE_HUB_TOKEN)
#   - Packages: huggingfaceR, dplyr, tidymodels

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

cat("=== Hub, Datasets, and Modeling Vignette Test ===\n\n")

# ============================================================
# Section: Searching Models
# ============================================================
cat("Searching Models\n")

test("hf_search_models by task (text-classification)", {
  result <- hf_search_models(task = "text-classification", limit = 5)
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 5, "expected 5 rows")
  check("model_id" %in% names(result))
  result
})

test("hf_search_models by task (feature-extraction)", {
  result <- hf_search_models(task = "feature-extraction", limit = 5)
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 5, "expected 5 rows")
  result
})

test("hf_search_models by author", {
  result <- hf_search_models(author = "facebook", limit = 5)
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) >= 1, "expected at least 1 model")
  result
})

test("hf_search_models by search term", {
  result <- hf_search_models(search = "sentiment english", limit = 5)
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) >= 1, "expected at least 1 model")
  result
})

test("hf_search_models combined filters", {
  result <- hf_search_models(
    task = "text-classification",
    search = "emotion",
    sort = "likes",
    limit = 5
  )
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) >= 1, "expected at least 1 model")
  result
})

test("hf_search_models sort by likes", {
  result <- hf_search_models(
    task = "fill-mask",
    sort = "likes",
    limit = 5
  )
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) >= 1, "expected at least 1 model")
  result
})

test("Discovery to usage: search then classify", {
  models <- hf_search_models(task = "text-classification", search = "emotion", limit = 1)
  check(nrow(models) >= 1, "expected at least 1 model")

  result <- hf_classify("I'm so happy today!", model = models$model_id[1])
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) >= 1, "expected classification result")
  result
})

# ============================================================
# Section: Model Details
# ============================================================
cat("\nModel Details\n")

test("hf_model_info for BAAI/bge-small-en-v1.5", {
  info <- hf_model_info("BAAI/bge-small-en-v1.5")
  check(!is.null(info), "expected non-NULL info")
  check(!is.null(info$pipeline_tag), "expected pipeline_tag")
  check(!is.null(info$downloads), "expected downloads field")
  info
})

test("hf_list_tasks returns task list", {
  tasks <- hf_list_tasks()
  check(is.character(tasks), "expected character vector")
  check(length(tasks) > 10, "expected many tasks")
  check("text-classification" %in% tasks, "expected text-classification in list")
  tasks
})

test("hf_list_tasks with pattern filter", {
  tasks <- hf_list_tasks(pattern = "classification")
  check(is.character(tasks), "expected character vector")
  check(length(tasks) >= 2, "expected at least 2 classification tasks")
  check(all(grepl("classification", tasks)), "all should match pattern")
  tasks
})

# ============================================================
# Section: Searching Datasets
# ============================================================
cat("\nSearching Datasets\n")

test("hf_search_datasets by search term", {
  result <- hf_search_datasets(search = "sentiment", limit = 5)
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) >= 1, "expected at least 1 dataset")
  check("dataset_id" %in% names(result), "expected dataset_id column")
  result
})

test("hf_search_datasets sorted by likes", {
  result <- hf_search_datasets(search = "translation", sort = "likes", limit = 5)
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) >= 1, "expected at least 1 dataset")
  result
})

# ============================================================
# Section: Loading Datasets
# ============================================================
cat("\nLoading Datasets\n")

test("hf_load_dataset imdb train (limit=20)", {
  imdb <<- hf_load_dataset("imdb", split = "train", limit = 20)
  check(tibble::is_tibble(imdb), "expected tibble")
  check(nrow(imdb) == 20, sprintf("expected 20 rows, got %d", nrow(imdb)))
  check("text" %in% names(imdb), "expected text column")
  check("label" %in% names(imdb), "expected label column")
  imdb
})

test("hf_load_dataset imdb test (limit=10)", {
  result <- hf_load_dataset("imdb", split = "test", limit = 10)
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 10, "expected 10 rows")
  result
})

test("hf_load_dataset with offset (pagination)", {
  batch1 <- hf_load_dataset("imdb", split = "train", limit = 10, offset = 0)
  batch2 <- hf_load_dataset("imdb", split = "train", limit = 10, offset = 10)
  check(nrow(batch1) == 10, "expected 10 rows in batch1")
  check(nrow(batch2) == 10, "expected 10 rows in batch2")
  # Texts should be different between batches
  check(!identical(batch1$text[1], batch2$text[1]),
        "expected different texts across offsets")
  bind_rows(batch1, batch2)
})

test("hf_dataset_info for imdb", {
  info <- hf_dataset_info("imdb")
  check(!is.null(info), "expected non-NULL info")
  check(is.list(info), "expected list")
  info
})

# ============================================================
# Section: Tidymodels Integration (step_hf_embed)
# ============================================================
cat("\nTidymodels Integration\n")

test("step_hf_embed in recipe", {
  library(tidymodels)

  train_data <<- tibble(
    text = c(
      "This movie was fantastic, truly moving",
      "Terrible acting and boring plot",
      "A masterpiece of modern cinema",
      "Waste of time, do not watch",
      "Beautiful story and great performances",
      "Dull and predictable from start to finish"
    ),
    sentiment = factor(c("pos", "neg", "pos", "neg", "pos", "neg"))
  )

  rec <- recipe(sentiment ~ text, data = train_data) |>
    step_hf_embed(text)

  check(inherits(rec, "recipe"), "expected recipe object")
  rec
})

test("prep() and bake() with step_hf_embed", {
  library(tidymodels)

  rec <- recipe(sentiment ~ text, data = train_data) |>
    step_hf_embed(text)

  rec_prepped <<- prep(rec)
  check(inherits(rec_prepped, "recipe"), "expected prepped recipe")

  baked <- bake(rec_prepped, new_data = train_data)
  check(tibble::is_tibble(baked), "expected tibble")
  # Should have 384 embedding cols + 1 outcome = 385
  check(ncol(baked) >= 385,
        sprintf("expected 385+ cols, got %d", ncol(baked)))
  check("sentiment" %in% names(baked), "expected sentiment column")
  # Check embedding column naming
  check("text_emb_1" %in% names(baked), "expected text_emb_1 column")
  baked
})

test("Full workflow: recipe + model + fit", {
  library(tidymodels)

  # Use the 6 training texts from before
  rec <- recipe(sentiment ~ text, data = train_data) |>
    step_hf_embed(text)

  lr_model <- logistic_reg() |>
    set_engine("glm")

  wf <- workflow() |>
    add_recipe(rec) |>
    add_model(lr_model)

  fitted_wf <<- fit(wf, data = train_data)
  check(inherits(fitted_wf, "workflow"), "expected fitted workflow")
  fitted_wf
})

test("Predict with fitted workflow", {
  library(tidymodels)

  test_data <- tibble(
    text = c("Great film, loved it", "Awful movie, boring"),
    sentiment = factor(c("pos", "neg"))
  )

  predictions <- predict(fitted_wf, new_data = test_data) |>
    bind_cols(test_data)
  check(tibble::is_tibble(predictions), "expected tibble")
  check(".pred_class" %in% names(predictions), "expected .pred_class column")
  check(nrow(predictions) == 2, "expected 2 rows")
  predictions
})

test("tidy() on prepped step", {
  library(tidymodels)
  tidy_result <- tidy(rec_prepped, number = 1)
  check(tibble::is_tibble(tidy_result), "expected tibble")
  tidy_result
})

# ============================================================
# Section: End-to-End (IMDB load + embed + classify)
# ============================================================
cat("\nEnd-to-End Pipeline\n")

test("Load IMDB, embed, train, predict", {
  library(tidymodels)

  # Load small subsets - use offset to ensure both classes are present
  imdb_train <- hf_load_dataset("imdb", split = "train", limit = 20) |>
    mutate(sentiment = factor(ifelse(label == 1, "pos", "neg"),
                              levels = c("neg", "pos"))) |>
    select(text, sentiment)

  imdb_test <- hf_load_dataset("imdb", split = "test", limit = 5) |>
    mutate(sentiment = factor(ifelse(label == 1, "pos", "neg"),
                              levels = c("neg", "pos"))) |>
    select(text, sentiment)

  check(nrow(imdb_train) == 20, "expected 20 train rows")
  check(nrow(imdb_test) == 5, "expected 5 test rows")

  # Build workflow
  wf <- workflow() |>
    add_recipe(
      recipe(sentiment ~ text, data = imdb_train) |>
        step_hf_embed(text)
    ) |>
    add_model(logistic_reg())

  fitted <- fit(wf, data = imdb_train)
  check(inherits(fitted, "workflow"), "expected fitted workflow")

  # Predict
  preds <- predict(fitted, imdb_test) |>
    bind_cols(imdb_test)
  check(".pred_class" %in% names(preds), "expected .pred_class")
  check(nrow(preds) == 5, "expected 5 predictions")
  preds
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
