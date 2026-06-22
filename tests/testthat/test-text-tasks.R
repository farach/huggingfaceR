# Unit tests for the API-first text tasks (R/text-tasks.R). The shared engine
# `hf_task_request()` is mocked so these run offline and assert that arguments
# flow into the request and that responses parse into the documented tibbles.

# --- hf_summarize -----------------------------------------------------------

test_that("hf_summarize parses summary_text and forwards length params", {
  captured <- NULL
  testthat::local_mocked_bindings(
    hf_task_request = function(model, inputs, parameters = NULL,
                               token = NULL, endpoint_url = NULL) {
      captured <<- list(model = model, inputs = inputs, parameters = parameters)
      list(list(summary_text = paste0("SUMMARY: ", inputs)))
    }
  )

  res <- hf_summarize("A long article about R.", min_length = 5, max_length = 20)

  expect_s3_class(res, "tbl_df")
  expect_named(res, c("text", "summary"))
  expect_equal(res$text, "A long article about R.")
  expect_equal(res$summary, "SUMMARY: A long article about R.")
  expect_equal(captured$parameters$min_length, 5)
  expect_equal(captured$parameters$max_length, 20)
})

test_that("hf_summarize preserves NA rows and input order", {
  testthat::local_mocked_bindings(
    hf_task_request = function(model, inputs, parameters = NULL,
                               token = NULL, endpoint_url = NULL) {
      list(list(summary_text = toupper(inputs)))
    }
  )

  res <- hf_summarize(c("a", NA, "b"))
  expect_equal(res$text, c("a", NA, "b"))
  expect_equal(res$summary, c("A", NA, "B"))
})

test_that("hf_summarize returns an empty tibble for empty input", {
  res <- hf_summarize(character())
  expect_s3_class(res, "tbl_df")
  expect_equal(nrow(res), 0)
  expect_named(res, c("text", "summary"))
})

# --- hf_translate -----------------------------------------------------------

test_that("hf_translate forwards source/target and parses translation_text", {
  captured <- NULL
  testthat::local_mocked_bindings(
    hf_task_request = function(model, inputs, parameters = NULL,
                               token = NULL, endpoint_url = NULL) {
      captured <<- parameters
      list(list(translation_text = "Bonjour"))
    }
  )

  res <- hf_translate("Hello", source = "eng_Latn", target = "fra_Latn")
  expect_equal(res$translation, "Bonjour")
  expect_equal(captured$src_lang, "eng_Latn")
  expect_equal(captured$tgt_lang, "fra_Latn")
})

test_that("hf_translate leaves language params NULL when unset", {
  captured <- "unset"
  testthat::local_mocked_bindings(
    hf_task_request = function(model, inputs, parameters = NULL,
                               token = NULL, endpoint_url = NULL) {
      captured <<- parameters
      list(list(translation_text = "x"))
    }
  )

  hf_translate("Hello")
  expect_null(captured$src_lang)
  expect_null(captured$tgt_lang)
})

# --- hf_ner -----------------------------------------------------------------

test_that("hf_ner returns one row per entity and forwards aggregation_strategy", {
  captured <- NULL
  testthat::local_mocked_bindings(
    hf_task_request = function(model, inputs, parameters = NULL,
                               token = NULL, endpoint_url = NULL) {
      captured <<- parameters
      list(
        list(entity_group = "PER", word = "Obama", score = 0.99, start = 7L, end = 12L),
        list(entity_group = "LOC", word = "Hawaii", score = 0.98, start = 25L, end = 31L)
      )
    }
  )

  res <- hf_ner("Barack Obama was born in Hawaii.", aggregation_strategy = "first")
  expect_equal(nrow(res), 2)
  expect_named(res, c("text", "word", "entity_group", "score", "start", "end"))
  expect_equal(res$entity_group, c("PER", "LOC"))
  expect_equal(res$word, c("Obama", "Hawaii"))
  expect_equal(res$start, c(7L, 25L))
  expect_equal(captured$aggregation_strategy, "first")
})

test_that("hf_ner yields a single NA row when no entities are found", {
  testthat::local_mocked_bindings(
    hf_task_request = function(model, inputs, parameters = NULL,
                               token = NULL, endpoint_url = NULL) {
      list()
    }
  )
  res <- hf_ner("nothing notable here")
  expect_equal(nrow(res), 1)
  expect_true(is.na(res$word))
  expect_true(is.na(res$entity_group))
})

test_that("hf_ner falls back to the 'entity' field when 'entity_group' is absent", {
  testthat::local_mocked_bindings(
    hf_task_request = function(model, inputs, parameters = NULL,
                               token = NULL, endpoint_url = NULL) {
      list(list(entity = "PER", word = "Ada", score = 0.9, start = 0L, end = 3L))
    }
  )
  res <- hf_ner("Ada", aggregation_strategy = "none")
  expect_equal(res$entity_group, "PER")
})

# --- hf_question_answer -----------------------------------------------------

test_that("hf_question_answer builds the inputs object and parses the answer", {
  captured <- NULL
  testthat::local_mocked_bindings(
    hf_task_request = function(model, inputs, parameters = NULL,
                               token = NULL, endpoint_url = NULL) {
      captured <<- inputs
      list(answer = "Honolulu", score = 0.97, start = 18L, end = 26L)
    }
  )

  res <- hf_question_answer("Where?", "Obama was born in Honolulu.")
  expect_equal(res$answer, "Honolulu")
  expect_equal(res$score, 0.97)
  expect_equal(captured$question, "Where?")
  expect_equal(captured$context, "Obama was born in Honolulu.")
})

test_that("hf_question_answer recycles a single context across questions", {
  testthat::local_mocked_bindings(
    hf_task_request = function(model, inputs, parameters = NULL,
                               token = NULL, endpoint_url = NULL) {
      list(answer = inputs$question, score = 1, start = 0L, end = 1L)
    }
  )
  res <- hf_question_answer(c("Q1", "Q2"), "shared context")
  expect_equal(nrow(res), 2)
  expect_equal(res$question, c("Q1", "Q2"))
})

test_that("hf_question_answer errors on mismatched lengths", {
  expect_error(
    hf_question_answer(c("a", "b"), c("x", "y", "z")),
    "same length"
  )
})

# --- hf_table_question_answer -----------------------------------------------

test_that("hf_table_question_answer converts the data frame to string-cell arrays", {
  captured <- NULL
  testthat::local_mocked_bindings(
    hf_task_request = function(model, inputs, parameters = NULL,
                               token = NULL, endpoint_url = NULL) {
      captured <<- inputs
      list(answer = "120", aggregator = "MAX", cells = list("120"))
    }
  )

  sales <- data.frame(product = c("A", "B"), revenue = c(120, 80))
  res <- hf_table_question_answer("highest revenue?", sales)

  expect_named(res, c("query", "answer", "aggregator", "cells"))
  expect_equal(res$answer, "120")
  expect_equal(res$aggregator, "MAX")
  expect_equal(res$cells[[1]], "120")
  expect_equal(captured$query, "highest revenue?")
  # Each column is a list so it serializes as a JSON array, values are strings.
  expect_equal(captured$table$product, list("A", "B"))
  expect_equal(captured$table$revenue, list("120", "80"))
})

test_that("hf_table_question_answer requires a data frame", {
  expect_error(
    hf_table_question_answer("q", list(a = 1)),
    "data frame"
  )
})
