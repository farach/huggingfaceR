# All tests in this file require Python/transformers
# Skip helper function
skip_if_no_python <- function() {
  python_available <- tryCatch({
    reticulate::py_available(initialize = FALSE) &&
      !is.null(tryCatch(reticulate::import("transformers", delay_load = TRUE), error = function(e) NULL))
  }, error = function(e) FALSE)

 if (!python_available) {
    testthat::skip("Python/transformers not available")
  }
}

test_that("load model fails if not given correct model name", {
  skip_on_cran()
  skip_if_no_python()

  expect_error(hf_load_pipeline(model_id = "x not a model x"))
})

# Tokenizer tests
test_that("tokenizer loads and works correctly", {
  skip_on_cran()
  skip_if_no_python()

  tokenizer <- hf_load_tokenizer("distilbert-base-uncased")

  # incorrect model not loaded
  expect_false(stringr::str_detect(as.character(tokenizer), "GPT|gpt"))
  # tokenizer loads correct model
  expect_true(stringr::str_detect(as.character(tokenizer), "distilbert"))
  # load tokenizer function loads correctly
  expect_true("vocab" %in% names(tokenizer))
  # tokenizer loads fast by default
  expect_true(stringr::str_detect(as.character(tokenizer), "is_fast=True"))
})

test_that("tokenizer use_fast parameter works", {
  skip_on_cran()
  skip_if_no_python()

  slow_tokenizer <- hf_load_tokenizer('distilbert-base-uncased', use_fast = FALSE)
  expect_false(stringr::str_detect(as.character(slow_tokenizer), "is_fast=True"))
})

# AutoModel tests
test_that("Question Answering AutoModel loads correctly", {
  skip_on_cran()
  skip_if_no_python()

  qa_model <- hf_load_AutoModel_for_task(
    model_type = "AutoModelForQuestionAnswering",
    model_id = "deepset/roberta-base-squad2"
  )
  expect_true(stringr::str_detect(as.character(qa_model$config), "QuestionAnswering"))
})

test_that("Sentiment classification AutoModel loads correctly", {
  skip_on_cran()
  skip_if_no_python()

  sent_model <- hf_load_AutoModel_for_task(
    model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    model_type = "AutoModelForSequenceClassification"
  )

  # Model is of Sequence Classification type
  expect_true(stringr::str_detect(as.character(sent_model$config), "ForSequenceClass"))
  # Sentiment model outputs 3 classes
  expect_equal(sent_model$config$num_labels, 3)
  # Sentiment model has label2id and id2label
  expect_true("label2id" %in% names(sent_model$config))
  expect_true("id2label" %in% names(sent_model$config))
})

# Pipeline tests
test_that("hf_load_pipeline loads correctly", {
  skip_on_cran()
  skip_if_no_python()

  pipe <- hf_load_pipeline(model_id = "distilbert-base-uncased")

  # loads tokenizer
  expect_true(stringr::str_detect(as.character(pipe$tokenizer), "distilb"))
  # does not load incorrect model
  expect_false(stringr::str_detect(as.character(pipe$model), "GPT|gpt"))
  # loads correct model
  expect_true(stringr::str_detect(as.character(pipe$model), "distilb"))
})

test_that("hf_load_pipeline task argument works", {
  skip_on_cran()
  skip_if_no_python()

  pipe_qa <- hf_load_pipeline(model_id = "distilbert-base-uncased", task = "question-answering")
  expect_equal("question-answering", pipe_qa$task)
})





