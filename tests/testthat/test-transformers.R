test_that("load model fails if not given correct model nae", {
  expect_error(hf_load_pipeline(model_id = "x not a model x"))
})

#Tokenizer tests
tokenizer <- hf_load_tokenizer("distilbert-base-uncased")
slow_tokenizer <- hf_load_tokenizer('distilbert-base-uncased', use_fast = FALSE)


test_that("incorrect model not loaded", {
  expect_false(stringr::str_detect(tokenizer, "GPT|gpt"))
})

test_that("tokenizer loads correct model", {
  expect_true(stringr::str_detect(tokenizer,"distilbert"))
})

test_that("load tokenizer function loads correctly", {
  expect_true("vocab" %in% names(tokenizer))
})

test_that("tokenizer function properly passess ...",{
  expect_false(stringr::str_detect(slow_tokenizer, "is_fast=True"))
})

test_that("tokenizer loads fast if not told otherwise by ...",{
  expect_true(stringr::str_detect(tokenizer, "is_fast=True"))
})


#placeholder text for model tests


#pipeline tests
pipe <- hf_load_pipeline(model_id = "distilbert-base-uncased")
pipe_qa <- hf_load_pipeline(model_id = "distilbert-base-uncased", task = "question-answering")


test_that("hf_load_pipeline loads tokenizer", {
  expect_true(stringr::str_detect(pipe$tokenizer, "distilb"))
})

test_that("hf_load_pipeline does not load incorrect model", {
  expect_false(stringr::str_detect(pipe$model, "GPT|gpt"))
})

test_that("hf_load_pipeline loads correct model", {
  expect_true(stringr::str_detect(pipe$model, "distilb"))
})

test_that("hf_load_pipeline task argument is functioning", {

  expect_equal("question-answering", pipe_qa$task)
})





