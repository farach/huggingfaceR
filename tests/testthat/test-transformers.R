test_that("load model fails if not given correct model nae", {
  expect_error(hf_load_model(model_id = "x not a model x"))
})


tokenizer <- hf_load_tokenizer("distilbert-base-uncased")
test_that("load tokenizer function loads correctly", {
  expect_true("vocab" %in% names(tokenizer))
})
