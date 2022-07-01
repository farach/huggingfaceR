test_that("load model fails if not given correct model nae", {
  expect_error(hf_load_model(model_id = "x not a model x"))
})

test_that("multiplication works", {
  expect_equal(2 * 2, 4)
})

tokenizer <- hf_load_tokenizer("distilbert-base-uncased")
test_that("load tokenizer function loads correctly", {
  expect_true("vocab" %in% names(tokenizer))
})
