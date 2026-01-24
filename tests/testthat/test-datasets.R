test_that("hf_load_dataset requires dataset argument", {
  expect_error(hf_load_dataset())
})

test_that("hf_load_dataset works with full dataset ID", {
  skip_on_cran()
  result <- hf_load_dataset("stanfordnlp/imdb", split = "train", limit = 5)
  expect_s3_class(result, "tbl_df")
  expect_equal(nrow(result), 5)
  expect_true("text" %in% names(result))
  expect_true("label" %in% names(result))
  expect_true(".dataset" %in% names(result))
  expect_true(".split" %in% names(result))
})

test_that("hf_load_dataset resolves short dataset names", {
  skip_on_cran()
  result <- hf_load_dataset("imdb", split = "test", limit = 3)
  expect_s3_class(result, "tbl_df")
  expect_equal(nrow(result), 3)
})

test_that("hf_load_dataset respects config parameter", {
  skip_on_cran()
  result <- hf_load_dataset("stanfordnlp/imdb", split = "train",
                            config = "plain_text", limit = 2)
  expect_s3_class(result, "tbl_df")
  expect_equal(nrow(result), 2)
})

test_that("hf_load_dataset errors on non-existent dataset", {
  skip_on_cran()
  expect_error(hf_load_dataset("this-dataset-does-not-exist-xyz123"))
})
