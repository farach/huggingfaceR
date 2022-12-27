imdb_datasets <- hf_list_datasets("imdb")
test_that("hf_list_datasets is still returning a character vector", {
  expect_equal("character", class(imdb_datasets))
})

emotions_datasets <- hf_list_datasets('emotion')
test_that("hf_list_datasets returns a vector longer than 1 for emotions pattern", {
  expect_gt(length(emotions_datasets), 1)
})
