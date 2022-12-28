imdb_datasets <- hf_list_datasets("imdb")
test_that("hf_list_datasets is still returning a character vector", {
  expect_equal("character", class(imdb_datasets))
})

emo_datasets <- hf_list_datasets('emo')
test_that("hf_list_datasets returns a vector longer than 1 for emotions pattern", {
  expect_gt(length(emo_datasets), 1)
})
