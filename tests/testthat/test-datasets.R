emo <- hf_load_dataset("emo")

test_that("hf_load_dataset does not load datasets that do not exist", {
  expect_error(hf_load_dataset())
})

test_that("hf_load_dataset returns all splits if given no further arugments", {
  expect_equal(2, length(names(emo)))
})

train_tibble_int2str <- hf_load_dataset("emo", split = "train")
test_that("as_tibble, split = 'train' and label_name load a tibble", {
  expect_equal(class(train_tibble_int2str)[1], "tbl_df")
})

test_tibble_int2str <- hf_load_dataset("emo", split = "test")
test_that("emotions test dataset has 2k rows", {
  expect_equal(nrow(test_tibble_int2str), 5509)
})

#No validation set in emo as opposed to previous emotion dataset, deprecate test
# validation_tibble_in2str <- hf_load_dataset("emo", split = "validation")
# test_that("validation set loads and first row is different to test", {
#   expect_false(validation_tibble_in2str$text[1] == test_tibble_int2str$text[1])
# })

#Old functionality, deprecate test
# test_that("hf_load_dataset test split can be specified without tibble = TRUE & label_name", {
  # testthat::expect_equal(class(hf_load_dataset("emotion", split = "test"))[1], "datasets.arrow_dataset.Dataset")
# })

int_to_str <- hf_load_dataset("emo", label_conversion = "int2str")
test_that("int2str is working", {
  expect_true(int_to_str[[1]][1, 3]$label_name == "others")
})
