emotion <- hf_load_dataset("emotion")

test_that("hf_load_dataset does not load datasets that do not exist",{
  expect_error(hf_load_dataset())
})

test_that("hf_load_dataset returns all splits if given no further arugments",{
  expect_equal(3, length(names(emotion)))
})

train_tibble_int2str <- hf_load_dataset("emotion", as_tibble = TRUE, split = "train", label_name ="int2str")
test_that("as_tibble, split = 'train' and label_name load a tibble", {
  expect_equal(class(train_tibble_int2str)[1], "tbl_df")
})

test_tibble_int2str <-hf_load_dataset('emotion', split = "test", as_tibble = TRUE, label_name = "int2str")
test_that("emotions test dataset has 2k rows",{
  expect_equal(nrow(test_tibble_int2str), 2000)
})

validation_tibble_in2str <- hf_load_dataset('emotion', split = "validation", as_tibble = TRUE, label_name = "in2str")
test_that('validation set loads and first row is different to test',{
  expect_false(validation_tibble_in2str$text[1] == test_tibble_int2str$text[1])
})

test_that('hf_load_dataset test split can be specified without tibble = TRUE & label_name',{
  testthat::expect_success(hf_load_dataset("emotions", split = "test"))
})
