test_that("hf_search_models returns tibble", {
  skip_on_cran()
  
  # Basic search should return results
  result <- hf_search_models(limit = 5)
  
  expect_s3_class(result, "tbl_df")
  expect_true("model_id" %in% names(result))
})

test_that("hf_list_tasks returns character vector", {
  tasks <- hf_list_tasks()
  
  expect_type(tasks, "character")
  expect_true(length(tasks) > 0)
  expect_true("text-classification" %in% tasks)
})

test_that("hf_list_tasks filters by pattern", {
  tasks <- hf_list_tasks(pattern = "classification")
  
  expect_type(tasks, "character")
  expect_true(all(grepl("classification", tasks, ignore.case = TRUE)))
})

test_that("hf_search_datasets returns tibble", {
  skip_on_cran()
  
  result <- hf_search_datasets(limit = 5)
  
  expect_s3_class(result, "tbl_df")
  expect_true("dataset_id" %in% names(result))
})

# Helper function
skip_on_cran <- function() {
  if (identical(Sys.getenv("NOT_CRAN"), "true")) {
    return(invisible(TRUE))
  }
  testthat::skip("Skipping on CRAN")
}
