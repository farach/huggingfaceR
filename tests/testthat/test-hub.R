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

test_that("hf_check_inference returns list with expected structure", {
  skip_on_cran()

  result <- hf_check_inference("BAAI/bge-small-en-v1.5", quiet = TRUE)

  expect_type(result, "list")
  expect_true("model_id" %in% names(result))
  expect_true("available" %in% names(result))
  expect_true("pipeline_tag" %in% names(result))
  expect_equal(result$model_id, "BAAI/bge-small-en-v1.5")
  expect_type(result$available, "logical")
})

test_that("hf_check_inference reports FALSE for nonexistent model", {
  skip_on_cran()

  result <- hf_check_inference("nonexistent-org/fake-model-12345", quiet = TRUE)

  expect_false(result$available)
})

test_that("hf_search_datasets returns tibble", {
  skip_on_cran()
  
  result <- hf_search_datasets(limit = 5)
  
  expect_s3_class(result, "tbl_df")
  expect_true("dataset_id" %in% names(result))
})

test_that("hf_search_spaces returns tibble", {
  skip_on_cran()

  result <- hf_search_spaces(search = "chat", limit = 2)

  expect_s3_class(result, "tbl_df")
  expect_equal(nrow(result), 2)
  expect_true("space_id" %in% names(result))
})

test_that("hf_search_papers returns tibble", {
  skip_on_cran()

  result <- hf_search_papers("transformers", limit = 2)

  expect_s3_class(result, "tbl_df")
  expect_equal(nrow(result), 2)
  expect_true("paper_id" %in% names(result))
})

test_that("hf_list_repo_files returns file metadata", {
  skip_on_cran()

  result <- hf_list_repo_files("BAAI/bge-small-en-v1.5", recursive = FALSE)

  expect_s3_class(result, "tbl_df")
  expect_true(all(c("path", "type", "size", "oid") %in% names(result)))
  expect_true("README.md" %in% result$path)
})

test_that("hf_hub_download downloads a file", {
  skip_on_cran()

  dest <- tempfile(fileext = ".md")
  result <- hf_hub_download("BAAI/bge-small-en-v1.5", "README.md", dest = dest)

  expect_equal(
    normalizePath(result, winslash = "/", mustWork = FALSE),
    normalizePath(dest, winslash = "/", mustWork = FALSE)
  )
  expect_true(file.exists(dest))
  expect_gt(file.info(dest)$size, 0)
})

test_that("hf_list_providers parses router metadata", {
  skip_on_cran()

  result <- hf_list_providers("Qwen/Qwen2.5-72B-Instruct")

  expect_s3_class(result, "tbl_df")
  expect_true(all(c("provider", "status", "input_price", "supports_tools") %in% names(result)))
  expect_true(any(result$status == "live"))
})

test_that("hf_check_inference includes provider metadata", {
  skip_on_cran()

  result <- hf_check_inference("Qwen/Qwen2.5-72B-Instruct", quiet = TRUE)

  expect_true("providers" %in% names(result))
  expect_s3_class(result$providers, "tbl_df")
  expect_true(result$available)
})

test_that("pagination link parser extracts next URLs", {
  link <- '<https://huggingface.co/api/spaces?cursor=abc>; rel="next", <x>; rel="prev"'

  expect_equal(hf_link_next(link), "https://huggingface.co/api/spaces?cursor=abc")
  expect_null(hf_link_next(NULL))
  expect_null(hf_link_next('<x>; rel="prev"'))
})

test_that("repo helpers build expected URLs and destinations", {
  expect_equal(
    hf_repo_resolve_url("dataset", "org/data", "main", "data/file.csv"),
    "https://huggingface.co/datasets/org/data/resolve/main/data/file.csv"
  )
  expect_equal(hf_split_repo_id("org/repo")$namespace, "org")
  expect_equal(hf_split_repo_id("repo")$name, "repo")
  expect_error(hf_split_repo_id("a/b/c"), "repo_id")
})

test_that("write helpers require explicit confirmation", {
  expect_error(
    hf_create_repo("me/test", token = "hf_fake"),
    "confirm = TRUE"
  )
  expect_error(
    hf_upload_file(tempfile(), "me/test", token = "hf_fake"),
    "confirm = TRUE"
  )
  expect_error(
    hf_delete_repo("me/test", token = "hf_fake"),
    "confirm = TRUE"
  )
  expect_error(
    hf_push_dataset(data.frame(x = 1), "me/test", token = "hf_fake"),
    "confirm = TRUE"
  )
})

test_that("hf_push_dataset validates data frames before writing", {
  expect_error(
    hf_push_dataset(list(x = 1), "me/test", token = "hf_fake", confirm = TRUE),
    "data frame"
  )
})
