test_that("hf_search_models returns tibble", {
  skip_on_cran()
  skip_if_hub_unreachable()
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
  skip_if_hub_unreachable()
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
  skip_if_hub_unreachable()
  result <- hf_check_inference("nonexistent-org/fake-model-12345", quiet = TRUE)

  expect_false(result$available)
})

test_that("hf_search_datasets returns tibble", {
  skip_on_cran()
  skip_if_hub_unreachable()
  result <- hf_search_datasets(limit = 5)
  
  expect_s3_class(result, "tbl_df")
  expect_true("dataset_id" %in% names(result))
})

test_that("hf_search_spaces returns tibble", {
  skip_on_cran()
  skip_if_hub_unreachable()
  result <- hf_search_spaces(search = "chat", limit = 2)

  expect_s3_class(result, "tbl_df")
  expect_equal(nrow(result), 2)
  expect_true("space_id" %in% names(result))
})

test_that("hf_search_papers returns tibble", {
  skip_on_cran()
  skip_if_hub_unreachable()
  result <- hf_search_papers("transformers", limit = 2)

  expect_s3_class(result, "tbl_df")
  expect_equal(nrow(result), 2)
  expect_true("paper_id" %in% names(result))
})

test_that("hf_list_repo_files returns file metadata", {
  skip_on_cran()
  skip_if_hub_unreachable()
  result <- hf_list_repo_files("BAAI/bge-small-en-v1.5", recursive = FALSE)

  expect_s3_class(result, "tbl_df")
  expect_true(all(c("path", "type", "size", "oid") %in% names(result)))
  expect_true("README.md" %in% result$path)
})

test_that("hf_hub_download downloads a file", {
  skip_on_cran()
  skip_if_hub_unreachable()
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
  skip_if_hub_unreachable()
  result <- hf_list_providers("Qwen/Qwen2.5-72B-Instruct")

  expect_s3_class(result, "tbl_df")
  expect_true(all(c("provider", "status", "input_price", "supports_tools") %in% names(result)))
  expect_true(any(result$status == "live"))
})

test_that("hf_check_inference includes provider metadata", {
  skip_on_cran()
  skip_if_hub_unreachable()
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

# Serves `pages` of Hub records, linking each page to the next with a cursor.
mock_hub_pages <- function(pages, requested) {
  function(req) {
    requested(req$url)
    cursor <- httr2::url_parse(req$url)$query$cursor
    page <- if (is.null(cursor)) 1L else as.integer(cursor)
    headers <- list("Content-Type" = "application/json")
    if (page < length(pages)) {
      headers$Link <- paste0(
        "<https://huggingface.co/api/models?cursor=", page + 1L, '>; rel="next"'
      )
    }
    httr2::response(
      status_code = 200,
      url = req$url,
      headers = headers,
      body = charToRaw(as.character(jsonlite::toJSON(pages[[page]], auto_unbox = TRUE)))
    )
  }
}

test_that("hf_search_models pages through results and binds records", {
  pages <- list(
    list(
      list(
        id = "org/a", author = "org", pipeline_tag = "fill-mask", downloads = 10L,
        likes = 1L, tags = list("x", "y"), library_name = "transformers"
      ),
      list(id = "org/b", downloads = 3000000000, likes = 2L, tags = list())
    ),
    list(
      list(
        id = "org/c", author = "org", pipeline_tag = "summarization",
        downloads = 5L, likes = 0L, tags = list("z")
      )
    )
  )
  requested <- character()
  httr2::local_mocked_responses(
    mock_hub_pages(pages, function(url) requested <<- c(requested, url))
  )

  result <- hf_search_models(limit = 10)

  expect_length(requested, 2)
  expect_equal(result$model_id, c("org/a", "org/b", "org/c"))
  expect_equal(result$author, c("org", NA, "org"))
  expect_equal(result$task, c("fill-mask", NA, "summarization"))
  expect_type(result$downloads, "double")
  expect_equal(result$downloads, c(10, 3e9, 5))
  expect_equal(result$likes, c(1L, 2L, 0L))
  expect_equal(result$tags, list(c("x", "y"), character(0), "z"))
  expect_equal(result$library, c("transformers", NA, NA))

  requested <- character()
  limited <- hf_search_models(limit = 2)
  expect_length(requested, 1)
  expect_equal(limited$model_id, c("org/a", "org/b"))
})

test_that("hf_list_repo_files binds file metadata by column", {
  files <- list(
    list(type = "file", oid = "1", size = 12L, path = "README.md"),
    list(type = "directory", oid = "2", path = "data"),
    list(type = "file", oid = "3", size = 5000000000, path = "data/big.parquet")
  )
  httr2::local_mocked_responses(function(req) {
    httr2::response(
      status_code = 200,
      url = req$url,
      headers = list("Content-Type" = "application/json"),
      body = charToRaw(as.character(jsonlite::toJSON(files, auto_unbox = TRUE)))
    )
  })

  result <- hf_list_repo_files("org/repo")

  expect_equal(result$path, c("README.md", "data", "data/big.parquet"))
  expect_equal(result$type, c("file", "directory", "file"))
  expect_equal(result$size, c(12, NA, 5e9))
  expect_equal(result$oid, c("1", "2", "3"))
})

test_that("hf_bind_records combines scalar and list cells by column", {
  records <- list(
    list(id = "a", n = 1L, tags = list("x")),
    list(id = "b", n = 2.5, tags = list())
  )

  result <- hf_bind_records(records, function(record) {
    list(
      id = record$id,
      n = record$n,
      tags = list(unlist(record$tags) %||% character(0))
    )
  })

  expect_equal(
    result,
    tibble::tibble(id = c("a", "b"), n = c(1, 2.5), tags = list("x", character(0)))
  )
})

test_that("hf_bind_records falls back to row binding for irregular cells", {
  records <- list(list(a = 1, b = "x"), list(a = 2, b = c("y", "z")))

  result <- hf_bind_records(records, function(record) list(a = record$a, b = record$b))

  expect_equal(
    result,
    dplyr::bind_rows(
      tibble::tibble(a = 1, b = "x"),
      tibble::tibble(a = 2, b = c("y", "z"))
    )
  )
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
