# Offline tests for the dataset viewer helpers. Every request is mocked.

viewer_json <- function(req, body, status = 200L) {
  httr2::response(
    status_code = status,
    url = req$url,
    headers = list("Content-Type" = "application/json"),
    body = charToRaw(as.character(jsonlite::toJSON(body, auto_unbox = TRUE, null = "null")))
  )
}

viewer_path <- function(req) {
  httr2::url_parse(req$url)$path
}

viewer_query <- function(req) {
  httr2::url_parse(req$url)$query
}

# Rows payload for ids [offset, offset + length) of a split with `total` rows.
viewer_rows_payload <- function(query, total, partial = FALSE) {
  offset <- as.integer(query$offset)
  size <- max(0L, min(as.integer(query$length), total - offset))
  ids <- seq.int(offset, length.out = size)
  list(
    rows = lapply(ids, function(i) {
      list(row_idx = i, row = list(id = i, text = paste("row", i)), truncated_cells = list())
    }),
    num_rows_total = total,
    num_rows_per_page = 100L,
    partial = partial
  )
}

parquet_bytes <- function(df) {
  path <- tempfile(fileext = ".parquet")
  on.exit(unlink(path))
  arrow::write_parquet(df, path)
  readBin(path, "raw", file.size(path))
}

test_that("hf_load_dataset requests later pages concurrently and keeps order", {
  requested <- character()
  batches <- 0L
  real_batch <- hf_perform_batch
  local_mocked_bindings(hf_perform_batch = function(reqs, ...) {
    batches <<- batches + 1L
    real_batch(reqs, ...)
  })
  httr2::local_mocked_responses(function(req) {
    query <- viewer_query(req)
    requested <<- c(requested, paste(query$offset, query$length))
    viewer_json(req, viewer_rows_payload(query, total = 250L))
  })

  out <- hf_load_dataset("org/data", config = "default", limit = Inf, max_active = 3)

  expect_equal(batches, 1L)
  expect_setequal(requested, c("0 100", "100 100", "200 50"))
  expect_equal(nrow(out), 250)
  expect_equal(out$id, 0:249)
  expect_equal(unique(out$.split), "train")
})

test_that("hf_load_dataset pages respect offset and limit", {
  requested <- character()
  httr2::local_mocked_responses(function(req) {
    query <- viewer_query(req)
    requested <<- c(requested, paste(query$offset, query$length))
    viewer_json(req, viewer_rows_payload(query, total = 1000L))
  })

  out <- hf_load_dataset("org/data", config = "default", offset = 50, limit = 180)

  expect_setequal(requested, c("50 100", "150 80"))
  expect_equal(out$id, 50:229)
})

test_that("hf_load_dataset gives the same rows one request at a time", {
  requested <- character()
  local_mocked_bindings(hf_perform_batch = function(...) stop("unexpected batch"))
  httr2::local_mocked_responses(function(req) {
    query <- viewer_query(req)
    requested <<- c(requested, query$offset)
    viewer_json(req, viewer_rows_payload(query, total = 250L))
  })

  out <- hf_load_dataset("org/data", config = "default", limit = Inf, max_active = 1)

  expect_equal(requested, c("0", "100", "200"))
  expect_equal(out$id, 0:249)
})

test_that("a failing later page stops hf_load_dataset with the API message", {
  httr2::local_mocked_responses(function(req) {
    query <- viewer_query(req)
    if (query$offset == "100") {
      return(viewer_json(req, list(error = "Bad page"), status = 400L))
    }
    viewer_json(req, viewer_rows_payload(query, total = 300L))
  })

  expect_error(
    hf_load_dataset("org/data", config = "default", limit = Inf, max_active = 2),
    "API error: Bad page"
  )
})

test_that("hf_load_dataset rejects an invalid max_active", {
  expect_error(hf_load_dataset("org/data", max_active = 0), "max_active")
})

test_that("hf_load_dataset reads Parquet shards in order and slices rows", {
  skip_if_not_installed("arrow")
  shards <- list(
    "https://hf.test/0000.parquet" = parquet_bytes(data.frame(id = 0:4, text = letters[1:5])),
    "https://hf.test/0001.parquet" = parquet_bytes(data.frame(id = 5:9, text = letters[6:10]))
  )
  downloaded <- character()
  httr2::local_mocked_responses(function(req) {
    if (viewer_path(req) == "/parquet") {
      expect_equal(viewer_query(req)$config, "default")
      # Listed out of order, and with another split, to check both are handled.
      return(viewer_json(req, list(
        parquet_files = list(
          list(split = "train", filename = "0001.parquet", url = names(shards)[2]),
          list(split = "test", filename = "0000.parquet", url = "https://hf.test/test.parquet"),
          list(split = "train", filename = "0000.parquet", url = names(shards)[1])
        ),
        partial = FALSE
      )))
    }
    downloaded <<- c(downloaded, req$url)
    httr2::response(status_code = 200, url = req$url, body = shards[[req$url]])
  })

  all_rows <- hf_load_dataset("org/data", config = "default", limit = Inf,
                              format = "parquet")
  expect_equal(all_rows$id, 0:9)
  expect_equal(all_rows$text, letters[1:10])
  expect_equal(unique(all_rows$.dataset), "org/data")

  downloaded <- character()
  middle <- hf_load_dataset("org/data", config = "default", offset = 3, limit = 4,
                            format = "parquet", max_active = 1)
  expect_equal(middle$id, 3:6)
  expect_equal(downloaded, names(shards))

  downloaded <- character()
  first <- hf_load_dataset("org/data", config = "default", limit = 2,
                           format = "parquet", max_active = 1)
  expect_equal(first$id, 0:1)
  expect_equal(downloaded, names(shards)[1])

  sliced <- hf_load_dataset("org/data", split = "train[8:20]", config = "default",
                            format = "parquet")
  expect_equal(sliced$id, 8:9)
  expect_equal(unique(sliced$.split), "train[8:20]")
})

test_that("hf_load_dataset explains a missing Parquet export", {
  skip_if_not_installed("arrow")
  httr2::local_mocked_responses(function(req) {
    viewer_json(req, list(parquet_files = list(), partial = FALSE))
  })

  expect_error(
    hf_load_dataset("org/data", config = "default", format = "parquet"),
    "No Parquet export"
  )
})

test_that("a finite Parquet load downloads only the shards it needs", {
  skip_if_not_installed("arrow")
  urls <- sprintf("https://hf.test/%04d.parquet", 0:3)
  shards <- stats::setNames(lapply(0:3, function(s) {
    parquet_bytes(data.frame(id = s * 5L + 0:4))
  }), urls)
  downloaded <- character()
  httr2::local_mocked_responses(function(req) {
    if (viewer_path(req) == "/parquet") {
      return(viewer_json(req, list(
        parquet_files = lapply(seq_along(urls), function(i) {
          list(split = "train", filename = basename(urls[i]), url = urls[i])
        }),
        partial = FALSE
      )))
    }
    downloaded <<- c(downloaded, req$url)
    httr2::response(status_code = 200, url = req$url, body = shards[[req$url]])
  })

  first <- hf_load_dataset("org/data", config = "default", limit = 2,
                           format = "parquet", max_active = 4)
  expect_equal(first$id, 0:1)
  expect_equal(downloaded, urls[1])

  # After one 5-row shard, 7 more rows need two more shards, not three.
  downloaded <- character()
  spanning <- hf_load_dataset("org/data", config = "default", offset = 3,
                              limit = 9, format = "parquet", max_active = 4)
  expect_equal(spanning$id, 3:11)
  expect_setequal(downloaded, urls[1:3])

  downloaded <- character()
  expect_warning(
    none <- hf_load_dataset("org/data", config = "default", limit = 0,
                            format = "parquet"),
    "No rows returned"
  )
  expect_equal(nrow(none), 0L)
  expect_length(downloaded, 0L)
})

test_that("hf_dataset_search sends the query and keeps row positions", {
  seen <- NULL
  httr2::local_mocked_responses(function(req) {
    seen <<- viewer_query(req)
    expect_equal(viewer_path(req), "/search")
    viewer_json(req, list(
      rows = list(
        list(row_idx = 41L, row = list(text = "dull film"), truncated_cells = list()),
        list(row_idx = 7L, row = list(text = "so dull"), truncated_cells = list())
      ),
      num_rows_total = 2L,
      partial = TRUE
    ))
  })

  expect_message(
    out <- hf_dataset_search("org/data", "dull", config = "default", token = "hf_test"),
    "first part"
  )

  expect_equal(seen$query, "dull")
  expect_equal(seen$split, "train")
  expect_equal(seen$length, "100")
  expect_equal(out$text, c("dull film", "so dull"))
  expect_equal(out$.row_idx, c(41L, 7L))
  expect_equal(unique(out$.dataset), "org/data")
})

test_that("hf_dataset_filter sends where and orderby only when given", {
  seen <- list()
  httr2::local_mocked_responses(function(req) {
    seen[[length(seen) + 1L]] <<- viewer_query(req)
    expect_equal(viewer_path(req), "/filter")
    viewer_json(req, list(rows = list(), num_rows_total = 0L, partial = FALSE))
  })

  out <- hf_dataset_filter("org/data", where = "\"label\" = 1",
                           orderby = "\"score\" DESC", config = "default")
  hf_dataset_filter("org/data", where = "\"label\" = 1", config = "default")

  expect_equal(seen[[1]]$where, "\"label\" = 1")
  expect_equal(seen[[1]]$orderby, "\"score\" DESC")
  expect_null(seen[[2]]$orderby)
  expect_equal(nrow(out), 0)
  expect_named(out, c(".row_idx", ".dataset", ".split"))
})

test_that("viewer queries validate their inputs", {
  expect_error(hf_dataset_search("org/data", ""), "query")
  expect_error(hf_dataset_filter("org/data", where = NA_character_), "where")
  expect_error(hf_dataset_filter("org/data", "x", orderby = 1), "orderby")
  expect_error(hf_dataset_search("org/data", "x", split = "train[:10]"), "Slice syntax")
  expect_error(hf_dataset_statistics("org/data", split = "train[:10]"), "Slice syntax")
  expect_error(hf_dataset_validity(character()), "character vector")
})

test_that("hf_dataset_statistics returns one tidy row per column", {
  httr2::local_mocked_responses(function(req) {
    expect_equal(viewer_path(req), "/statistics")
    viewer_json(req, list(
      num_examples = 8530L,
      partial = FALSE,
      statistics = list(
        list(
          column_name = "label",
          column_type = "class_label",
          column_statistics = list(
            nan_count = 0L, nan_proportion = 0, no_label_count = 0L,
            n_unique = 2L, frequencies = list(neg = 4265L, pos = 4265L)
          )
        ),
        list(
          column_name = "text",
          column_type = "string_text",
          column_statistics = list(
            nan_count = 0L, nan_proportion = 0, min = 4L, max = 267L,
            mean = 113.97, median = 111, std = 51.05,
            histogram = list(hist = c(302L, 955L), bin_edges = c(4L, 31L, 58L))
          )
        ),
        list(
          column_name = "created",
          column_type = "datetime",
          column_statistics = list(
            nan_count = 1L, nan_proportion = 0.1,
            min = "2020-01-01 00:00:00", max = "2021-01-01 00:00:00"
          )
        )
      )
    ))
  })

  stats <- hf_dataset_statistics("org/data", config = "default")

  expect_equal(stats$column, c("label", "text", "created"))
  expect_equal(stats$type, c("class_label", "string_text", "datetime"))
  expect_equal(stats$n_unique, c(2L, NA, NA))
  expect_equal(stats$max, c(NA, 267, NA))
  expect_equal(stats$frequencies[[1]]$value, c("neg", "pos"))
  expect_equal(stats$frequencies[[1]]$count, c(4265L, 4265L))
  expect_null(stats$frequencies[[2]])
  expect_equal(stats$histogram[[2]]$bin_start, c(4L, 31L))
  expect_equal(stats$histogram[[2]]$bin_end, c(31L, 58L))
  expect_equal(stats$histogram[[2]]$count, c(302L, 955L))
  expect_equal(stats$statistics[[3]]$min, "2020-01-01 00:00:00")
  expect_equal(attr(stats, "num_examples"), 8530L)
})

test_that("hf_dataset_validity reports each dataset in input order", {
  httr2::local_mocked_responses(function(req) {
    expect_equal(viewer_path(req), "/is-valid")
    dataset <- viewer_query(req)$dataset
    viewer_json(req, list(
      viewer = TRUE, preview = TRUE,
      search = dataset == "org/a", filter = dataset == "org/a",
      statistics = TRUE
    ))
  })

  out <- hf_dataset_validity(c("org/a", "org/b"))

  expect_equal(out$dataset, c("org/a", "org/b"))
  expect_equal(out$search, c(TRUE, FALSE))
  expect_equal(out$viewer, c(TRUE, TRUE))
})

test_that("viewer errors name the dataset", {
  httr2::local_mocked_responses(function(req) {
    viewer_json(req, list(error = "The dataset does not exist"), status = 401L)
  })

  expect_error(hf_dataset_validity("org/missing"), "API error: The dataset does not exist")
})

test_that("a viewer 429 points to a token and the Parquet format", {
  resp <- httr2::response(
    status_code = 429,
    headers = list("Content-Type" = "text/html"),
    body = charToRaw("<html><p>We had to rate limit you.</p></html>")
  )
  msg <- hf_viewer_error_body("org/data", "train")(resp)

  expect_match(msg, "rate limit reached")
  expect_match(msg, "hf_set_token")
  expect_match(msg, 'format = "parquet"', fixed = TRUE)
})
