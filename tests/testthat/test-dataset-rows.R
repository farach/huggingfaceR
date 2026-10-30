# Tests for converting datasets-server row payloads into tibbles.
#
# Row fields are not always scalars. Datasets such as openai/gdpval carry
# variable-length fields, and converting those rows naively recycles a single
# source row into several output rows. These tests are offline: they exercise
# hf_rows_to_tibble() against payloads shaped like real API responses.

test_that("scalar-only rows produce one row each", {
  rows <- list(
    list(text = "a", label = 0L),
    list(text = "b", label = 1L)
  )

  out <- hf_rows_to_tibble(rows)

  expect_s3_class(out, "tbl_df")
  expect_equal(nrow(out), 2)
  expect_equal(out$text, c("a", "b"))
  expect_equal(out$label, c(0L, 1L))
  expect_false(is.list(out$text))
})

test_that("variable-length fields become list-columns without multiplying rows", {
  # This is the openai/gdpval shape from issue #61: one scalar field and one
  # field whose length differs per row. Previously the row with three files
  # expanded into three rows.
  rows <- list(
    list(task_id = "t1", reference_files = list("a.pdf", "b.pdf", "c.pdf")),
    list(task_id = "t2", reference_files = list("d.pdf"))
  )

  out <- hf_rows_to_tibble(rows)

  expect_equal(nrow(out), 2)
  expect_equal(out$task_id, c("t1", "t2"))
  expect_true(is.list(out$reference_files))
  expect_length(out$reference_files[[1]], 3)
  expect_length(out$reference_files[[2]], 1)
})

test_that("a field that is a list in every row still yields one row per row", {
  rows <- list(
    list(id = "a", tags = list("x", "y")),
    list(id = "b", tags = list("z", "w"))
  )

  out <- hf_rows_to_tibble(rows)

  expect_equal(nrow(out), 2)
  expect_true(is.list(out$tags))
  expect_equal(out$tags[[2]], list("z", "w"))
})

test_that("null and missing fields are handled without dropping rows", {
  rows <- list(
    list(id = "a", note = "hello", files = list("f1")),
    list(id = "b", note = NULL),
    list(id = "c", files = list("f2", "f3"))
  )

  out <- hf_rows_to_tibble(rows)

  expect_equal(nrow(out), 3)
  expect_equal(out$id, c("a", "b", "c"))
  # `note` is scalar throughout, so a null becomes NA rather than a list cell.
  expect_false(is.list(out$note))
  expect_true(is.na(out$note[2]))
  # `files` is a list-column; the row that lacks it gets an empty list.
  expect_true(is.list(out$files))
  expect_length(out$files[[2]], 0)
  expect_length(out$files[[3]], 2)
})

test_that("empty input returns an empty tibble", {
  expect_equal(nrow(hf_rows_to_tibble(list())), 0)
  expect_s3_class(hf_rows_to_tibble(list()), "tbl_df")
})

test_that("a zero-length field does not collapse the row", {
  rows <- list(
    list(id = "a", empties = list()),
    list(id = "b", empties = list("x"))
  )

  out <- hf_rows_to_tibble(rows)

  expect_equal(nrow(out), 2)
  expect_true(is.list(out$empties))
  expect_length(out$empties[[1]], 0)
})

test_that("scalar columns keep the combined type of their values", {
  rows <- list(
    list(n = 1L, x = "a"),
    list(n = 2.5, x = NULL),
    list(n = NULL, x = "c")
  )

  out <- hf_rows_to_tibble(rows)

  expect_type(out$n, "double")
  expect_equal(out$n, c(1, 2.5, NA))
  expect_equal(out$x, c("a", NA, "c"))
})

test_that("a length-one list value makes a list-column", {
  rows <- list(
    list(id = "a", tags = list("x")),
    list(id = "b", tags = "y")
  )

  out <- hf_rows_to_tibble(rows)

  expect_true(is.list(out$tags))
  expect_equal(out$tags[[1]], list("x"))
  expect_equal(out$tags[[2]], "y")
})

test_that("non-list rows are read like named lists", {
  rows <- list(c(id = 1, score = 0.5), NULL, list(id = 3))

  out <- hf_rows_to_tibble(rows)

  expect_equal(nrow(out), 3)
  expect_equal(out$id, c(1, NA, 3))
  expect_equal(out$score, c(0.5, NA, NA))
})

test_that("hf_load_dataset pages through rows in order", {
  offsets <- character()
  httr2::local_mocked_responses(function(req) {
    query <- httr2::url_parse(req$url)$query
    offset <- as.integer(query$offset)
    ids <- seq.int(offset, length.out = min(as.integer(query$length), 250L - offset))
    offsets <<- c(offsets, query$offset)
    payload <- list(rows = lapply(ids, function(i) {
      list(
        row_idx = i,
        row = list(id = i, text = paste("row", i), tags = as.list(rep("t", i %% 3))),
        truncated_cells = list()
      )
    }))
    httr2::response(
      status_code = 200,
      url = req$url,
      headers = list("Content-Type" = "application/json"),
      body = charToRaw(as.character(jsonlite::toJSON(payload, auto_unbox = TRUE)))
    )
  })

  out <- hf_load_dataset("org/data", split = "train", config = "default", limit = 230)

  expect_equal(offsets, c("0", "100", "200"))
  expect_equal(nrow(out), 230)
  expect_equal(out$id, 0:229)
  expect_equal(out$text[c(1, 230)], c("row 0", "row 229"))
  expect_true(is.list(out$tags))
  expect_length(out$tags[[1]], 0)
  expect_length(out$tags[[3]], 2)
  expect_equal(unique(out$.dataset), "org/data")
})
