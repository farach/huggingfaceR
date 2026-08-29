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
