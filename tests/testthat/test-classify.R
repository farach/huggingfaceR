test_that("hf_classify returns tibble", {
  skip_on_cran()
  
  # Test with empty input
  result <- hf_classify(character())
  expect_s3_class(result, "tbl_df")
  expect_equal(nrow(result), 0)
})

test_that("hf_classify handles NA values", {
  skip_on_cran()
  
  result <- hf_classify(NA_character_)
  expect_s3_class(result, "tbl_df")
  expect_true(is.na(result$label[1]))
})

test_that("hf_classify batches non-missing inputs", {
  skip_on_cran()

  calls <- 0L
  testthat::local_mocked_bindings(
    hf_classify_batch = function(text, model, token, batch_size, max_active, progress, endpoint_url) {
      calls <<- calls + 1L
      expect_equal(text, c("good", "bad"))
      expect_equal(batch_size, 2L)
      expect_equal(max_active, 1L)
      expect_false(progress)

      tibble::tibble(
        text = text,
        label = c("POSITIVE", "NEGATIVE"),
        score = c(0.99, 0.98),
        .input_idx = seq_along(text),
        .error = FALSE,
        .error_msg = NA_character_
      )
    }
  )

  result <- hf_classify(c("good", NA, "bad"))

  expect_equal(calls, 1L)
  expect_equal(result$text, c("good", NA, "bad"))
  expect_equal(result$label, c("POSITIVE", NA, "NEGATIVE"))
  expect_equal(result$score, c(0.99, NA, 0.98))
})

test_that("hf_classify_zero_shot requires labels", {
  skip_on_cran()
  
  expect_error(
    hf_classify_zero_shot("test text", labels = character()),
    "At least one label"
  )
})

test_that("hf_classify_zero_shot returns correct structure", {
  skip_on_cran()
  
  # Test with empty input
  result <- hf_classify_zero_shot(
    character(),
    labels = c("positive", "negative")
  )
  
  expect_s3_class(result, "tbl_df")
  expect_equal(nrow(result), 0)
})
