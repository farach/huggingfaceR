# Offline tests for the step_hf_embed() recipe step. Embedding requests are
# mocked, so these run everywhere without a token or network access.

test_that("step_hf_embed bakes embeddings into one numeric column per dimension", {
  skip_if_not_installed("recipes")
  testthat::local_mocked_bindings(
    hf_embed = function(text, model, token, ...) {
      embedding <- lapply(text, function(x) {
        if (identical(x, "skip")) NULL else c(nchar(x), 0.5)
      })
      tibble::tibble(
        text = text,
        embedding = embedding,
        n_dims = ifelse(vapply(embedding, is.null, logical(1)), NA_integer_, 2L)
      )
    }
  )
  train <- data.frame(
    text = c("a", "bb", "skip"),
    y = c(1, 2, 3),
    stringsAsFactors = FALSE
  )

  rec <- recipes::recipe(y ~ text, data = train)
  rec <- step_hf_embed(rec, text, model = "org/model")
  baked <- recipes::bake(recipes::prep(rec, training = train), new_data = train)

  expect_false("text" %in% names(baked))
  expect_true(all(c("text_emb_1", "text_emb_2") %in% names(baked)))
  expect_equal(baked$text_emb_1, c(1, 2, NA))
  expect_equal(baked$text_emb_2, c(0.5, 0.5, NA))
  expect_equal(baked$y, c(1, 2, 3))
})
