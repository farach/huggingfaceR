# --- hf_default_model -------------------------------------------------------

test_that("hf_default_model returns the expected default for a known task", {
  expect_identical(hf_default_model("translate"), "Helsinki-NLP/opus-mt-en-fr")
  expect_identical(hf_default_model("embed"), "BAAI/bge-small-en-v1.5")
  expect_identical(hf_default_model("chat"), "meta-llama/Llama-3.1-8B-Instruct")
})

test_that("every registered task resolves to a single non-empty model id", {
  registry <- hf_default_model()
  for (task in registry$task) {
    model <- hf_default_model(task)
    expect_type(model, "character")
    expect_length(model, 1L)
    expect_true(nzchar(model))
  }
})

test_that("hf_default_model() with no task returns the full registry tibble", {
  registry <- hf_default_model()
  expect_s3_class(registry, "tbl_df")
  expect_named(registry, c("task", "model"))
  expect_true(nrow(registry) >= 11L)
  expect_false(any(duplicated(registry$task)))
})

test_that("an unknown task errors informatively", {
  expect_error(hf_default_model("not_a_task"), "must be one of")
  expect_error(hf_default_model(c("chat", "embed")), "must be one of")
})

test_that("exported functions resolve their model default through the registry", {
  # The function default arg is the call hf_default_model("<task>"); evaluating
  # it must equal the registry value, proving signatures stay in sync.
  expect_identical(
    eval(formals(hf_translate)$model),
    hf_default_model("translate")
  )
  expect_identical(
    eval(formals(hf_classify)$model),
    hf_default_model("classify")
  )
  expect_identical(
    eval(formals(hf_embed)$model),
    hf_default_model("embed")
  )
})
