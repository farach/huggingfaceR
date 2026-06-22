devtools::load_all()

results <- list()

test <- function(name, expr) {
  cat(sprintf("  %-50s", name))
  tryCatch(
    {
      value <- eval(expr)
      cat("[PASS]\n")
      results[[name]] <<- list(status = "PASS", value = value)
      value
    },
    error = function(e) {
      cat(sprintf("[FAIL] %s\n", conditionMessage(e)))
      results[[name]] <<- list(status = "FAIL", error = conditionMessage(e))
      NULL
    }
  )
}

check <- function(condition, msg = "assertion failed") {
  if (!isTRUE(condition)) stop(msg, call. = FALSE)
}

cat("=== Multimodal Examples ===\n\n")

image <- "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cat.png"
audio <- "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"

test("hf_transcribe example", {
  res <- hf_transcribe(audio, return_timestamps = "word")
  check(grepl("dream", res$text[1], ignore.case = TRUE), "expected dream")
  check(length(res$chunks[[1]]) > 0, "expected chunks")
  res
})

test("hf_text_to_image example", {
  res <- hf_text_to_image(
    "a small red cube on a white background",
    seed = 1,
    num_inference_steps = 2,
    guidance_scale = 0
  )
  check(file.exists(res$path[1]), "expected generated file")
  unlink(res$path[1])
  res
})

test("hf_classify_image example", {
  res <- hf_classify_image(image, top_k = 3)
  check(nrow(res) == 3, "expected three labels")
  res
})

test("hf_caption_image example", {
  res <- hf_caption_image(image, max_tokens = 40, temperature = 0)
  check(nchar(res$caption[1]) > 0, "expected caption")
  res
})

test("hf_detect_objects example", {
  res <- hf_detect_objects(image, threshold = 0.5)
  check(nrow(res) > 0, "expected object boxes")
  check(all(c("xmin", "ymin", "xmax", "ymax") %in% names(res)), "expected box columns")
  res
})

fail <- sum(vapply(results, function(x) x$status == "FAIL", logical(1)))
cat(sprintf("\nPASS: %d / %d\n", length(results) - fail, length(results)))
if (fail > 0) quit(status = 1)

