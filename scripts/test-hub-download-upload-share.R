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

cat("=== Hub Download, Upload, and Share Examples ===\n\n")

test("hf_whoami token metadata", {
  res <- hf_whoami()
  check("token_role" %in% names(res), "expected token_role")
  tibble::tibble(
    token_role = res$token_role,
    billing_mode = res$billing_mode,
    is_pro = res$is_pro
  )
})

test("hf_list_repo_files and hf_hub_download", {
  files <- hf_list_repo_files("BAAI/bge-small-en-v1.5", recursive = FALSE)
  check("README.md" %in% files$path, "expected README.md")
  readme <- hf_hub_download("BAAI/bge-small-en-v1.5", "README.md")
  check(file.exists(readme), "expected downloaded README")
  unlink(readme)
  files
})

test("hf_search_spaces and hf_search_papers", {
  spaces <- hf_search_spaces("chat", limit = 3)
  papers <- hf_search_papers("transformers", limit = 3)
  check(nrow(spaces) == 3, "expected Spaces")
  check(nrow(papers) == 3, "expected papers")
  list(spaces = spaces, papers = papers)
})

test("hf_list_providers", {
  providers <- hf_list_providers("Qwen/Qwen2.5-72B-Instruct")
  check(any(providers$status == "live"), "expected live provider")
  providers
})

test("write guards", {
  check(inherits(
    try(hf_create_repo("me/test-dataset", repo_type = "dataset"), silent = TRUE),
    "try-error"
  ), "expected create guard")
  check(inherits(
    try(hf_delete_repo("me/test-dataset", repo_type = "dataset"), silent = TRUE),
    "try-error"
  ), "expected delete guard")
  TRUE
})

fail <- sum(vapply(results, function(x) x$status == "FAIL", logical(1)))
cat(sprintf("\nPASS: %d / %d\n", length(results) - fail, length(results)))
if (fail > 0) quit(status = 1)
