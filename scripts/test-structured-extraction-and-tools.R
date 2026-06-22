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

cat("=== Structured Extraction and Tools Examples ===\n\n")

test("hf_extract schema example", {
  res <- hf_extract(
    "Amelie is a chef in Paris who mentions burnout.",
    c(name = "string", occupation = "string", city = "string", theme = "string"),
    max_tokens = 120
  )
  check(res$name[1] == "Amelie", "expected Amelie")
  check(res$city[1] == "Paris", "expected Paris")
  res
})

test("hf_chat streaming example", {
  pieces <- character()
  res <- hf_chat(
    "Reply with exactly: OK",
    stream = TRUE,
    callback = function(delta) pieces <<- c(pieces, delta),
    temperature = 0,
    max_tokens = 8
  )
  check(nchar(paste0(pieces, collapse = "")) > 0, "expected deltas")
  check(nchar(res$content[1]) > 0, "expected response")
  res
})

test("hf_tool and hf_run_tools example", {
  add_tool <- hf_tool("add", "Add two numbers.", c(x = "number", y = "number"))
  convo <- hf_conversation(model = "Qwen/Qwen2.5-72B-Instruct")
  convo <- chat(
    convo,
    "Use the add tool to add x=2 and y=3, then tell me the answer.",
    tools = list(add_tool),
    tool_choice = "auto",
    temperature = 0,
    max_tokens = 120
  )
  check(length(convo$history[[2]]$tool_calls) > 0, "expected tool call")
  convo <- hf_run_tools(convo, list(add = function(x, y) x + y),
                        temperature = 0, max_tokens = 120)
  check(grepl("5", convo$history[[length(convo$history)]]$content),
        "expected final answer to mention 5")
  convo
})

fail <- sum(vapply(results, function(x) x$status == "FAIL", logical(1)))
cat(sprintf("\nPASS: %d / %d\n", length(results) - fail, length(results)))
if (fail > 0) quit(status = 1)

