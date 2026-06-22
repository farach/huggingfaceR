test_that("hf_tool builds an OpenAI-compatible tool definition", {
  tool <- hf_tool(
    "get_weather",
    "Get current weather.",
    c(city = "string", units = "string")
  )

  expect_equal(tool$type, "function")
  expect_equal(tool$`function`$name, "get_weather")
  expect_equal(
    names(tool$`function`$parameters$properties),
    c("city", "units")
  )
})

test_that("hf_chat forwards tools and returns tool_calls", {
  captured <- NULL
  tool_call <- list(
    id = "call_1",
    type = "function",
    `function` = list(name = "add", arguments = '{"x":2,"y":3}')
  )
  testthat::local_mocked_bindings(
    hf_perform_chat_request = function(body, token = NULL, endpoint_url = NULL) {
      captured <<- body
      list(
        choices = list(list(message = list(
          role = "assistant",
          content = NULL,
          tool_calls = list(tool_call)
        ))),
        usage = list(completion_tokens = 4)
      )
    }
  )

  tool <- hf_tool("add", "Add two numbers.", c(x = "number", y = "number"))
  res <- hf_chat(
    "Add 2 and 3",
    tools = list(tool),
    tool_choice = "add",
    token = "token"
  )

  expect_equal(captured$tools[[1]]$`function`$name, "add")
  expect_equal(captured$tool_choice$`function`$name, "add")
  expect_equal(res$content, "")
  expect_equal(res$tool_calls[[1]][[1]]$id, "call_1")
})

test_that("hf_chat supports streaming callback path", {
  captured <- NULL
  deltas <- character()
  testthat::local_mocked_bindings(
    hf_perform_chat_stream = function(body, callback = NULL, token = NULL,
                                      endpoint_url = NULL) {
      captured <<- body
      callback("Hel")
      callback("lo")
      list(choices = list(list(message = list(
        role = "assistant",
        content = "Hello"
      ))))
    }
  )

  res <- hf_chat(
    "Say hello",
    stream = TRUE,
    callback = function(delta) deltas <<- c(deltas, delta),
    token = "token"
  )

  expect_equal(deltas, c("Hel", "lo"))
  expect_equal(res$content, "Hello")
  expect_null(captured$stream)
})

test_that("hf_chat builds multimodal image content", {
  captured <- NULL
  testthat::local_mocked_bindings(
    hf_perform_chat_request = function(body, token = NULL, endpoint_url = NULL) {
      captured <<- body
      list(choices = list(list(message = list(
        role = "assistant",
        content = "A tiny image."
      ))))
    }
  )

  res <- hf_chat(
    "Describe this",
    image = as.raw(c(0x89, 0x50, 0x4e, 0x47)),
    token = "token"
  )

  content <- captured$messages[[1]]$content
  expect_equal(content[[1]]$type, "text")
  expect_equal(content[[2]]$type, "image_url")
  expect_match(content[[2]]$image_url$url, "^data:image/png;base64,")
  expect_equal(res$content, "A tiny image.")
})

test_that("hf_describe_image maps images to descriptions", {
  captured_images <- character()
  testthat::local_mocked_bindings(
    hf_chat = function(message, model, max_tokens, token = NULL, image = NULL,
                       endpoint_url = NULL, ...) {
      captured_images <<- c(captured_images, image)
      tibble::tibble(
        role = "assistant",
        content = paste("description for", image),
        model = model,
        tokens_used = 4,
        tool_calls = list(list())
      )
    }
  )

  res <- hf_describe_image(c("https://example.com/a.png", "https://example.com/b.png"))

  expect_equal(captured_images, c("https://example.com/a.png", "https://example.com/b.png"))
  expect_named(res, c("image", "description"))
  expect_equal(nrow(res), 2)
})

test_that("hf_run_tools executes tool calls and appends final response", {
  tool_call <- list(
    id = "call_1",
    type = "function",
    `function` = list(name = "add", arguments = '{"x":2,"y":3}')
  )
  convo <- hf_conversation()
  convo$history <- list(
    list(role = "user", content = "Add 2 and 3"),
    list(role = "assistant", content = "", tool_calls = list(tool_call))
  )

  testthat::local_mocked_bindings(
    hf_perform_chat_request = function(body, token = NULL, endpoint_url = NULL) {
      expect_equal(body$messages[[3]]$role, "tool")
      expect_equal(body$messages[[3]]$content, "5")
      list(choices = list(list(message = list(
        role = "assistant",
        content = "The answer is 5."
      ))))
    }
  )

  convo <- hf_run_tools(convo, list(add = function(x, y) x + y))

  expect_equal(length(convo$history), 4)
  expect_equal(convo$history[[3]]$role, "tool")
  expect_equal(convo$history[[4]]$content, "The answer is 5.")
})
