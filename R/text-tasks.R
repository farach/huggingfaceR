#' Summarize Text
#'
#' Condense longer text into a shorter summary using a summarization model via
#' the Hugging Face Inference Providers API. Accepts a character vector and
#' returns one row per input, composing naturally with dplyr pipelines.
#'
#' @param text Character vector of text(s) to summarize.
#' @param model Character string. Model ID from the Hugging Face Hub. Append
#'   `":provider"` to select an inference provider. Default:
#'   "facebook/bart-large-cnn".
#' @param min_length Integer or NULL. Minimum length of the summary in tokens.
#'   Default: NULL (model default).
#' @param max_length Integer or NULL. Maximum length of the summary in tokens.
#'   Default: NULL (model default).
#' @param token Character string or NULL. API token for authentication.
#' @param endpoint_url Character string or NULL. A custom Inference Endpoint URL.
#'   When provided, requests are sent to this URL instead of the public
#'   Inference API.
#' @param ... Additional arguments (currently unused).
#'
#' @returns A tibble with columns: text, summary.
#' @export
#' @seealso \url{https://huggingface.co/docs/inference-providers/tasks/summarization}
#'
#' @examples
#' \dontrun{
#' hf_summarize("Long article text goes here ...", max_length = 60)
#'
#' library(dplyr)
#' articles |>
#'   mutate(tldr = hf_summarize(body)$summary)
#' }
hf_summarize <- function(text,
                         model = "facebook/bart-large-cnn",
                         min_length = NULL,
                         max_length = NULL,
                         token = NULL,
                         endpoint_url = NULL,
                         ...) {

  if (length(text) == 0) {
    return(tibble::tibble(text = character(), summary = character()))
  }

  purrr::map_dfr(text, function(single_text) {
    if (is.na(single_text)) {
      return(tibble::tibble(text = single_text, summary = NA_character_))
    }

    result <- hf_task_request(
      model = model,
      inputs = single_text,
      parameters = list(min_length = min_length, max_length = max_length),
      token = token,
      endpoint_url = endpoint_url
    )

    tibble::tibble(
      text = single_text,
      summary = hf_first_field(result, "summary_text")
    )
  })
}


#' Translate Text
#'
#' Translate text from one language to another using a translation model via the
#' Hugging Face Inference Providers API.
#'
#' The default model, `Helsinki-NLP/opus-mt-en-fr`, translates English to French
#' and is chosen for easy onboarding: it is small, fast, broadly known, and
#' encodes the translation direction in the model ID, so `hf_translate("Hello")`
#' works with no extra arguments. To translate a different language pair, swap in
#' another Helsinki-NLP `opus-mt-*` model (for example
#' `"Helsinki-NLP/opus-mt-en-es"` for English to Spanish).
#'
#' Translation models vary in how they expect languages to be specified.
#' Language-pair models such as the Helsinki-NLP `opus-mt-*` family encode the
#' direction in the model ID and ignore `source`/`target`. Multilingual models
#' such as NLLB (`facebook/nllb-200-distilled-600M`) instead require `source` and
#' `target` to be set to FLORES-200 codes (for example "eng_Latn", "fra_Latn").
#'
#' @param text Character vector of text(s) to translate.
#' @param model Character string. Model ID from the Hugging Face Hub. Append
#'   `":provider"` to select an inference provider. Default:
#'   "Helsinki-NLP/opus-mt-en-fr" (English to French).
#' @param source Character string or NULL. Source language code (model-specific;
#'   ignored by `opus-mt-*` language-pair models).
#' @param target Character string or NULL. Target language code (model-specific;
#'   ignored by `opus-mt-*` language-pair models).
#' @param token Character string or NULL. API token for authentication.
#' @param endpoint_url Character string or NULL. A custom Inference Endpoint URL.
#' @param ... Additional arguments (currently unused).
#'
#' @returns A tibble with columns: text, translation.
#' @export
#' @seealso \url{https://huggingface.co/docs/inference-providers/tasks/translation}
#'
#' @examples
#' \dontrun{
#' # Simplest call: English to French with the default model
#' hf_translate("Hello, how are you?")
#'
#' # A different language pair (English to Spanish)
#' hf_translate("Hello, how are you?", model = "Helsinki-NLP/opus-mt-en-es")
#'
#' # Multilingual model (FLORES-200 codes)
#' hf_translate(
#'   "Hello, how are you?",
#'   model = "facebook/nllb-200-distilled-600M",
#'   source = "eng_Latn",
#'   target = "fra_Latn"
#' )
#' }
hf_translate <- function(text,
                         model = "Helsinki-NLP/opus-mt-en-fr",
                         source = NULL,
                         target = NULL,
                         token = NULL,
                         endpoint_url = NULL,
                         ...) {

  if (length(text) == 0) {
    return(tibble::tibble(text = character(), translation = character()))
  }

  params <- list(src_lang = source, tgt_lang = target)

  purrr::map_dfr(text, function(single_text) {
    if (is.na(single_text)) {
      return(tibble::tibble(text = single_text, translation = NA_character_))
    }

    result <- hf_task_request(
      model = model,
      inputs = single_text,
      parameters = params,
      token = token,
      endpoint_url = endpoint_url
    )

    tibble::tibble(
      text = single_text,
      translation = hf_first_field(result, "translation_text")
    )
  })
}


#' Named Entity Recognition (Token Classification)
#'
#' Extract named entities (people, organizations, locations, etc.) from text
#' using a token-classification model via the Hugging Face Inference Providers
#' API. Returns one row per detected entity, with character offsets that let you
#' highlight or join back to the source text. Inputs that produce no entities
#' (and `NA` inputs) yield a single row with `NA` entity fields so every input is
#' represented.
#'
#' @param text Character vector of text(s) to analyze.
#' @param model Character string. Model ID from the Hugging Face Hub. Append
#'   `":provider"` to select an inference provider. Default: "dslim/bert-base-NER".
#' @param aggregation_strategy Character string. How sub-word tokens are grouped
#'   into entities: one of "none", "simple", "first", "average", "max".
#'   Default: "simple".
#' @param token Character string or NULL. API token for authentication.
#' @param endpoint_url Character string or NULL. A custom Inference Endpoint URL.
#' @param ... Additional arguments (currently unused).
#'
#' @returns A tibble with columns: text, word, entity_group, score, start, end.
#' @export
#' @seealso \url{https://huggingface.co/docs/inference-providers/tasks/token-classification}
#'
#' @examples
#' \dontrun{
#' hf_ner("Barack Obama was born in Hawaii.")
#'
#' # One row per entity, ready to count or join
#' library(dplyr)
#' hf_ner(headlines) |>
#'   filter(!is.na(word)) |>
#'   count(entity_group, sort = TRUE)
#' }
hf_ner <- function(text,
                   model = "dslim/bert-base-NER",
                   aggregation_strategy = "simple",
                   token = NULL,
                   endpoint_url = NULL,
                   ...) {

  empty <- tibble::tibble(
    text = character(),
    word = character(),
    entity_group = character(),
    score = numeric(),
    start = integer(),
    end = integer()
  )
  if (length(text) == 0) {
    return(empty)
  }

  purrr::map_dfr(text, function(single_text) {
    na_row <- tibble::tibble(
      text = single_text,
      word = NA_character_,
      entity_group = NA_character_,
      score = NA_real_,
      start = NA_integer_,
      end = NA_integer_
    )

    if (is.na(single_text)) {
      return(na_row)
    }

    result <- hf_task_request(
      model = model,
      inputs = single_text,
      parameters = list(aggregation_strategy = aggregation_strategy),
      token = token,
      endpoint_url = endpoint_url
    )

    if (!is.list(result) || length(result) == 0) {
      return(na_row)
    }

    purrr::map_dfr(result, function(ent) {
      tibble::tibble(
        text = single_text,
        word = ent$word %||% NA_character_,
        entity_group = ent$entity_group %||% ent$entity %||% NA_character_,
        score = ent$score %||% NA_real_,
        start = as_int_or_na(ent$start),
        end = as_int_or_na(ent$end)
      )
    })
  })
}


#' Extractive Question Answering
#'
#' Answer a question from a supplied context passage using an extractive
#' question-answering model via the Hugging Face Inference Providers API. The
#' answer is a span extracted verbatim from the context. `question` and `context`
#' are recycled to a common length (each may be length 1).
#'
#' @param question Character vector of question(s).
#' @param context Character vector of context passage(s) to answer from.
#' @param model Character string. Model ID from the Hugging Face Hub. Append
#'   `":provider"` to select an inference provider. Default:
#'   "deepset/roberta-base-squad2".
#' @param token Character string or NULL. API token for authentication.
#' @param endpoint_url Character string or NULL. A custom Inference Endpoint URL.
#' @param ... Additional arguments (currently unused).
#'
#' @returns A tibble with columns: question, answer, score, start, end.
#' @export
#' @seealso \url{https://huggingface.co/docs/inference-providers/tasks/question-answering}
#'
#' @examples
#' \dontrun{
#' hf_question_answer(
#'   question = "Where was Obama born?",
#'   context = "Barack Obama was born in Honolulu, Hawaii."
#' )
#'
#' # One context, several questions
#' hf_question_answer(
#'   question = c("Who?", "Where?"),
#'   context = "Ada Lovelace worked in London."
#' )
#' }
hf_question_answer <- function(question,
                               context,
                               model = "deepset/roberta-base-squad2",
                               token = NULL,
                               endpoint_url = NULL,
                               ...) {

  n <- max(length(question), length(context))
  if (n == 0) {
    return(tibble::tibble(
      question = character(),
      answer = character(),
      score = numeric(),
      start = integer(),
      end = integer()
    ))
  }

  if (length(question) == 1) question <- rep(question, n)
  if (length(context) == 1) context <- rep(context, n)
  if (length(question) != length(context)) {
    stop("`question` and `context` must have the same length (or length 1).",
         call. = FALSE)
  }

  purrr::map_dfr(seq_len(n), function(i) {
    q <- question[i]
    ctx <- context[i]

    if (is.na(q) || is.na(ctx)) {
      return(tibble::tibble(
        question = q,
        answer = NA_character_,
        score = NA_real_,
        start = NA_integer_,
        end = NA_integer_
      ))
    }

    result <- hf_task_request(
      model = model,
      inputs = list(question = q, context = ctx),
      token = token,
      endpoint_url = endpoint_url
    )

    # QA usually returns a single object; some models return a list of spans.
    obj <- if (!is.null(result$answer)) {
      result
    } else if (is.list(result) && length(result) > 0) {
      result[[1]]
    } else {
      list()
    }

    tibble::tibble(
      question = q,
      answer = obj$answer %||% NA_character_,
      score = obj$score %||% NA_real_,
      start = as_int_or_na(obj$start),
      end = as_int_or_na(obj$end)
    )
  })
}


#' Table Question Answering
#'
#' Ask a question in plain language about a data frame, using a table
#' question-answering model (such as TAPAS) via the Hugging Face Inference
#' Providers API. The data frame is converted to the string-cell format the API
#' expects; all values are coerced to character.
#'
#' @param query Character vector of question(s) to ask about the table.
#' @param table A data frame to query.
#' @param model Character string. Model ID from the Hugging Face Hub. Append
#'   `":provider"` to select an inference provider. Default:
#'   "google/tapas-base-finetuned-wtq".
#' @param token Character string or NULL. API token for authentication.
#' @param endpoint_url Character string or NULL. A custom Inference Endpoint URL.
#' @param ... Additional arguments (currently unused).
#'
#' @returns A tibble with columns: query, answer, aggregator, cells (a
#'   list-column of the source cells the answer was drawn from).
#' @export
#' @seealso \url{https://huggingface.co/docs/inference-providers/tasks/table-question-answering}
#'
#' @examples
#' \dontrun{
#' sales <- data.frame(
#'   product = c("Widgets", "Gadgets", "Gizmos"),
#'   revenue = c(120, 80, 50)
#' )
#' hf_table_question_answer("Which product had the highest revenue?", sales)
#' hf_table_question_answer("What is the total revenue?", sales)
#' }
hf_table_question_answer <- function(query,
                                     table,
                                     model = "google/tapas-base-finetuned-wtq",
                                     token = NULL,
                                     endpoint_url = NULL,
                                     ...) {

  if (!is.data.frame(table)) {
    stop("`table` must be a data frame.", call. = FALSE)
  }

  if (length(query) == 0) {
    return(tibble::tibble(
      query = character(),
      answer = character(),
      aggregator = character(),
      cells = list()
    ))
  }

  # The API expects a dict of column -> list of string cell values. Wrapping
  # each column in a list keeps it a JSON array even when there is one row.
  table_payload <- lapply(table, function(col) as.list(as.character(col)))

  purrr::map_dfr(query, function(q) {
    if (is.na(q)) {
      return(tibble::tibble(
        query = q,
        answer = NA_character_,
        aggregator = NA_character_,
        cells = list(NULL)
      ))
    }

    result <- hf_task_request(
      model = model,
      inputs = list(query = q, table = table_payload),
      token = token,
      endpoint_url = endpoint_url
    )

    tibble::tibble(
      query = q,
      answer = result$answer %||% NA_character_,
      aggregator = result$aggregator %||% NA_character_,
      cells = list(unlist(result$cells) %||% character(0))
    )
  })
}


#' Extract a Field from the First Element of a Task Response
#'
#' Inference task responses are sometimes a list of result objects and sometimes
#' a single object. This helper returns `field` from the first result in either
#' shape.
#'
#' @param result Parsed JSON response (a list).
#' @param field Character string. Name of the field to extract.
#'
#' @returns The field value, or `NA_character_` if absent.
#' @keywords internal
hf_first_field <- function(result, field) {
  if (is.list(result) && !is.null(result[[field]])) {
    return(result[[field]])
  }
  if (is.list(result) && length(result) > 0 && is.list(result[[1]])) {
    return(result[[1]][[field]] %||% NA_character_)
  }
  NA_character_
}


#' Coerce a Value to Integer or NA
#'
#' @param x A scalar value or NULL.
#'
#' @returns An integer scalar, or `NA_integer_` when `x` is NULL/NA.
#' @keywords internal
as_int_or_na <- function(x) {
  if (is.null(x) || length(x) == 0 || is.na(x)) NA_integer_ else as.integer(x)
}
