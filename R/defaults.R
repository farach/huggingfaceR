#' Default Models for Hugging Face Tasks
#'
#' Central registry of the default model used by each `huggingfaceR` inference
#' function. Every `hf_*` function that takes a `model` argument resolves its
#' default through this single function, so default models live in exactly one
#' place and can be audited or updated without hunting through the codebase.
#'
#' Defaults are chosen to be **beginner-friendly**: broadly known, small and
#' fast, low cost, and usable with no extra arguments — the goal is the quickest
#' path to a working first call (think of `mtcars` in base R). Power users can
#' always override any default by passing their own `model`.
#'
#' @param task Character string naming the task, or `NULL`. One of: "chat",
#'   "generate", "fill_mask", "classify", "zero_shot", "embed", "summarize",
#'   "translate", "ner", "question_answer", "table_question_answer". When `NULL`
#'   (the default), the full registry is returned as a tibble.
#'
#' @returns When `task` is supplied, a single model-ID character string. When
#'   `task` is `NULL`, a tibble with columns `task` and `model` listing every
#'   default.
#' @export
#'
#' @examples
#' # The default model for a given task
#' hf_default_model("translate")
#'
#' # The whole registry at a glance
#' hf_default_model()
hf_default_model <- function(task = NULL) {
  defaults <- c(
    chat                  = "meta-llama/Llama-3.1-8B-Instruct",
    generate              = "meta-llama/Llama-3.1-8B-Instruct",
    fill_mask             = "google-bert/bert-base-uncased",
    classify              = "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    zero_shot             = "facebook/bart-large-mnli",
    embed                 = "BAAI/bge-small-en-v1.5",
    summarize             = "facebook/bart-large-cnn",
    translate             = "Helsinki-NLP/opus-mt-en-fr",
    ner                   = "dslim/bert-base-NER",
    question_answer       = "deepset/roberta-base-squad2",
    table_question_answer = "google/tapas-base-finetuned-wtq"
  )

  if (is.null(task)) {
    return(tibble::tibble(
      task = names(defaults),
      model = unname(defaults)
    ))
  }

  if (length(task) != 1L || !is.character(task) || !task %in% names(defaults)) {
    stop(
      "`task` must be one of: ",
      paste(names(defaults), collapse = ", "),
      ".",
      call. = FALSE
    )
  }

  unname(defaults[[task]])
}
