#' List Authors
#'
#' List Model Authors
#' @param pattern A search term or regular expression. Defaults to NULL (return all results).
#' @export
#'
#' @examples
#' hf_list_authors(pattern = "^sam")
#' @seealso
#' \url{https://huggingface.co/docs/hub/searching-the-hub}
hf_list_authors <- function(pattern = NULL) {
  hf_list_attribute_options("author", pattern)
}

#' List Datasets
#'
#' List Model Datasets
#' @param pattern A search term or regular expression. Defaults to NULL (return all results).
#' @export
#'
#' @examples
#' hf_list_datasets("imdb")
#' @seealso
#' \url{https://huggingface.co/docs/hub/searching-the-hub}
hf_list_datasets <- function(pattern = NULL) {
  hf_list_attribute_options("dataset", pattern)
}

#' List Languages
#'
#' List Model Languages
#' @param pattern A search term or regular expression. Defaults to NULL (return all results).
#' @export
#'
#' @examples
#' hf_list_languages("es")
#' @seealso
#' \url{https://huggingface.co/docs/hub/searching-the-hub}
hf_list_languages <- function(pattern = NULL) {
  hf_list_attribute_options("language", pattern)
}

#' List Libraries
#'
#' List Model Libraries
#' @param pattern A search term or regular expression. Defaults to NULL (return all results).
#' @export
#'
#' @examples
#' hf_list_libraries("pytorch|tensorflow")
#' @seealso
#' \url{https://huggingface.co/docs/hub/searching-the-hub}
hf_list_libraries <- function(pattern = NULL) {
  hf_list_attribute_options("library", pattern)
}

#' List Licenses
#'
#' List Model Licenses
#' @param pattern A search term or regular expression. Defaults to NULL (return all results).
#' @export
#'
#' @examples
#' hf_list_licenses("mit")
#' @seealso
#' \url{https://huggingface.co/docs/hub/searching-the-hub}
hf_list_licenses <- function(pattern = NULL) {
  hf_list_attribute_options("license", pattern)
}

#' List Models
#'
#' List Model Names
#' @param pattern A search term or regular expression. Defaults to NULL (return all results).
#' @export
#'
#' @examples
#' hf_list_models("bert-base-cased")
#' @seealso
#' \url{https://huggingface.co/docs/hub/searching-the-hub}
hf_list_models <- function(pattern = NULL) {
  tibble::tibble(model = hf_list_attribute_options("model_name", pattern))
}

#' List Tasks
#'
#' List Model Tasks
#' @param pattern A search term or regular expression. Defaults to NULL (return all results).
#' @export
#'
#' @examples
#' hf_list_tasks("^image.*tion")
#' @seealso
#' \url{https://huggingface.co/docs/hub/searching-the-hub}
hf_list_tasks <- function(pattern = NULL) {
  hf_list_attribute_options("pipeline_tag", pattern)
}

#' Search Models
#'
#' Search Huggingface Models
#' @param author Filter by model author. Run hf_list_authors() for options.
#' @param language Filter by the languages the model accommodates. Run hf_list_languages() for options.
#' @param library Filter by the deep learning libraries which work with the model. Run hf_list_libraries() for options.
#' @param name Filter by model names. Run hf_list_models() for options.
#' @param tags Filter by model tags.
#' @param task Filter by tasks the model can accomplish. Run hf_list_tasks() for options.
#' @param dataset Filter by the datasets the model was trained on. hf_list_datasets()
#' @export
#'
#' @examples
#' hf_search_models(library = "pytorch", dataset = "mnli")
#' hf_search_models(author = "facebook", name = "bart")
#' @seealso
#' \url{https://huggingface.co/docs/hub/searching-the-hub}
hf_search_models <- function(author = NULL, language = NULL, library = NULL, name = NULL, tags = NULL, task = NULL, dataset = NULL) {
  stopifnot(hf_load_model_filter())

  model_filter <-
    reticulate::py$ModelFilter(author = author, language = language, library = library, model_name = name, tags = tags, task = task, trained_dataset = dataset)

  stopifnot(hf_load_api())

  reticulate::py$hf_api$list_models(filter = model_filter)
}
