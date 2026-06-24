#' List Authors
#'
#' List Model Authors
#' @param pattern A search term or regular expression. Defaults to NULL (return all results).
#' @returns A character vector of matching model authors.
#' @export
#'
#' @examples
#' \dontrun{
#' hf_list_authors(pattern = "^sam")
#' }
#' @seealso
#' \url{https://huggingface.co/docs/hub/searching-the-hub}
hf_list_authors <- function(pattern = NULL) {
  hf_list_attribute_options("author", pattern)
}

#' List Datasets
#'
#' List Model Datasets
#' @param pattern A search term or regular expression. Defaults to NULL (return all results).
#' @returns A character vector of matching dataset names.
#' @export
#'
#' @examples
#' \dontrun{
#' hf_list_datasets("imdb")
#' }
#' @seealso
#' \url{https://huggingface.co/docs/hub/searching-the-hub}
hf_list_datasets <- function(pattern = NULL) {
  hf_list_attribute_options("dataset", pattern)
}

#' List Languages
#'
#' List Model Languages
#' @param pattern A search term or regular expression. Defaults to NULL (return all results).
#' @returns A character vector of matching language tags.
#' @export
#'
#' @examples
#' \dontrun{
#' hf_list_languages("es")
#' }
#' @seealso
#' \url{https://huggingface.co/docs/hub/searching-the-hub}
hf_list_languages <- function(pattern = NULL) {
  hf_list_attribute_options("language", pattern)
}

#' List Libraries
#'
#' List Model Libraries
#' @param pattern A search term or regular expression. Defaults to NULL (return all results).
#' @returns A character vector of matching library names.
#' @export
#'
#' @examples
#' \dontrun{
#' hf_list_libraries("pytorch|tensorflow")
#' }
#' @seealso
#' \url{https://huggingface.co/docs/hub/searching-the-hub}
hf_list_libraries <- function(pattern = NULL) {
  hf_list_attribute_options("library", pattern)
}

#' List Licenses
#'
#' List Model Licenses
#' @param pattern A search term or regular expression. Defaults to NULL (return all results).
#' @returns A character vector of matching license identifiers.
#' @export
#'
#' @examples
#' \dontrun{
#' hf_list_licenses("mit")
#' }
#' @seealso
#' \url{https://huggingface.co/docs/hub/searching-the-hub}
hf_list_licenses <- function(pattern = NULL) {
  hf_list_attribute_options("license", pattern)
}

#' List Models
#'
#' List Model Names
#' @param pattern A search term or regular expression. Defaults to NULL (return all results).
#' @returns A tibble with a `model` column containing matching model IDs.
#' @export
#'
#' @examples
#' \dontrun{
#' hf_list_models("bert-base-cased")
#' }
#' @seealso
#' \url{https://huggingface.co/docs/hub/searching-the-hub}
hf_list_models <- function(pattern = NULL) {
  tibble::tibble(model = hf_list_attribute_options("model_name", pattern))
}
