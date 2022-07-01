#' Load dataset
#' Load dataset from Huggingface
#'
#' @param dataset The name of a huggingface dataset. Use hf_list_models() to find a dataset.
#' @param split can be either 'train', 'test', 'validation', or left NULL for all three.
#' @param as_tibble defaults to FALSE. Set to TRUE to return a tibble.
#' @returns A Huggingface dataset as a tibble or as it's default arrow dataset.
#' @export
#' @examples
#' # Retrieve the 'emotion' dataset
#' emotion <- hf_load_dataset("emotion")
#' emotion
#' # Extract and visualize the training split in the emotion data
#' hf_load_dataset("emotion", as_tibble = TRUE, split = "train") %>%
#'   add_count(label) %>%
#'   mutate(
#'     label = fct_reorder(as.factor(label), n)
#'   ) %>%
#'   ggplot(aes(label)) +
#'   geom_bar()
#' @seealso
#' \url{https://huggingface.co/docs/datasets/index}
hf_load_dataset <- function(dataset, split = NULL, as_tibble = FALSE, ...) {
  hf_load_datasets_transformers()

  if (is.null(split) & as_tibble == TRUE) {
    dataset_load <- reticulate::py$load_dataset(dataset)
    split_names <- names(dataset_load)
    hf_data <- NULL

    for (name in split_names) {
      hf_data_loop <- reticulate::py$load_dataset(dataset, split = name, ...)
      hf_data_loop <- dplyr::as_tibble(hf_data_loop$to_pandas())
      hf_data <- dplyr::bind_rows(hf_data, hf_data_loop)
    }
  } else if (!is.null(split) & as_tibble == TRUE) {
    hf_data <-
      reticulate::py$load_dataset(dataset, split = split, ...)
    hf_data <- dplyr::as_tibble(hf_data$to_pandas())
  } else if (!is.null(split & as_tibble == FALSE)) {
    hf_data <- reticulate::py$load_dataset(dataset, split = split, ...)
  } else {
    hf_data <- reticulate::py$load_dataset(dataset, ...)
  }

  hf_data
}

