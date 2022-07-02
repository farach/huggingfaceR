#' Load dataset
#' Load dataset from Huggingface
#'
#' @param dataset The name of a huggingface dataset. Use hf_list_models() to find a dataset.
#' @param split can be either 'train', 'test', 'validation', or left NULL for all three.
#' @param as_tibble defaults to FALSE. Set to TRUE to return a tibble.
#' @param label_name options are 'int2str' or 'str2int'. If as_tibble == TRUE this argument creates a new column 'label_name' that converts the label from an integer to a string or from a string to an integer.
#' @param ... fed to load_dataset()
#' @returns A Huggingface dataset as a tibble or as it's default arrow dataset.
#' @export
#' @seealso
#' \url{https://huggingface.co/docs/datasets/index}
hf_load_dataset <- function(dataset,
                            split = NULL,
                            as_tibble = FALSE,
                            label_name = NULL,
                            ...) {
  hf_load_datasets_transformers()

  if (is.null(split) & as_tibble == TRUE) {
    dataset_load <- reticulate::py$load_dataset(dataset)
    split_names <- names(dataset_load)
    hf_data <- NULL

    for (name in split_names) {
      hf_dataset_loop <-
        reticulate::py$load_dataset(dataset, split = name, ...)

      hf_data_loop <- dplyr::as_tibble(hf_dataset_loop$to_pandas())

      if (label_name == "int2str") {
        hf_data_loop$label_name <- hf_dataset_loop$features$label$int2str(as.integer(hf_data_loop$label))
      }

      if (label_name == "str2int") {
        hf_data_loop$label_name <- hf_dataset_loop$features$label$str2int(as.character(hf_data_loop$label))
      }

      hf_data <- dplyr::bind_rows(hf_data, hf_data_loop)
    }
  } else if (!is.null(split) & as_tibble == TRUE) {
    hf_dataset <-
      reticulate::py$load_dataset(dataset, split = split, ...)

    hf_data <- dplyr::as_tibble(hf_dataset$to_pandas())

    if (label_name == "int2str") {
      hf_data$label_name <- hf_dataset$features$label$int2str(as.integer(hf_data$label))
    }

    if (label_name == "str2int") {
      hf_data$label_name <- hf_dataset$features$label$str2int(as.character(hf_data$label))
    }
  } else if (!is.null(split & as_tibble == FALSE)) {
    hf_data <- reticulate::py$load_dataset(dataset, split = split, ...)
  } else {
    hf_data <- reticulate::py$load_dataset(dataset, ...)
  }

  hf_data
}

##' examples
##' dontrun{
##' # Retrieve the 'emotion' dataset
##' emotion <- hf_load_dataset("emotion")
##' emotion
##' # Extract and visualize the training split in the emotion data
##' hf_load_dataset("emotion", as_tibble = TRUE, split = "train") %>%
##'   dplyr::add_count(label) %>%
##'   dplyr::mutate(
##'     label = forcats::fct_reorder(as.factor(label), n)
##'   ) %>%
##'   ggplot2::ggplot(ggplot2::aes(label)) +
##'   ggplot2::geom_bar()
##' }
