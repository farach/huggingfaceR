#' Load a dataset from the Hugging Face Hub!
#'
#' Function has multiple uses - getting pre-made datasets for exploratory analysis, or to figure as means for evaluating your fine-tuned models.
#'
#' @param dataset String - the name of a Hugging Face dataset saved on the Hub. Use hf_list_models() to find a dataset.
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
  hf_import_datasets_transformers()

  if (!as_tibble & !is.null(label_name)) {
    stop("label_name must be specified with as_tibble = TRUE")
  }


  # Get this for str2int and int2str mapping later
  dataset_base <- reticulate::py$load_dataset(dataset)

  # If we just want the basic data set, unedited:
  if (is.null(split) & !as_tibble & is.null(label_name)) {
    return(reticulate::py$load_dataset(dataset, ...))

    # If we want a specific, unedited split:
  } else if (!is.null(split) & !as_tibble & is.null(label_name)) {
    return(reticulate::py$load_dataset(dataset, split = split, ...))

    # If we want all splits as a tibble without label_name specified:
  } else if (is.null(split) & as_tibble == TRUE) {
    dataset_load <- reticulate::py$load_dataset(dataset)
    split_names <- names(dataset_load)
    hf_data <- NULL

    for (name in split_names) {
      hf_dataset_loop <-
        reticulate::py$load_dataset(dataset, split = name, ...)

      hf_data_loop <- dplyr::as_tibble(hf_dataset_loop$to_pandas())

      hf_data <- dplyr::bind_rows(hf_data, hf_data_loop)
    }

    # Adding the str2int or int2str logic, and if argument is blank, return the dataset
    if (is.null(label_name)) {
      return(hf_data)
    } else if (label_name == "int2str") {
      hf_data$label_name <- dataset_base$train$features$label$int2str(as.integer(hf_data$label))
    } else if (label_name == "str2int") {
      hf_data$label_name <- dataset_base$train$features$label$str2int(as.character(hf_data$label))
    }

    return(hf_data)

    # Now add splits logic
  } else if (!is.null(split) & as_tibble == TRUE) {
    hf_data <- tibble::tibble(reticulate::py$load_dataset(dataset, split = split)$to_pandas())

    # Now add str2int and int2str logic for splits (this could be refactored to not duplicate code, but ok for now)
    if (is.null(label_name)) {
      return(hf_data)
    } else if (label_name == "int2str") {
      hf_data$label_name <- dataset_base$train$features$label$int2str(as.integer(hf_data$label))
    } else if (label_name == "str2int") {
      hf_data$label_name <- dataset_base$train$features$label$str2int(as.character(hf_data$label))
    }

    return(hf_data)
  }
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
