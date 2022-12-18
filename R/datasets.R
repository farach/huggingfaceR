#' Load a dataset from the Hugging Face Hub!
#'
#' Function has multiple uses - getting pre-made datasets for exploratory analysis, or to figure as means for evaluating your fine-tuned models.
#'
#' @param dataset The name of a Hugging Face dataset saved on the Hub. Use hf_list_models() to find a dataset.
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



#Space for Jack re-writing function from scratch
new_ds_function <- function(dataset, split = NULL,
                            label_conversion = c("str2int", "int2str")){

  hf_import_datasets_transformers()

  #Set the default value of label_conversion to 'intstr' unless specified, in which case match the input
  label_conversion <- match.arg(if (missing(label_conversion)) "int2str" else label_conversion, c("str2int", "int2str"))

  #read in the dataset in Hugging Face datasets format.
  .dataset <- reticulate::py$load_dataset(dataset)
  available_splits <- paste0(names(.dataset), collapse = ";")

  # Return an error message if inputted split isn't found in the dataset's metadata
  if(!is.null(split) && !split %in% names(.dataset)){
    stop(paste0("The split you're looking for is not available for this dataset, try one of: ", available_splits))
  }

  # If no split is supplied, set splits as all the splits.
  if(is.null(split)){
    #Get all of the splits for later mapping
    splits <- names(.dataset)
  } else {
    splits <- split #Checking this works, if it does should refactor as it makes no sense like this
  }

  # Map over splits to read in dataset as pandas
  datasets <- purrr::map(splits, ~.dataset[[.x]]$to_pandas() %>%
                    tibble::as_tibble())
  names(datasets) <- splits

  unsupervised <- NULL #instantiate an object for unsupervised splits (ones in which labels will not be present)

  #If there is an unsupervised split, separate it from the datasets for mapping over int2str/str2int
  if('unsupervised' %in% splits && split == "unsupervised"){
    message("Unsupervised detected in splits, so adding labels to other splits and leaving unsupervised as is")

    unsupervised <- datasets[["unsupervised"]]
    datasets <- datasets[!stringr::str_detect(names(datasets), "unsupervised")]
    }

  #get int2str & str2int which can later be called directly on the label variable
  if(!is.null(label_conversion)){
    x <- splits[[1]]
    x <- .dataset[[x]]
    x <- x[["features"]]
    x <- x[["label"]]
    int2str <- x[["int2str"]]
    str2int <- x[["str2int"]]
  }

  if(label_conversion == "int2str"){
    label_names <- purrr::map(datasets, ~int2str(.x[["label"]]))
    datasets <- purrr::map2(.x = datasets, .y = label_names, .f = ~ .x %>% dplyr::mutate(label_name = .y))
    }
  if(label_conversion == "str2int"){
    label_ids <- purrr::map(datasets, ~str2int(.x[["label"]]))
    datasets <- purrr::map2(.x = datasets, .y = label_ids, .f = ~ .x %>% dplyr::mutate(label_id = .y))

  }

  #Check for non-df objects and then filter them out (e.g. label, text etc.)
  logicals <- purrr::map(datasets, class) %>%
    purrr::map_lgl(~ "data.frame" %in% .x)
  datasets <- datasets[logicals]

  #If user asks for unsupervised split, give them it, if not, give them datasets and if datasets is a list of length 1, unlist it and return the tibble.
  if(!is.null(unsupervised) && split == "unsupervised"){
    return(unsupervised)
  }else if(length(datasets) == 1){
    datasets <- as_tibble(datasets[[1]])
    return(datasets)}
  else{
    return(datasets)
  }

}
