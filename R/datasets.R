#' Load dataset
#' Load dataset from Huggingface
#'
#' @param dataset The name of a huggingface dataset. Use hf_list_models() to find a dataset.
#' @returns A Huggingface dataset.
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

  hf_data <-
    reticulate::py$load_dataset(dataset, split = split, ...)

  if (as_tibble == TRUE) {
    hf_data <- as_tibble(hf_data$to_pandas())
  }

  hf_data
}


# hf_data_features <- function(dataset, split = NULL, feature) {
#   hf_load_datasets_transformers()
#
#   hf_data_features <-
#     reticulate::py$load_dataset(dataset, split = split)$feature
#
#   hf_data_features
# }
#
# #############
#
# test_data <- hf_load_dataset('emotion', split = "test", as_tibble = TRUE)
#
# test_data %>%
#   mutate(
#     label_name = hf_data_features('emotion', split = 'test', feature = 'label')
#   )
