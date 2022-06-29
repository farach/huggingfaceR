#' Load dataset
#' Load dataset from Huggingface
#'
#' @param dataset The name of a huggingface dataset. Use hf_list_models() to find a dataset.
#' @returns A Huggingface dataset.
#' @export
#' @examples
#' library(tidyverse)
#' # Retrieve the 'emotions' dataset
#' emotions <- hf_load_dataset('emotions')
#' emotions
#' # Extract the training split in the emotions data
#' emotions$test
#' @seealso
#' \url{https://huggingface.co/docs/datasets/index}
hf_load_dataset <- function(dataset){

  hf_load_datasets_transformers()

  hf_data <-
    reticulate::py$load_dataset(dataset)

  hf_data
}
