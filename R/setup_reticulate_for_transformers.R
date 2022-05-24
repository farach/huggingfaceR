#' Setup miniconda to work with the hugging face transformer python library
#'
#' @return NULL
#' @export
#'
#' @examples
#' setup_reticulate_for_transformers()

setup_reticulate_for_transformers <- function() {
  reticulate::conda_create(
    envname = "r-nlp",
    packages = c("PyTorch", "Tensorflow", "transformers", "sentencepiece"),
    conda = paste0(reticulate::miniconda_path(), "/condabin/conda")
  )

  reticulate::use_condaenv(condaenv = "r-nlp", required = T)
}
