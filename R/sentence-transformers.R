#' Load Sentence Model
#'
#' Load Sentence Transformers Model from Huggingface
#'
#' @param model_id The id of a sentence-transformers model. Use hf_search_models(author = 'sentence-transformers') to find suitable models.
#' @returns A Huggingface model ready for prediction.
#' @export
#' @seealso
#' \url{https://huggingface.co/sbentence-transformers}
hf_load_sentence_model <- function(model_id) {
  hf_load_sentence_transformers()

  model <-
    reticulate::py$sentence_transformer(model_name_or_path = model_id)

  model
}



##' examples
##' dontrun{
##' # Compute sentence embeddings
##' sentences_one <- c("Baby turtles are so cute!", "He walks as slowly as a turtle.")
##' sentences_two <- c("The lake is cold today.", "I enjoy swimming in the lake.")
##' sentences <- c(sentences_one, sentences_two)
##' model <- hf_load_sentence_model('paraphrase-MiniLM-L6-v2')
##' embeddings <- model$encode(sentences)
##' distances <- embeddings %>% dist() %>% as.matrix() %>% as.data.frame() %>% setNames(sentences)
##' distances <- distances %>% 
##'   dplyr::mutate(`sentence 1` = sentences) %>% 
##'   tidyr::pivot_longer(cols = -`sentence 1`, names_to = 'sentence 2', values_to = 'distance')
##' distances <- distances %>% dplyr::filter(distance > 0)
##' distances
##' # Cluster sentences
##' embeddings_pca <- embeddings %>% t() %>% prcomp() %>% purrr::pluck('rotation') %>% as.data.frame() %>% dplyr::mutate(sentence = sentences)
##' embeddings_pca %>% 
##'   ggplot2::ggplot(ggplot2::aes(PC1, PC2)) +
##'   ggplot2::geom_point() +
##'   ggplot2::geom_text(ggplot2::aes(label = sentence), vjust="inward", hjust="inward") + 
##'   ggplot2::theme_minimal()
##' }
