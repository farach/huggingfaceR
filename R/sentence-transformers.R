#' Load Sentence Model
#'
#' Load Sentence Transformers Model from Huggingface
#'
#' @param model_id The id of a sentence-transformers model. Use hf_search_models(author = 'sentence-transformers') to find suitable models.
#' @returns A Huggingface model ready for prediction.
#' @export
#' @seealso
#' \url{https://huggingface.co/sentence-transformers}
hf_load_sentence_model <- function(model_id) {
  hf_load_sentence_transformers()

  model <-
    reticulate::py$sentence_transformer(model_name_or_path = model_id)

  model
}



##' examples
##' dontrun{
##' # Compute sentence embeddings
##' sentences <- c("Baby turtles are so cute!", "He walks as slowly as a turtle.")
##' sentences_two <- c("The lake is cold today.", "I enjoy swimming in the lake.")
##' sentences <- c(sentences, sentences_two)
##' model <- hf_load_sentence_model('paraphrase-MiniLM-L6-v2')
##' embeddings <- model$encode(sentences)
##' embeddings %>% dist() %>% as.matrix() %>% as.data.frame() %>% setNames(sentences)
##' embddings <- embeddings %>% dplyr::mutate(`sentence 1` = sentences) %>% tidyr::pivot_longer(cols = -`sentence 1`, names_to = 'sentence 2', values_to = 'distance')
##' embeddings <- embeddings %>% filter(distance > 0)
##' # Cluster sentences
##' embeddings <- embeddings %>% t() %>% prcomp() %>% purrr::pluck('rotation') %>% as.data.frame() %>% dplyr::mutate(sentence = sentences)
##' plot <- embedidings %>% ggplot2::ggplot(aes(PC1, PC2)) + ggplot2::geom_label(ggplot2::aes(PC1, PC2, label = sentence, vjust="inward", hjust="inward")) + ggplot2::theme_minimal()
##' }
