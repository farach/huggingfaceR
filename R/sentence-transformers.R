#' Load a Sentence Transformers model to extract sentence/document embeddings
#'
#'
#' @param ... Sent to model call, could include arguments such as use_auth_token = auth_token, device = device etc.
#' @param model_id The id of a sentence-transformers model. Use hf_search_models(author = 'sentence-transformers') to find suitable models.
#' @returns A Huggingface model ready for prediction.
#' @export
#'
#' @examples
#' \dontrun{
#' # Compute sentence embeddings
#' sentences <- c("Baby turtles are so cute!", "He walks as slowly as a turtle.")
#' sentences_two <- c("The lake is cold today.", "I enjoy swimming in the lake.")
#' sentences <- c(sentences, sentences_two)
#' model <- hf_load_sentence_model('paraphrase-MiniLM-L6-v2')
#' embeddings <- model$encode(sentences)
#' embeddings %>% dist() %>% as.matrix() %>% as.data.frame() %>% setNames(sentences)
#' embddings <- embeddings %>% dplyr::mutate(`sentence 1` = sentences) %>%
#' tidyr::pivot_longer(cols = -`sentence 1`, names_to = 'sentence 2', values_to = 'distance')
#' embeddings <- embeddings %>% filter(distance > 0)
#' # Cluster sentences
#' embeddings <- embeddings %>%
#' t() %>% prcomp() %>%
#' purrr::pluck('rotation') %>%
#' as.data.frame() %>%
#' dplyr::mutate(sentence = sentences)
#' plot <- embedidings %>% ggplot2::ggplot(aes(PC1, PC2)) +
#' ggplot2::geom_label(ggplot2::aes(PC1, PC2, label = sentence, vjust="inward", hjust="inward")) +
#' ggplot2::theme_minimal()
#' }
#' @seealso
#' \url{https://huggingface.co/sentence-transformers}
hf_load_sentence_model <- function(model_id, ...) {
  hf_import_sentence_transformers()

  model <-
    reticulate::py$sentence_transformer(model_name_or_path = model_id, ...)

  model
}

#' Use a Sentence Transformers pipeline to extract document(s)/sentence(s) embedding(s)
#'
#' @param model Model object you loaded with `hf_load_sentence_model()`
#' @param text The text, or texts, you wish to embed/encode.
#' @param batch_size How many texts to embed at once.
#' @param show_progress_bar Whether to print a progress bar in the console or not.
#' @param tidy Whether to tidy the output into a tibble or not.
#' @param ... other args sent to the model's encode method, e.g. device = device
#'
#' @return n-dimensional embeddings for every input `text`
#' @export
#' @examples
#' \dontrun{
#' text <- c("There are things we do know, things we don't know, and then there is quantum mechanics.")
#' sentence_mod <- hf_load_sentence_model("paraphrase-MiniLM-L6-v2")
#' embeddings <- hf_sentence_encode(model = sentence_mod, text, show_progress_bar = TRUE)
#' }
hf_sentence_encode <- function(model, text, batch_size = 64L, show_progress_bar = TRUE, tidy = TRUE, ...){

  embedding <-model$encode(text, batch_size = batch_size, show_progress_bar = TRUE, ...)

  if(tidy){
    embedding <- embedding %>%
      as.data.frame() %>%
      tibble::as_tibble()
  }
  return(embedding)
}
