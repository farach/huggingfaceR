#' Load Sentence Model
#' Load Sentence Transformers Model from Huggingface
#'
#' @param model_id The id of a sentence-transformers model. Use hf_search_models(author = 'sentence-transformers') to find suitable models.
#' @returns A Huggingface model ready for prediction.
#' @export
#' @examples
#' library(tidyverse)
#' # Compute sentence embeddings
#' sentences <- c("Baby turtles are so cute!", "He walks as slowly as a turtle.","The lake is cold today.", "I enjoy swimming in the lake.")
#' model <- hf_load_sentence_model('paraphrase-MiniLM-L6-v2')
#' embeddings <- model$encode(sentences)
#' embeddings
#' # Get distances between sentences
#' embeddings %>% dist() %>% as.matrix() %>% as.data.frame() %>% setNames(sentences) %>% mutate(`sentence 1` = sentences) %>%
#' pivot_longer(cols = -`sentence 1`, names_to = 'sentence 2', values_to = 'distance') %>% filter(distance > 0)
#' # Cluster sentences
#' embeddings %>% t() %>% prcomp() %>% pluck('rotation') %>% as.data.frame() %>% mutate(sentence = sentences) %>%
#' ggplot(aes(PC1, PC2)) + geom_label(aes(PC1, PC2, label = sentence, vjust="inward", hjust="inward")) + theme_minimal()
#' @seealso
#' \url{https://huggingface.co/sentence-transformers}
hf_load_sentence_model <- function(model_id){

  hf_load_sentence_transformers()

  model <-
    reticulate::py$sentence_transformer(model_name_or_path = model_id)

  model
}
