
#' Inference using a downloaded Hugging Face model or pipeline, or using the Inference API
#'
#' If a model_id is provided, the Inference API will be used to make the prediction.
#' If you wish to download a model or pipeline rather than running your predictions through the Inference API, download the model with one of the hf_load_*_model() or hf_load_pipeline() functions.
#'
#' @param model Either a downloaded model or pipeline from the Hugging Face Hub (using hf_load_pipeline()), or a model_id. Run hf_search_models(...) for model_ids.
#' @param payload The data to predict on. Use one of the hf_*_payload() functions to create.
#' @param flatten Whether to flatten the results into a data frame. Default: TRUE (flatten the results)
#' @param use_gpu API Only - Whether to use GPU for inference.
#' @param use_cache API Only - Whether to use cached inference results for previously seen inputs.
#' @param wait_for_model API Only - Whether to wait for the model to be ready instead of receiving a 503 error after a certain amount of time.
#' @param use_auth_token API Only - The token to use as HTTP bearer authorization for the Inference API. Defaults to HUGGING_FACE_HUB_TOKEN environment variable.
#' @param stop_on_error API Only - Whether to throw an error if an API error is encountered. Defaults to FALSE (do not throw error).
#'
#' @returns The results of the inference
#' @export
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/index}
hf_inference <- function(model, payload, flatten = TRUE, use_gpu = FALSE, use_cache = FALSE, wait_for_model = FALSE, use_auth_token = NULL, stop_on_error = FALSE) {

  # If model is a model_id, use Inference API
  if (is.character(model)) {

    if (is.null(use_auth_token) && Sys.getenv("HUGGING_FACE_HUB_TOKEN") != "") use_auth_token <- Sys.getenv("HUGGING_FACE_HUB_TOKEN")

    response <-
      httr2::request(glue::glue("https://router.huggingface.co/hf-inference/models/{model}")) %>%
      httr2::req_auth_bearer_token(token = use_auth_token) %>%
      httr2::req_body_json(
        payload
      ) %>%
      httr2::req_error(is_error = function(resp) stop_on_error) %>%
      httr2::req_perform()

    results <-
      response %>%
      httr2::resp_body_json(auto_unbox = TRUE)

  } else {

    # If local model object is passed in to model, perform local inference.
    if (any(stringr::str_detect(class(model), "pipelines"))) {

      # If inputs is an unnamed list of strings
      if(length(names(payload[[1]])) == 0){
        function_params <-
          append(list(payload[[1]] %>% as.character()), payload[-1] %>% unname() %>% unlist(recursive = F) %>% as.list())
      }else{
        function_params <-
          payload %>% unname() %>% unlist(recursive = F) %>% as.list()
      }

      results <-
        do.call(model, function_params)

    }else{

      if (any(stringr::str_detect(class(model), "sentence_transformers"))) {
        if(payload$task == 'sentence-similarity'){

          if(!require('lsa', quietly = T)) stop("You must install package lsa to compute sentence similarities.")

          results <-
            apply(model$encode(payload$inputs$sentences), 1, function(x) lsa::cosine(x, model$encode(payload$inputs$source_sentence) %>% as.numeric()))
        }
      } else{

        stop("model must be a downloaded Hugging Face model or pipeline, or model_id")
      }
    }
  }

  if(flatten){
    results %>%
      jsonlite::toJSON(auto_unbox = TRUE) %>%
      jsonlite::fromJSON(flatten = T)
  }else{
    results
  }
}

#' Fill Mask Task Payload
#'
#' Tries to fill in a hole with a missing word (token to be precise). That’s the base task for BERT models.
#'
#' @param string a string to be filled from, must contain the [MASK] token (check model card for exact name of the mask)
#'
#' @returns An inference payload
#' @export
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/detailed_parameters#fill-mask-task}
hf_fill_mask_payload <- function(string){

  list(inputs = string)
}


#' Summarization Task Payload
#'
#' This task is well known to summarize longer text into shorter text. Be careful, some models have a maximum length of input. That means that the summary cannot handle full books for instance.
#'
#' @param string a string to be summarized
#' @param min_length Integer to define the minimum length in tokens of the output summary. Default: NULL
#' @param max_length Integer to define the maximum length in tokens of the output summary. Default: NULL
#' @param top_k Integer to define the top tokens considered within the sample operation to create new text. Default: NULL
#' @param top_p Float to define the tokens that are within the sample operation of text generation. Add tokens in the sample for more probable to least probable until the sum of the probabilities is greater than top_p. Default: NULL
#' @param temperature Float (0.0-100.0). The temperature of the sampling operation. 1 means regular sampling, 0 means always take the highest score, 100.0 is getting closer to uniform probability. Default: 1.0
#' @param repetition_penalty Float (0.0-100.0). The more a token is used within generation the more it is penalized to not be picked in successive generation passes. Default: NULL
#' @param max_time Float (0-120.0). The amount of time in seconds that the query should take maximum. Network can cause some overhead so it will be a soft limit. Default: NULL
#'
#' @returns An inference payload
#' @export
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/detailed_parameters#summarization-task}
hf_summarization_payload <- function(string, min_length = NULL, max_length = NULL, top_k = NULL, top_p = NULL, temperature = 1.0, repetition_penalty = NULL, max_time = NULL){

  list(
    inputs = string,
    parameters = list(
      min_length = min_length,
      max_length = max_length,
      top_k = top_k,
      top_p = top_p,
      temperature = temperature,
      repetition_penalty = repetition_penalty,
      max_time = max_time
    ) %>%
      purrr::compact()
  )
}


#' Question Answering Payload
#'
#' Want to have a nice know-it-all bot that can answer any question?
#'
#' @param question a question to be answered based on the provided context
#' @param context the context to consult for answering the question
#'
#' @returns An inference payload
#' @export
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/detailed_parameters#question-answering-task}
hf_question_answering_payload <- function(question, context){

  list(
    inputs = list(
      question = question,
      context = context
    )
  )
}



#' Table Question Answering Payload
#'
#' Don’t know SQL? Don’t want to dive into a large spreadsheet? Ask questions in plain english!
#'
#' @param query The query in plain text that you want to ask the table
#' @param table A table of data represented as a dict of list where entries are headers and the lists are all the values, all lists must have the same size.
#'
#' @returns An inference payload
#' @export
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/detailed_parameters#table-question-answering-task}
hf_table_question_answering_payload <- function(query, table){

  list(
    inputs = list(
      query = query,
      table = table
    )
  )
}


#' Sentence Similarity Payload
#'
#' Calculate the semantic similarity between one text and a list of other sentences by comparing their embeddings.
#'
#' @param source_sentence The string that you wish to compare the other strings with. This can be a phrase, sentence, or longer passage, depending on the model being used.
#' @param sentences A list of strings which will be compared against the source_sentence.
#'
#' @returns An inference payload
#' @export
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/detailed_parameters#sentence-similarity-task}
hf_sentence_similarity_payload <- function(source_sentence, sentences){

  list(
    inputs = list(
      source_sentence = source_sentence,
      sentences = sentences
    ),
    task = 'sentence-similarity'
  )
}



#' Text Classification Payload
#'
#' Usually used for sentiment-analysis this will output the likelihood of classes of an input.
#'
#' @param string a string to be classified
#'
#' @returns An inference payload
#' @export
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/detailed_parameters#text-classification-task}
hf_text_classification_payload <- function(string){

  list(
    inputs = string
  )
}


#' Text Generation Payload
#'
#' Use to continue text from a prompt. This is a very generic task.
#'
#' @param string a string to be generated from
#' @param top_k (Default: None). Integer to define the top tokens considered within the sample operation to create new text.
#' @param top_p (Default: None). Float to define the tokens that are within the sample operation of text generation. Add tokens in the sample for more probable to least probable until the sum of the probabilities is greater than top_p
#' @param temperature Float (0.0-100.0). The temperature of the sampling operation. 1 means regular sampling, 0 means always take the highest score, 100.0 is getting closer to uniform probability. Default: 1.0
#' @param repetition_penalty (Default: None). Float (0.0-100.0). The more a token is used within generation the more it is penalized to not be picked in successive generation passes.
#' @param max_new_tokens (Default: None). Int (0-250). The amount of new tokens to be generated, this does not include the input length it is a estimate of the size of generated text you want. Each new tokens slows down the request, so look for balance between response times and length of text generated.
#' @param max_time (Default: None). Float (0-120.0). The amount of time in seconds that the query should take maximum. Network can cause some overhead so it will be a soft limit. Use that in combination with max_new_tokens for best results.
#' @param return_full_text (Default: True). Bool. If set to False, the return results will not contain the original query making it easier for prompting.
#' @param num_return_sequences (Default: 1). Integer. The number of proposition you want to be returned.
#' @param do_sample (Optional: True). Bool. Whether or not to use sampling, use greedy decoding otherwise.
#'
#' @returns An inference payload
#' @export
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task}
hf_text_generation_payload <- function(string, top_k = NULL, top_p = NULL, temperature = 1.0, repetition_penalty = NULL, max_new_tokens = NULL, max_time = NULL, return_full_text = TRUE, num_return_sequences = 1L, do_sample = TRUE){

  list(
    inputs = string,
    parameters =
      list(
        top_k = top_k,
        top_p = top_p,
        temperature = temperature,
        repetition_penalty = repetition_penalty,
        max_new_tokens = max_new_tokens,
        max_time = max_time,
        return_full_text = return_full_text,
        num_return_sequences = num_return_sequences,
        do_sample = do_sample
      ) %>%
      purrr::compact()
  )
}


#' Text2Text Generation Payload
#'
#' takes an input containing the sentence including the task and returns the output of the accomplished task.
#'
#' @param string a string containing a question or task and a sentence from which the answer is derived
#'
#' @returns An inference payload
#' @export
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/detailed_parameters#text2text-generation-task}
hf_text2text_generation_payload <- function(string){

  list(
    inputs = string
  )
}


#' Token Classification Payload
#'
#' Usually used for sentence parsing, either grammatical, or Named Entity Recognition (NER) to understand keywords contained within text.
#'
#' @param string a string to be classified
#' @param aggregation_strategy (Default: simple). There are several aggregation strategies. \cr
#' none: Every token gets classified without further aggregation.  \cr
#' simple: Entities are grouped according to the default schema (B-, I- tags get merged when the tag is similar).  \cr
#' first: Same as the simple strategy except words cannot end up with different tags. Words will use the tag of the first token when there is ambiguity.  \cr
#' average: Same as the simple strategy except words cannot end up with different tags. Scores are averaged across tokens and then the maximum label is applied.  \cr
#' max: Same as the simple strategy except words cannot end up with different tags. Word entity will be the token with the maximum score.  \cr
#'
#' @returns An inference payload
#' @export
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/detailed_parameters#token-classification-task}
hf_token_classification_payload <- function(string, aggregation_strategy = 'simple'){

  list(
    inputs = string,
    parameters =
      list(
        aggregation_strategy = aggregation_strategy
      )
  )
}


#' Translation Payload
#'
#' This task is well known to translate text from one language to another
#'
#' @param string a string to be translated in the original languages
#'
#' @returns An inference payload
#' @export
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/detailed_parameters#translation-task}
hf_translation_payload <- function(string){

  list(
    inputs = string
  )
}



#' Zero Shot Classification Payload
#'
#' This task is super useful to try out classification with zero code, you simply pass a sentence/paragraph and the possible labels for that sentence, and you get a result.
#'
#' @param string a string or list of strings
#' @param candidate_labels a list of strings that are potential classes for inputs. (max 10 candidate_labels, for more, simply run multiple requests, results are going to be misleading if using too many candidate_labels anyway. If you want to keep the exact same, you can simply run multi_label=True and do the scaling on your end. )
#' @param multi_label (Default: false) Boolean that is set to True if classes can overlap
#'
#' @returns An inference payload
#' @export
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/detailed_parameters#zeroshot-classification-task}
hf_zero_shot_classification_payload <- function(string, candidate_labels, multi_label = FALSE){

  list(
    inputs = string,
    parameters =
      list(
        candidate_labels = candidate_labels,
        multi_label = multi_label
      )
  )
}

