
hf_make_api_request <- function(model, payload, use_auth_token = NULL, stop_on_error = FALSE){

  req <- httr2::request(glue::glue("https://api-inference.huggingface.co/models/{model}"))

  if (is.null(use_auth_token) && Sys.getenv("HUGGING_FACE_HUB_TOKEN") != "") use_auth_token <- Sys.getenv("HUGGING_FACE_HUB_TOKEN")
  if(!is.null(use_auth_token)) req <- req %>% httr2::req_auth_bearer_token(token = use_auth_token)

  req %>%
    httr2::req_body_json(
      payload
    ) %>%
    httr2::req_error(is_error = function(resp) stop_on_error) %>%
    httr2::req_perform() %>%
    httr2::resp_body_json(auto_unbox = TRUE)
}

################# Fill Mask ##################

#' Load a Fill-in-the-Blank Model
#'
#' Tries to fill in a hole with a missing word (token to be precise). That’s the base task for BERT models.
#'
#' @param model_id A model_id. Run hf_search_models(...) for model_ids. Defaults to 'bert-base-uncased'.
#' @param use_api Whether to use the Inference API to run the model (TRUE) or download and run the model locally (FALSE). Defaults to FALSE
#'
#' @returns A fill mask object
#' @examples
#' \dontrun{
#' # Load the default model and use local inference
#' fm <- hf_ez_fill_mask()
#' fm$infer(string = "The answer to the universe is [MASK].")
#'
#' # Load a specific model and use the api for inference. Note the mask is different for different models.
#' fill_mask <- hf_ez_fill_mask(model_id = 'xlm-roberta-base', use_api = TRUE)
#' fill_mask$infer(string = "The answer to the universe is <MASK>.")
#' }
#' @export
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/detailed_parameters#fill-mask-task}
hf_ez_fill_mask <- function(model_id = 'bert-base-uncased', use_api = FALSE){

  task <- 'fill-mask'

  if(use_api){
    infer_function <- function() {args <- as.list(environment()); do.call(hf_ez_fill_mask_api_inference, args %>% append(list(model = model_id)))}

    formals(infer_function) <- formals(hf_ez_fill_mask_api_inference)

    list(
      model_id = model_id,
      task = task,
      infer = infer_function
    )

  }else{
    pipeline <- hf_load_pipeline(model_id = model_id, task = task)
    infer_function <- function() {args <- as.list(environment()); do.call(hf_ez_fill_mask_local_inference, args %>% append(list(model = pipeline)))}

    formals(infer_function) <- formals(hf_ez_fill_mask_local_inference)

    list(
      model_id = model_id,
      task = task,
      infer = infer_function,
      .raw = pipeline
    )
  }
}


#' Fill Mask Local Inference
#'
#' @param string a string to be filled from, must contain the [MASK] token (check model card for exact name of the mask)
#' @param tidy Whether to tidy the results into a tibble. Default: TRUE (tidy the results)
#'
#' @returns The results of the inference
#' @seealso
#' \url{https://huggingface.co/docs/transformers/main/en/pipeline_tutorial}
hf_ez_fill_mask_local_inference <- function(string, tidy = TRUE, ...) {

  dots <- list(...)

  model <- dots$model

  payload <- list(inputs = string)

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

  # Create an unnamed list by default.
  if(!is.null(names(results))){
    results <- list(results)
  }

  if(tidy){
    results %>%
      jsonlite::toJSON(auto_unbox = TRUE) %>%
      jsonlite::fromJSON(flatten = TRUE)
  }else{
    results
  }
}


#' Fill Mask API Inference
#'
#' @param string a string to be filled from, must contain the [MASK] token (check model card for exact name of the mask)
#' @param tidy Whether to tidy the results into a tibble. Default: TRUE (tidy the results)
#' @param use_gpu Whether to use GPU for inference.
#' @param use_cache Whether to use cached inference results for previously seen inputs.
#' @param wait_for_model Whether to wait for the model to be ready instead of receiving a 503 error after a certain amount of time.
#' @param use_auth_token The token to use as HTTP bearer authorization for the Inference API. Defaults to HUGGING_FACE_HUB_TOKEN environment variable.
#' @param stop_on_error Whether to throw an error if an API error is encountered. Defaults to FALSE (do not throw error).
#'
#' @returns The results of the inference
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/index}
hf_ez_fill_mask_api_inference <- function(string, tidy = TRUE, use_gpu = FALSE, use_cache = FALSE, wait_for_model = FALSE, use_auth_token = NULL, stop_on_error = FALSE, ...) {

  dots <- list(...)

  model <- dots$model

  payload <- list(inputs = string,
                  options = environment() %>% as.list() %>% purrr::list_modify(string = NULL, use_auth_token = NULL, model = NULL) %>% purrr::compact())

  results <-
    hf_make_api_request(model = model, payload = payload, use_auth_token = use_auth_token, stop_on_error = stop_on_error)

  # Create an unnamed list by default.
  if(!is.null(names(results))){
    results <- list(results)
  }

  if(tidy){
    results %>%
      jsonlite::toJSON(auto_unbox = TRUE) %>%
      jsonlite::fromJSON(flatten = TRUE)
  }else{
    results
  }
}



################# Summarization ##################

#' EZ Summarization
#'
#' This task is well known to summarize longer text into shorter text. Be careful, some models have a maximum length of input. That means that the summary cannot handle full books for instance. Be careful when choosing your model.
#'
#' @param model_id A model_id. Run hf_search_models(...) for model_ids. Defaults to 'facebook/bart-large-cnn'.
#' @param use_api Whether to use the Inference API to run the model (TRUE) or \cr download and run the model locally (FALSE). Defaults to FALSE
#'
#' @returns A summarization object
#' @examples
#' \dontrun{
#' # Load the default model and use local inference
#' summarizer <- hf_ez_fill_summarization()
#' summarizer$infer(string = "The tower is 324 metres (1,063 ft) tall, about the same \cr
#' height as an 81-storey building, and the tallest structure in Paris. Its base \cr
#' is square, measuring 125 metres (410 ft) on each side. During its construction, \cr
#' the Eiffel Tower surpassed the Washington Monument to become the tallest man-made \cr
#' structure in the world, a title it held for 41 years until the Chrysler Building \cr
#' in New York City was finished in 1930. It was the first structure to reach a height \cr
#' of 300 metres. Due to the addition of a broadcasting aerial at the top of the \cr
#' tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). \cr
#' Excluding transmitters, the Eiffel Tower is the second tallest free-standing \cr
#' structure in France after the Millau Viaduct.",
#' min_length = 10, max_length = 100)
#'
#' # Load a specific model and use the api for inference. Note the mask is different for different models.
#' summarizer <- hf_ez_summarization(model_id = 'xlm-roberta-base', use_api = TRUE)
#' summarizer$infer(string = "huggingface is the <mask>!")
#' }
#' @export
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/detailed_parameters#summarization-task}
hf_ez_summarization <- function(model_id = 'facebook/bart-large-cnn', use_api = FALSE){

  task <- 'summarization'

  if(use_api){
    infer_function <- function() {args <- as.list(environment()); do.call(hf_ez_summarization_api_inference, args %>% append(list(model = model_id)))}

    formals(infer_function) <- formals(hf_ez_summarization_api_inference)

    list(
      model_id = model_id,
      task = task,
      infer = infer_function
    )

  }else{
    pipeline <- hf_load_pipeline(model_id = model_id, task = task)
    infer_function <- function() {args <- as.list(environment()); do.call(hf_ez_summarization_local_inference, args %>% append(list(model = pipeline)))}

    formals(infer_function) <- formals(hf_ez_summarization_local_inference)

    list(
      model_id = model_id,
      task = task,
      infer = infer_function,
      .raw = pipeline
    )
  }
}


#' Summarization Local Inference
#'
#' @param string a string to be summarized
#' @param min_length Integer to define the minimum length in tokens of the output summary. Default: NULL
#' @param max_length Integer to define the maximum length in tokens of the output summary. Default: NULL
#' @param top_k Integer to define the top tokens considered within the sample operation to create new text. Default: NULL
#' @param top_p Float to define the tokens that are within the sample operation of text generation. Add tokens in the sample for more probable to least probable until the sum of the probabilities is greater than top_p. Default: NULL
#' @param temperature Float (0.0-100.0). The temperature of the sampling operation. 1 means regular sampling, 0 means always take the highest score, 100.0 is getting closer to uniform probability. Default: 1.0
#' @param repetition_penalty Float (0.0-100.0). The more a token is used within generation the more it is penalized to not be picked in successive generation passes. Default: NULL
#' @param max_time Float (0-120.0). The amount of time in seconds that the query should take maximum. Network can cause some overhead so it will be a soft limit. Default: NULL
#' @param tidy Whether to tidy the results into a tibble. Default: TRUE (tidy the results)
#'
#' @returns The results of the inference
#' @seealso
#' \url{https://huggingface.co/docs/transformers/main/en/pipeline_tutorial}
hf_ez_summarization_local_inference <- function(string, min_length = NULL, max_length = NULL, top_k = NULL, top_p = NULL, temperature = 1.0, repetition_penalty = NULL, max_time = NULL, tidy = TRUE, ...) {

  dots <- list(...)

  model <- dots$model

  payload <- list(inputs = string)

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

  # Create an unnamed list by default.
  if(!is.null(names(results))){
    results <- list(results)
  }

  if(tidy){
    results %>%
      jsonlite::toJSON(auto_unbox = TRUE) %>%
      jsonlite::fromJSON(flatten = TRUE)
  }else{
    results
  }
}


#' Summarization API Inference
#'
#' @param string a string to be summarized
#' @param min_length Integer to define the minimum length in tokens of the output summary. Default: NULL
#' @param max_length Integer to define the maximum length in tokens of the output summary. Default: NULL
#' @param top_k Integer to define the top tokens considered within the sample operation to create new text. Default: NULL
#' @param top_p Float to define the tokens that are within the sample operation of text generation. Add tokens in the sample for more probable to least probable until the sum of the probabilities is greater than top_p. Default: NULL
#' @param temperature Float (0.0-100.0). The temperature of the sampling operation. 1 means regular sampling, 0 means always take the highest score, 100.0 is getting closer to uniform probability. Default: 1.0
#' @param repetition_penalty Float (0.0-100.0). The more a token is used within generation the more it is penalized to not be picked in successive generation passes. Default: NULL
#' @param max_time Float (0-120.0). The amount of time in seconds that the query should take maximum. Network can cause some overhead so it will be a soft limit. Default: NULL
#' @param tidy Whether to tidy the results into a tibble. Default: TRUE (tidy the results)
#' @param use_gpu Whether to use GPU for inference.
#' @param use_cache Whether to use cached inference results for previously seen inputs.
#' @param wait_for_model Whether to wait for the model to be ready instead of receiving a 503 error after a certain amount of time.
#' @param use_auth_token The token to use as HTTP bearer authorization for the Inference API. Defaults to HUGGING_FACE_HUB_TOKEN environment variable.
#' @param stop_on_error Whether to throw an error if an API error is encountered. Defaults to FALSE (do not throw error).
#'
#' @returns The results of the inference
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/index}
hf_ez_summarization_api_inference <- function(string, min_length = NULL, max_length = NULL, top_k = NULL, top_p = NULL, temperature = 1.0, repetition_penalty = NULL, max_time = NULL, tidy = TRUE, use_gpu = FALSE, use_cache = FALSE, wait_for_model = FALSE, use_auth_token = NULL, stop_on_error = FALSE, ...) {

  dots <- list(...)

  model <- dots$model

  payload <- list(inputs = string,
                  options = environment() %>% as.list() %>% purrr::list_modify(string = NULL, use_auth_token = NULL, model = NULL) %>% purrr::compact())

  results <-
    hf_make_api_request(model = model, payload = payload, use_auth_token = use_auth_token, stop_on_error = stop_on_error)

  # Create an unnamed list by default.
  if(!is.null(names(results))){
    results <- list(results)
  }

  if(tidy){
    results %>%
      jsonlite::toJSON(auto_unbox = TRUE) %>%
      jsonlite::fromJSON(flatten = TRUE)
  }else{
    results
  }
}


################# Question Answering ##################

#' Load a Model that can Answer Questions
#'
#' Want to have a nice know-it-all bot that can answer any question?
#'
#' @param model_id A model_id. Run hf_search_models(...) for model_ids. Defaults to 'deepset/roberta-base-squad2'.
#' @param use_api Whether to use the Inference API to run the model (TRUE) or download and run the model locally (FALSE). Defaults to FALSE
#'
#' @returns A question answering object
#' @export
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/detailed_parameters#question-answering-task}
hf_ez_question_answering <- function(model_id = 'deepset/roberta-base-squad2', use_api = FALSE){

  task <- 'question-answering'

  if(use_api){
    infer_function <- function() {args <- as.list(environment()); do.call(hf_ez_question_answering_api_inference, args %>% append(list(model = model_id)))}

    formals(infer_function) <- formals(hf_ez_question_answering_api_inference)

    list(
      model_id = model_id,
      task = task,
      infer = infer_function
    )

  }else{
    pipeline <- hf_load_pipeline(model_id = model_id, task = task)
    infer_function <- function() {args <- as.list(environment()); do.call(hf_ez_question_answering_local_inference, args %>% append(list(model = pipeline)))}

    formals(infer_function) <- formals(hf_ez_question_answering_local_inference)

    list(
      model_id = model_id,
      task = task,
      infer = infer_function,
      .raw = pipeline
    )
  }
}


#' Question Answering Local Inference
#'
#' @param question a question to be answered based on the provided context
#' @param context the context to consult for answering the question
#' @param tidy Whether to tidy the results into a tibble. Default: TRUE (tidy the results)
#'
#' @returns The results of the inference
#' @seealso
#' \url{https://huggingface.co/docs/transformers/main/en/pipeline_tutorial}
hf_ez_question_answering_local_inference <- function(question, context, tidy = TRUE, ...) {

  dots <- list(...)

  model <- dots$model

  payload <-
    list(
      inputs =
        list(
          question = question,
          context = context
        ))

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

  # Create an unnamed list by default.
  if(!is.null(names(results))){
    results <- list(results)
  }

  if(tidy){
    results %>%
      jsonlite::toJSON(auto_unbox = TRUE) %>%
      jsonlite::fromJSON(flatten = TRUE)
  }else{
    results
  }
}


#' Question Answering API Inference
#'
#' @param question a question to be answered based on the provided context
#' @param context the context to consult for answering the question
#' @param tidy Whether to tidy the results into a tibble. Default: TRUE (tidy the results)
#' @param use_gpu Whether to use GPU for inference.
#' @param use_cache Whether to use cached inference results for previously seen inputs.
#' @param wait_for_model Whether to wait for the model to be ready instead of receiving a 503 error after a certain amount of time.
#' @param use_auth_token The token to use as HTTP bearer authorization for the Inference API. Defaults to HUGGING_FACE_HUB_TOKEN environment variable.
#' @param stop_on_error Whether to throw an error if an API error is encountered. Defaults to FALSE (do not throw error).
#'
#' @returns The results of the inference
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/index}
hf_ez_question_answering_api_inference <- function(question, context, tidy = TRUE, use_gpu = FALSE, use_cache = FALSE, wait_for_model = FALSE, use_auth_token = NULL, stop_on_error = FALSE, ...) {

  dots <- list(...)

  model <- dots$model

  payload <-
    list(
      inputs =
        list(
          question = question,
          context = context
        ),
      options = environment() %>% as.list() %>% purrr::list_modify(string = NULL, use_auth_token = NULL, model = NULL) %>% purrr::compact()
    )

  results <-
    hf_make_api_request(model = model, payload = payload, use_auth_token = use_auth_token, stop_on_error = stop_on_error)

  # Create an unnamed list by default.
  if(!is.null(names(results))){
    results <- list(results)
  }

  if(tidy){
    results %>%
      jsonlite::toJSON(auto_unbox = TRUE) %>%
      jsonlite::fromJSON(flatten = TRUE)
  }else{
    results
  }
}



################# Table Question Answering ##################


#' Load a Model that can Answer Questions about a Table
#'
#' Don’t know SQL? Don’t want to dive into a large spreadsheet? Ask questions in plain english!
#'
#' @param model_id A model_id. Run hf_search_models(...) for model_ids. Defaults to 'google/tapas-base-finetuned-wtq'.
#' @param use_api Whether to use the Inference API to run the model (TRUE) or download and run the model locally (FALSE). Defaults to FALSE
#'
#' @returns A table question answering object
#' @export
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/detailed_parameters#table-question-answering-task}
hf_ez_table_question_answering <- function(model_id = 'google/tapas-base-finetuned-wtq', use_api = FALSE){

  task <- 'table-question-answering'

  if(use_api){
    infer_function <- function() {args <- as.list(environment()); do.call(hf_ez_table_question_answering_api_inference, args %>% append(list(model = model_id)))}

    formals(infer_function) <- formals(hf_ez_table_question_answering_api_inference)

    list(
      model_id = model_id,
      task = task,
      infer = infer_function
    )

  }else{
    pipeline <- hf_load_pipeline(model_id = model_id, task = task)
    infer_function <- function() {args <- as.list(environment()); do.call(hf_ez_table_question_answering_local_inference, args %>% append(list(model = pipeline)))}

    formals(infer_function) <- formals(hf_ez_table_question_answering_local_inference)

    list(
      model_id = model_id,
      task = task,
      infer = infer_function,
      .raw = pipeline
    )
  }
}


#' Table Question Answering Local Inference
#'
#' @param query The query in plain text that you want to ask the table
#' @param table A dataframe with all text columns.
#' @param tidy Whether to tidy the results into a tibble. Default: TRUE (tidy the results)
#'
#' @returns The results of the inference
#' @seealso
#' \url{https://huggingface.co/docs/transformers/main/en/pipeline_tutorial}
hf_ez_table_question_answering_local_inference <- function(query, table, tidy = TRUE, ...) {

  dots <- list(...)

  model <- dots$model

  payload <-
    list(
      inputs =
        list(
          query = query,
          table = table
        ))

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

  # Create an unnamed list by default.
  if(!is.null(names(results))){
    results <- list(results)
  }

  if(tidy){
    results %>%
      jsonlite::toJSON(auto_unbox = TRUE) %>%
      jsonlite::fromJSON(flatten = TRUE)
  }else{
    results
  }
}


#' Table Question Answering API Inference
#'
#' @param query The query in plain text that you want to ask the table
#' @param table A dataframe with all text columns.
#' @param tidy Whether to tidy the results into a tibble. Default: TRUE (tidy the results)
#' @param use_gpu Whether to use GPU for inference.
#' @param use_cache Whether to use cached inference results for previously seen inputs.
#' @param wait_for_model Whether to wait for the model to be ready instead of receiving a 503 error after a certain amount of time.
#' @param use_auth_token The token to use as HTTP bearer authorization for the Inference API. Defaults to HUGGING_FACE_HUB_TOKEN environment variable.
#' @param stop_on_error Whether to throw an error if an API error is encountered. Defaults to FALSE (do not throw error).
#'
#' @returns The results of the inference
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/index}
hf_ez_table_question_answering_api_inference <- function(query, table, tidy = TRUE, use_gpu = FALSE, use_cache = FALSE, wait_for_model = FALSE, use_auth_token = NULL, stop_on_error = FALSE, ...) {

  dots <- list(...)

  model <- dots$model

  payload <-
    list(
      inputs =
        list(
          query = query,
          table = table
        ),
      options = environment() %>% as.list() %>% purrr::list_modify(string = NULL, use_auth_token = NULL, model = NULL) %>% purrr::compact()
    )

  results <-
    hf_make_api_request(model = model, payload = payload, use_auth_token = use_auth_token, stop_on_error = stop_on_error)

  # Create an unnamed list by default.
  if(!is.null(names(results))){
    results <- list(results)
  }

  if(tidy){
    results %>%
      jsonlite::toJSON(auto_unbox = TRUE) %>%
      jsonlite::fromJSON(flatten = TRUE)
  }else{
    results
  }
}



################# Sentence Similarity ##################


#' Load a Model that can Compare Sentences
#'
#' Calculate the semantic similarity between one text and a list of other sentences by comparing their embeddings.
#'
#' @param model_id A model_id. Run hf_search_models(...) for model_ids. Defaults to 'sentence-transformers/all-MiniLM-L6-v2'.
#' @param use_api Whether to use the Inference API to run the model (TRUE) or download and run the model locally (FALSE). Defaults to FALSE
#'
#' @returns A sentence similarity object
#' @export
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/detailed_parameters#sentence-similarity-task}
hf_ez_sentence_similarity <- function(model_id = 'sentence-transformers/all-MiniLM-L6-v2', use_api = FALSE){

  task <- 'sentence-similarity'

  if(use_api){
    infer_function <- function() {args <- as.list(environment()); do.call(hf_ez_sentence_similarity_api_inference, args %>% append(list(model = model_id)))}

    formals(infer_function) <- formals(hf_ez_sentence_similarity_api_inference)

    list(
      model_id = model_id,
      task = task,
      infer = infer_function
    )

  }else{
    pipeline <- hf_load_sentence_model(model_id = model_id)
    infer_function <- function() {args <- as.list(environment()); do.call(hf_ez_sentence_similarity_local_inference, args %>% append(list(model = pipeline)))}

    formals(infer_function) <- formals(hf_ez_sentence_similarity_local_inference)

    list(
      model_id = model_id,
      task = task,
      infer = infer_function,
      .raw = pipeline
    )
  }
}


#' Sentence Similarity Local Inference
#'
#' @param source_sentence The string that you wish to compare the other strings with. This can be a phrase, sentence, or longer passage, depending on the model being used.
#' @param sentences A list of strings which will be compared against the source_sentence.
#' @param tidy Whether to tidy the results into a tibble. Default: TRUE (tidy the results)
#'
#' @returns The results of the inference
#' @seealso
#' \url{https://huggingface.co/docs/transformers/main/en/pipeline_tutorial}
hf_ez_sentence_similarity_local_inference <- function(source_sentence, sentences, tidy = TRUE, ...) {

  dots <- list(...)

  model <- dots$model

  payload <-
    list(
      inputs =
        list(
          source_sentence = source_sentence,
          sentences = sentences
        ))

  if(!require('lsa', quietly = T)) stop("You must install package lsa to compute sentence similarities.")

  similarities <-
    apply(model$encode(payload$inputs$sentences), 1, function(x) lsa::cosine(x, model$encode(payload$inputs$source_sentence) %>% as.numeric()))

  results <-
    list(
      sentence = sentences,
      similarity = similarities
    )

  # Create an unnamed list by default.
  if(!is.null(names(results))){
    results <- list(results)
  }

  if(tidy){
    results %>%
      as.data.frame()
  }else{
    results
  }
}


#' Sentence Similarity API Inference
#'
#' @param source_sentence The string that you wish to compare the other strings with. This can be a phrase, sentence, or longer passage, depending on the model being used.
#' @param sentences A list of strings which will be compared against the source_sentence.
#' @param tidy Whether to tidy the results into a tibble. Default: TRUE (tidy the results)
#' @param use_gpu Whether to use GPU for inference.
#' @param use_cache Whether to use cached inference results for previously seen inputs.
#' @param wait_for_model Whether to wait for the model to be ready instead of receiving a 503 error after a certain amount of time.
#' @param use_auth_token The token to use as HTTP bearer authorization for the Inference API. Defaults to HUGGING_FACE_HUB_TOKEN environment variable.
#' @param stop_on_error Whether to throw an error if an API error is encountered. Defaults to FALSE (do not throw error).
#'
#' @returns The results of the inference
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/index}
hf_ez_sentence_similarity_api_inference <- function(source_sentence, sentences, tidy = TRUE, use_gpu = FALSE, use_cache = FALSE, wait_for_model = FALSE, use_auth_token = NULL, stop_on_error = FALSE, ...) {

  function_args <- environment() %>% as.list()

  api_args <- function_args[c('use_gpu', 'use_cache', 'wait_for_model', 'stop_on_error')]

  dots <- list(...)

  model <- dots$model

  payload <-
    list(
      inputs =
        list(
          source_sentence = source_sentence,
          sentences = sentences
        ),
      options = api_args
    )

  results <-
    hf_make_api_request(model = model, payload = payload, use_auth_token = use_auth_token, stop_on_error = stop_on_error)

  similarities <-
    results %>%
    as.numeric()

  results <-
    list(
      sentence = sentences,
      similarity = similarities
    )

  # Create an unnamed list by default.
  if(!is.null(names(results))){
    results <- list(results)
  }

  if(tidy){
    results %>%
      as.data.frame()
  }else{
    results
  }
}



################# Text Classification ##################

#' Load a Model that can Classify Texts
#'
#' Usually used for sentiment-analysis this will output the likelihood of classes of an input.
#'
#' @param model_id A model_id. Run hf_search_models(...) for model_ids. Defaults to 'distilbert-base-uncased-finetuned-sst-2-english'.
#' @param use_api Whether to use the Inference API to run the model (TRUE) or download and run the model locally (FALSE). Defaults to FALSE
#'
#' @returns A text classification object
#' @export
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/detailed_parameters#text-classification-task}
hf_ez_text_classification <- function(model_id = 'distilbert-base-uncased-finetuned-sst-2-english', use_api = FALSE){

  task <- 'text-classification'

  if(use_api){
    infer_function <- function() {args <- as.list(environment()); do.call(hf_ez_text_classification_api_inference, args %>% append(list(model = model_id)))}

    formals(infer_function) <- formals(hf_ez_text_classification_api_inference)

    list(
      model_id = model_id,
      task = task,
      infer = infer_function
    )

  }else{
    pipeline <- hf_load_pipeline(model_id = model_id, task = task)
    infer_function <- function() {args <- as.list(environment()); do.call(hf_ez_text_classification_local_inference, args %>% append(list(model = pipeline)))}

    formals(infer_function) <- formals(hf_ez_text_classification_local_inference)

    list(
      model_id = model_id,
      task = task,
      infer = infer_function,
      .raw = pipeline
    )
  }
}


#' Text Classification Local Inference
#'
#' @param string a string to be classified
#' @param tidy Whether to tidy the results into a tibble. Default: TRUE (tidy the results)
#'
#' @returns The results of the inference
#' @seealso
#' \url{https://huggingface.co/docs/transformers/main/en/pipeline_tutorial}
hf_ez_text_classification_local_inference <- function(string, tidy = TRUE, ...) {

  dots <- list(...)

  model <- dots$model

  payload <-
    list(
      inputs = string
    )

  # If local model object is passed in to model, perform local inference.
  if (any(stringr::str_detect(class(model), "pipelines"))) {

    # If inputs is an unnamed list of strings
    if(length(names(payload[[1]])) == 0){
      function_params <-
        append(list(payload[[1]] %>% as.character()), payload[-1] %>% unname() %>% unlist(recursive = F) %>% as.list()) %>% append(list(return_all_scores = TRUE))
    }else{
      function_params <-
        payload %>% unname() %>% unlist(recursive = F) %>% as.list() %>% append(list(return_all_scores = TRUE))
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

  # Create an unnamed list by default.
  if(!is.null(names(results))){
    results <-
      list(results)
  }

  # Reformat results.
  results <-
    results %>%
    purrr::imap(~ append(list(string = string[[.y]]), .x %>% dplyr::bind_rows() %>% as.list()))

  if(tidy){
    results %>%
      dplyr::bind_rows() %>%
      tidyr::pivot_wider(names_from = label, values_from = score)
  }else{
    results
  }
}


#' Text Classification API Inference
#'
#' @param string a string to be classified
#' @param tidy Whether to tidy the results into a tibble. Default: TRUE (tidy the results)
#' @param use_gpu Whether to use GPU for inference.
#' @param use_cache Whether to use cached inference results for previously seen inputs.
#' @param wait_for_model Whether to wait for the model to be ready instead of receiving a 503 error after a certain amount of time.
#' @param use_auth_token The token to use as HTTP bearer authorization for the Inference API. Defaults to HUGGING_FACE_HUB_TOKEN environment variable.
#' @param stop_on_error Whether to throw an error if an API error is encountered. Defaults to FALSE (do not throw error).
#'
#' @returns The results of the inference
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/index}
hf_ez_text_classification_api_inference <- function(string, tidy = TRUE, use_gpu = FALSE, use_cache = FALSE, wait_for_model = FALSE, use_auth_token = NULL, stop_on_error = FALSE, ...) {

  function_args <- environment() %>% as.list()

  api_args <- function_args[c('use_gpu', 'use_cache', 'wait_for_model', 'stop_on_error')]

  dots <- list(...)

  model <- dots$model

  payload <-
    list(
      inputs = string,
      options = api_args
    )

  results <-
    hf_make_api_request(model = model, payload = payload, use_auth_token = use_auth_token, stop_on_error = stop_on_error)

  # Reformat results.
  results <-
    results %>%
    purrr::imap(~ append(list(string = string[[.y]]), .x %>% dplyr::bind_rows() %>% as.list()))

  if(tidy){
    results %>%
      dplyr::bind_rows() %>%
      tidyr::pivot_wider(names_from = label, values_from = score)
  }else{
    results
  }
}


################# Text Generation ##################

#' Load a Model that can Generate Text from a Prompt
#'
#' Use to continue text from a prompt. This is a very generic task.
#'
#' @param model_id A model_id. Run hf_search_models(...) for model_ids. Defaults to 'gpt2'.
#' @param use_api Whether to use the Inference API to run the model (TRUE) or download and run the model locally (FALSE). Defaults to FALSE
#'
#' @returns A text generation object
#' @export
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task}
hf_ez_text_generation <- function(model_id = 'gpt2', use_api = FALSE){

  task <- 'text-generation'

  if(use_api){
    infer_function <- function() {args <- as.list(environment()); do.call(hf_ez_text_generation_api_inference, args %>% append(list(model = model_id)))}

    formals(infer_function) <- formals(hf_ez_text_generation_api_inference)

    list(
      model_id = model_id,
      task = task,
      infer = infer_function
    )

  }else{
    pipeline <- hf_load_pipeline(model_id = model_id, task = task)
    infer_function <- function() {args <- as.list(environment()); do.call(hf_ez_text_generation_local_inference, args %>% append(list(model = pipeline)))}

    formals(infer_function) <- formals(hf_ez_text_generation_local_inference)

    list(
      model_id = model_id,
      task = task,
      infer = infer_function,
      .raw = pipeline
    )
  }
}


#' Text Generation Local Inference
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
#' @param do_sample (Optional: True). Bool. Whether or not to use sampling, use greedy decoding otherwise.#'
#' @returns The results of the inference
#' @seealso
#' \url{https://huggingface.co/docs/transformers/main/en/pipeline_tutorial}
hf_ez_text_generation_local_inference <- function(string, top_k = NULL, top_p = NULL, temperature = 1.0, repetition_penalty = NULL, max_new_tokens = NULL, max_time = NULL, return_full_text = TRUE, num_return_sequences = 1L, do_sample = TRUE, tidy = TRUE, ...) {

  dots <- list(...)

  model <- dots$model

  payload <- list(inputs = string,
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
                    ))

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

  # Create an unnamed list by default.
  if(!is.null(names(results))){
    results <- list(results)
  }

  if(tidy){
    results %>%
      jsonlite::toJSON(auto_unbox = TRUE) %>%
      jsonlite::fromJSON(flatten = TRUE)
  }else{
    results
  }
}


#' Text Generation API Inference
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
#' @param do_sample (Optional: True). Bool. Whether or not to use sampling, use greedy decoding otherwise.#'
#' @param use_gpu Whether to use GPU for inference.
#' @param use_cache Whether to use cached inference results for previously seen inputs.
#' @param wait_for_model Whether to wait for the model to be ready instead of receiving a 503 error after a certain amount of time.
#' @param use_auth_token The token to use as HTTP bearer authorization for the Inference API. Defaults to HUGGING_FACE_HUB_TOKEN environment variable.
#' @param stop_on_error Whether to throw an error if an API error is encountered. Defaults to FALSE (do not throw error).
#'
#' @returns The results of the inference
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/index}
hf_ez_text_generation_api_inference <- function(string, top_k = NULL, top_p = NULL, temperature = 1.0, repetition_penalty = NULL, max_new_tokens = NULL, max_time = NULL, return_full_text = TRUE, num_return_sequences = 1L, do_sample = TRUE, tidy = TRUE, use_gpu = FALSE, use_cache = FALSE, wait_for_model = FALSE, use_auth_token = NULL, stop_on_error = FALSE, ...) {

  function_args <- environment() %>% as.list()

  api_args <- function_args[c('use_gpu', 'use_cache', 'wait_for_model', 'stop_on_error')]

  dots <- list(...)

  model <- dots$model

  payload <-
    list(inputs = string,
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
           ),
         options = api_args
    )

  results <-
    hf_make_api_request(model = model, payload = payload, use_auth_token = use_auth_token, stop_on_error = stop_on_error)

  # Create an unnamed list by default.
  if(!is.null(names(results))){
    results <- list(results)
  }

  if(tidy){
    results %>%
      jsonlite::toJSON(auto_unbox = TRUE) %>%
      jsonlite::fromJSON(flatten = TRUE)
  }else{
    results
  }
}




################# Text2Text Generation ##################

#' Load a Model that can Answer General Questions
#'
#' Essentially Text-generation task. But uses Encoder-Decoder architecture, so might change in the future for more options.
#'
#' @param model_id A model_id. Run hf_search_models(...) for model_ids. Defaults to 'google/flan-t5-small'.
#' @param use_api Whether to use the Inference API to run the model (TRUE) or download and run the model locally (FALSE). Defaults to FALSE
#'
#' @returns A text2text generation object
#' @export
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/detailed_parameters#text2text-generation-task}
hf_ez_text2text_generation <- function(model_id = 'google/flan-t5-small', use_api = FALSE){

  task <- 'text2text-generation'

  if(use_api){
    infer_function <- function() {args <- as.list(environment()); do.call(hf_ez_text2text_generation_api_inference, args %>% append(list(model = model_id)))}

    formals(infer_function) <- formals(hf_ez_text2text_generation_api_inference)

    list(
      model_id = model_id,
      task = task,
      infer = infer_function
    )

  }else{
    pipeline <- hf_load_pipeline(model_id = model_id, task = task)
    infer_function <- function() {args <- as.list(environment()); do.call(hf_ez_text2text_generation_local_inference, args %>% append(list(model = pipeline)))}

    formals(infer_function) <- formals(hf_ez_text2text_generation_local_inference)

    list(
      model_id = model_id,
      task = task,
      infer = infer_function,
      .raw = pipeline
    )
  }
}


#' Text2Text Generation Local Inference
#'
#' @param string a general request for the model to perform or answer
#' @param tidy Whether to tidy the results into a tibble. Default: TRUE (tidy the results)
#' @returns The results of the inference
#' @seealso
#' \url{https://huggingface.co/docs/transformers/main/en/pipeline_tutorial}
hf_ez_text2text_generation_local_inference <- function(string, tidy = TRUE, ...) {

  dots <- list(...)

  model <- dots$model

  payload <- list(inputs = string)

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

  # Create an unnamed list by default.
  if(!is.null(names(results))){
    results <- list(results)
  }

  if(tidy){
    results %>%
      jsonlite::toJSON(auto_unbox = TRUE) %>%
      jsonlite::fromJSON(flatten = TRUE)
  }else{
    results
  }
}


#' Text2Text Generation API Inference
#'
#' @param string a general request for the model to perform or answer
#' @param tidy Whether to tidy the results into a tibble. Default: TRUE (tidy the results)
#' @param use_gpu Whether to use GPU for inference.
#' @param use_cache Whether to use cached inference results for previously seen inputs.
#' @param wait_for_model Whether to wait for the model to be ready instead of receiving a 503 error after a certain amount of time.
#' @param use_auth_token The token to use as HTTP bearer authorization for the Inference API. Defaults to HUGGING_FACE_HUB_TOKEN environment variable.
#' @param stop_on_error Whether to throw an error if an API error is encountered. Defaults to FALSE (do not throw error).
#'
#' @returns The results of the inference
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/index}
hf_ez_text2text_generation_api_inference <- function(string, tidy = TRUE, use_gpu = FALSE, use_cache = FALSE, wait_for_model = FALSE, use_auth_token = NULL, stop_on_error = FALSE, ...) {

  function_args <- environment() %>% as.list()

  api_args <- function_args[c('use_gpu', 'use_cache', 'wait_for_model', 'stop_on_error')]

  dots <- list(...)

  model <- dots$model

  payload <-
    list(inputs = string,
         options = api_args
    )

  results <-
    hf_make_api_request(model = model, payload = payload, use_auth_token = use_auth_token, stop_on_error = stop_on_error)

  # Create an unnamed list by default.
  if(!is.null(names(results))){
    results <- list(results)
  }

  if(tidy){
    results %>%
      jsonlite::toJSON(auto_unbox = TRUE) %>%
      jsonlite::fromJSON(flatten = TRUE)
  }else{
    results
  }
}



################# Token Classification ##################

#' Load a Model that can Classify parts of a Text
#'
#' Usually used for sentence parsing, either grammatical, or Named Entity Recognition (NER) to understand keywords contained within text.
#'
#' @param model_id A model_id. Run hf_search_models(...) for model_ids. Defaults to 'dbmdz/bert-large-cased-finetuned-conll03-english'.
#' @param use_api Whether to use the Inference API to run the model (TRUE) or download and run the model locally (FALSE). Defaults to FALSE
#'
#' @returns A text2text generation object
#' @examples
#' \dontrun{
#' # Load the default named entity recognition model
#' ner <- hf_ez_token_classification()
#'
#' # Run NER. Note how the full name is aggregated into one named entity.
#' ner$infer(string = "My name is Sarah Jessica Parker but you can call me Jessica",
#' aggregation_strategy = 'simple')
#'
#' # Run NER without aggregation. Note how the full name is separated into distinct named entities.
#' ner$infer(string = "My name is Sarah Jessica Parker but you can call me Jessica",
#' aggregation_strategy = 'none')
#' }
#' @export
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/detailed_parameters#token-classification-task}
hf_ez_token_classification <- function(model_id = 'dbmdz/bert-large-cased-finetuned-conll03-english', use_api = FALSE){

  task <- 'token-classification'

  if(use_api){
    infer_function <- function() {args <- as.list(environment()); do.call(hf_ez_token_classification_api_inference, args %>% append(list(model = model_id)))}

    formals(infer_function) <- formals(hf_ez_token_classification_api_inference)

    list(
      model_id = model_id,
      task = task,
      infer = infer_function
    )

  }else{
    pipeline <- hf_load_pipeline(model_id = model_id, task = task)
    infer_function <- function() {args <- as.list(environment()); do.call(hf_ez_token_classification_local_inference, args %>% append(list(model = pipeline)))}

    formals(infer_function) <- formals(hf_ez_token_classification_local_inference)

    list(
      model_id = model_id,
      task = task,
      infer = infer_function,
      .raw = pipeline
    )
  }
}


#' Token Classification Local Inference
#'
#' @param string a string to be classified
#' @param aggregation_strategy (Default: simple). There are several aggregation strategies. \cr
#' none: Every token gets classified without further aggregation.  \cr
#' simple: Entities are grouped according to the default schema (B-, I- tags get merged when the tag is similar).  \cr
#' first: Same as the simple strategy except words cannot end up with different tags. Words will use the tag of the first token when there is ambiguity.  \cr
#' average: Same as the simple strategy except words cannot end up with different tags. Scores are averaged across tokens and then the maximum label is applied.  \cr
#' max: Same as the simple strategy except words cannot end up with different tags. Word entity will be the token with the maximum score.  \cr
#' @param tidy Whether to tidy the results into a tibble. Default: TRUE (tidy the results)
#' @returns The results of the inference
#' @seealso
#' \url{https://huggingface.co/docs/transformers/main/en/pipeline_tutorial}
hf_ez_token_classification_local_inference <- function(string, aggregation_strategy = 'simple', tidy = TRUE, ...) {

  dots <- list(...)

  model <- dots$model

  payload <-
    list(
      inputs = string,
      parameters =
        list(aggregation_strategy = aggregation_strategy)
    )

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

  # Create an unnamed list by default.
  if(!is.null(names(results))){
    results <- list(results)
  }

  if(tidy){
    results %>%
      jsonlite::toJSON(auto_unbox = TRUE) %>%
      jsonlite::fromJSON(flatten = TRUE)
  }else{
    results
  }
}


#' Token Classification API Inference
#'
#' @param string a string to be classified
#' @param aggregation_strategy (Default: simple). There are several aggregation strategies. \cr
#' none: Every token gets classified without further aggregation.  \cr
#' simple: Entities are grouped according to the default schema (B-, I- tags get merged when the tag is similar).  \cr
#' first: Same as the simple strategy except words cannot end up with different tags. Words will use the tag of the first token when there is ambiguity.  \cr
#' average: Same as the simple strategy except words cannot end up with different tags. Scores are averaged across tokens and then the maximum label is applied.  \cr
#' max: Same as the simple strategy except words cannot end up with different tags. Word entity will be the token with the maximum score.  \cr
#' @param tidy Whether to tidy the results into a tibble. Default: TRUE (tidy the results)
#' @param use_gpu Whether to use GPU for inference.
#' @param use_cache Whether to use cached inference results for previously seen inputs.
#' @param wait_for_model Whether to wait for the model to be ready instead of receiving a 503 error after a certain amount of time.
#' @param use_auth_token The token to use as HTTP bearer authorization for the Inference API. Defaults to HUGGING_FACE_HUB_TOKEN environment variable.
#' @param stop_on_error Whether to throw an error if an API error is encountered. Defaults to FALSE (do not throw error).
#'
#' @returns The results of the inference
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/index}
hf_ez_token_classification_api_inference <- function(string, aggregation_strategy = 'simple', tidy = TRUE, use_gpu = FALSE, use_cache = FALSE, wait_for_model = FALSE, use_auth_token = NULL, stop_on_error = FALSE, ...) {

  function_args <- environment() %>% as.list()

  api_args <- function_args[c('use_gpu', 'use_cache', 'wait_for_model', 'stop_on_error')]

  dots <- list(...)

  model <- dots$model

  payload <-
    list(inputs = string,
         parameters =
           list(aggregation_strategy = aggregation_strategy),
         options = api_args
    )

  results <-
    hf_make_api_request(model = model, payload = payload, use_auth_token = use_auth_token, stop_on_error = stop_on_error)

  # Create an unnamed list by default.
  if(!is.null(names(results))){
    results <- list(results)
  }

  if(tidy){
    results %>%
      jsonlite::toJSON(auto_unbox = TRUE) %>%
      jsonlite::fromJSON(flatten = TRUE)
  }else{
    results
  }
}


################# Translation ##################

#' Load a Model that can Translate between Languages
#'
#' This task is well known to translate text from one language to another
#'
#' @param model_id A model_id. Run hf_search_models(...) for model_ids. Defaults to 'Helsinki-NLP/opus-mt-en-es'.
#' @param use_api Whether to use the Inference API to run the model (TRUE) or download and run the model locally (FALSE). Defaults to FALSE
#'
#' @returns A translation object
#' @examples
#' \dontrun{
#' # Load the default translation model
#' tr <- hf_ez_translation()
#'
#' # Translate from Engllish to Spanish.
#' tr$infer(string = "My name is Sarah and I live in London")
#' }
#' @export
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/detailed_parameters#translation-task}
hf_ez_translation <- function(model_id = 'Helsinki-NLP/opus-mt-en-es', use_api = FALSE){

  task <- 'translation'

  if(use_api){
    infer_function <- function() {args <- as.list(environment()); do.call(hf_ez_translation_api_inference, args %>% append(list(model = model_id)))}

    formals(infer_function) <- formals(hf_ez_translation_api_inference)

    list(
      model_id = model_id,
      task = task,
      infer = infer_function
    )

  }else{
    pipeline <- hf_load_pipeline(model_id = model_id, task = task)
    infer_function <- function() {args <- as.list(environment()); do.call(hf_ez_translation_local_inference, args %>% append(list(model = pipeline)))}

    formals(infer_function) <- formals(hf_ez_translation_local_inference)

    list(
      model_id = model_id,
      task = task,
      infer = infer_function,
      .raw = pipeline
    )
  }
}


#' Translation Local Inference
#'
#' @param string a string to be translated
#' @param tidy Whether to tidy the results into a tibble. Default: TRUE (tidy the results)
#' @returns The results of the inference
#' @seealso
#' \url{https://huggingface.co/docs/transformers/main/en/pipeline_tutorial}
hf_ez_translation_local_inference <- function(string, tidy = TRUE, ...) {

  dots <- list(...)

  model <- dots$model

  payload <-
    list(
      inputs = string
    )

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

  # Create an unnamed list by default.
  if(!is.null(names(results))){
    results <- list(results)
  }

  if(tidy){
    results %>%
      jsonlite::toJSON(auto_unbox = TRUE) %>%
      jsonlite::fromJSON(flatten = TRUE)
  }else{
    results
  }
}


#' Translation API Inference
#'
#' @param string a string to be translated
#' @param tidy Whether to tidy the results into a tibble. Default: TRUE (tidy the results)
#' @param use_gpu Whether to use GPU for inference.
#' @param use_cache Whether to use cached inference results for previously seen inputs.
#' @param wait_for_model Whether to wait for the model to be ready instead of receiving a 503 error after a certain amount of time.
#' @param use_auth_token The token to use as HTTP bearer authorization for the Inference API. Defaults to HUGGING_FACE_HUB_TOKEN environment variable.
#' @param stop_on_error Whether to throw an error if an API error is encountered. Defaults to FALSE (do not throw error).
#'
#' @returns The results of the inference
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/index}
hf_ez_token_classification_api_inference <- function(string, tidy = TRUE, use_gpu = FALSE, use_cache = FALSE, wait_for_model = FALSE, use_auth_token = NULL, stop_on_error = FALSE, ...) {

  function_args <- environment() %>% as.list()

  api_args <- function_args[c('use_gpu', 'use_cache', 'wait_for_model', 'stop_on_error')]

  dots <- list(...)

  model <- dots$model

  payload <-
    list(inputs = string,
         options = api_args
    )

  results <-
    hf_make_api_request(model = model, payload = payload, use_auth_token = use_auth_token, stop_on_error = stop_on_error)

  # Create an unnamed list by default.
  if(!is.null(names(results))){
    results <- list(results)
  }

  if(tidy){
    results %>%
      jsonlite::toJSON(auto_unbox = TRUE) %>%
      jsonlite::fromJSON(flatten = TRUE)
  }else{
    results
  }
}
