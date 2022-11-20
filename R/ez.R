
################# Fill Mask ##################

#' EZ Fill Mask
#'
#' Tries to fill in a hole with a missing word (token to be precise). That’s the base task for BERT models.
#'
#' @param model_id A model_id. Run hf_search_models(...) for model_ids. Defaults to 'bert-base-uncased'.
#' @param use_api Whether to use the Inference API to run the model (TRUE) or download and run the model locally (FALSE). Defaults to FALSE
#'
#' @returns A fill mask object
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
#' @param flatten Whether to flatten the results into a data frame. Default: TRUE (flatten the results)
#'
#' @returns The results of the inference
#' @seealso
#' \url{https://huggingface.co/docs/transformers/main/en/pipeline_tutorial}
hf_ez_fill_mask_local_inference <- function(string, flatten = TRUE, ...) {

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

  if(flatten){
    results %>%
      jsonlite::toJSON(auto_unbox = TRUE) %>%
      jsonlite::fromJSON(flatten = T)
  }else{
    results
  }
}


#' Fill Mask API Inference
#'
#' @param string a string to be filled from, must contain the [MASK] token (check model card for exact name of the mask)
#' @param flatten Whether to flatten the results into a data frame. Default: TRUE (flatten the results)
#' @param use_gpu Whether to use GPU for inference.
#' @param use_cache Whether to use cached inference results for previously seen inputs.
#' @param wait_for_model Whether to wait for the model to be ready instead of receiving a 503 error after a certain amount of time.
#' @param use_auth_token The token to use as HTTP bearer authorization for the Inference API. Defaults to HUGGING_FACE_HUB_TOKEN environment variable.
#' @param stop_on_error Whether to throw an error if an API error is encountered. Defaults to FALSE (do not throw error).
#'
#' @returns The results of the inference
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/index}
hf_ez_fill_mask_api_inference <- function(string, flatten = TRUE, use_gpu = FALSE, use_cache = FALSE, wait_for_model = FALSE, use_auth_token = NULL, stop_on_error = FALSE, ...) {

  dots <- list(...)

  model <- dots$model

  payload <- list(inputs = string,
                  options = environment() %>% as.list() %>% purrr::list_modify(string = NULL, use_auth_token = NULL, model = NULL) %>% purrr::compact())

  if (is.null(use_auth_token) && Sys.getenv("HUGGING_FACE_HUB_TOKEN") != "") use_auth_token <- Sys.getenv("HUGGING_FACE_HUB_TOKEN")

  response <-
    httr2::request(glue::glue("https://api-inference.huggingface.co/models/{model}")) %>%
    httr2::req_auth_bearer_token(token = use_auth_token) %>%
    httr2::req_body_json(
      payload
    ) %>%
    httr2::req_error(is_error = function(resp) stop_on_error) %>%
    httr2::req_perform()

  results <-
    response %>%
    httr2::resp_body_json(auto_unbox = TRUE)

  # Create an unnamed list by default.
  if(!is.null(names(results))){
    results <- list(results)
  }

  if(flatten){
    results %>%
      jsonlite::toJSON(auto_unbox = TRUE) %>%
      jsonlite::fromJSON(flatten = T)
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
#' @param use_api Whether to use the Inference API to run the model (TRUE) or download and run the model locally (FALSE). Defaults to FALSE
#'
#' @returns A summarization object
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
#' @param flatten Whether to flatten the results into a data frame. Default: TRUE (flatten the results)
#'
#' @returns The results of the inference
#' @seealso
#' \url{https://huggingface.co/docs/transformers/main/en/pipeline_tutorial}
hf_ez_summarization_local_inference <- function(string, min_length = NULL, max_length = NULL, top_k = NULL, top_p = NULL, temperature = 1.0, repetition_penalty = NULL, max_time = NULL, flatten = TRUE, ...) {

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

  if(flatten){
    results %>%
      jsonlite::toJSON(auto_unbox = TRUE) %>%
      jsonlite::fromJSON(flatten = T)
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
#' @param flatten Whether to flatten the results into a data frame. Default: TRUE (flatten the results)
#' @param use_gpu Whether to use GPU for inference.
#' @param use_cache Whether to use cached inference results for previously seen inputs.
#' @param wait_for_model Whether to wait for the model to be ready instead of receiving a 503 error after a certain amount of time.
#' @param use_auth_token The token to use as HTTP bearer authorization for the Inference API. Defaults to HUGGING_FACE_HUB_TOKEN environment variable.
#' @param stop_on_error Whether to throw an error if an API error is encountered. Defaults to FALSE (do not throw error).
#'
#' @returns The results of the inference
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/index}
hf_ez_summarization_api_inference <- function(string, min_length = NULL, max_length = NULL, top_k = NULL, top_p = NULL, temperature = 1.0, repetition_penalty = NULL, max_time = NULL, flatten = TRUE, use_gpu = FALSE, use_cache = FALSE, wait_for_model = FALSE, use_auth_token = NULL, stop_on_error = FALSE, ...) {

  dots <- list(...)

  model <- dots$model

  payload <- list(inputs = string,
                  options = environment() %>% as.list() %>% purrr::list_modify(string = NULL, use_auth_token = NULL, model = NULL) %>% purrr::compact())

  if (is.null(use_auth_token) && Sys.getenv("HUGGING_FACE_HUB_TOKEN") != "") use_auth_token <- Sys.getenv("HUGGING_FACE_HUB_TOKEN")

  response <-
    httr2::request(glue::glue("https://api-inference.huggingface.co/models/{model}")) %>%
    httr2::req_auth_bearer_token(token = use_auth_token) %>%
    httr2::req_body_json(
      payload
    ) %>%
    httr2::req_error(is_error = function(resp) stop_on_error) %>%
    httr2::req_perform()

  results <-
    response %>%
    httr2::resp_body_json(auto_unbox = TRUE)

  # Create an unnamed list by default.
  if(!is.null(names(results))){
    results <- list(results)
  }

  if(flatten){
    results %>%
      jsonlite::toJSON(auto_unbox = TRUE) %>%
      jsonlite::fromJSON(flatten = T)
  }else{
    results
  }
}


################# Question Answering ##################


#' EZ Question Answering
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
#' @param flatten Whether to flatten the results into a data frame. Default: TRUE (flatten the results)
#'
#' @returns The results of the inference
#' @seealso
#' \url{https://huggingface.co/docs/transformers/main/en/pipeline_tutorial}
hf_ez_question_answering_local_inference <- function(question, context, flatten = TRUE, ...) {

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

  if(flatten){
    results %>%
      jsonlite::toJSON(auto_unbox = TRUE) %>%
      jsonlite::fromJSON(flatten = T)
  }else{
    results
  }
}


#' Question Answering API Inference
#'
#' @param question a question to be answered based on the provided context
#' @param context the context to consult for answering the question
#' @param flatten Whether to flatten the results into a data frame. Default: TRUE (flatten the results)
#' @param use_gpu Whether to use GPU for inference.
#' @param use_cache Whether to use cached inference results for previously seen inputs.
#' @param wait_for_model Whether to wait for the model to be ready instead of receiving a 503 error after a certain amount of time.
#' @param use_auth_token The token to use as HTTP bearer authorization for the Inference API. Defaults to HUGGING_FACE_HUB_TOKEN environment variable.
#' @param stop_on_error Whether to throw an error if an API error is encountered. Defaults to FALSE (do not throw error).
#'
#' @returns The results of the inference
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/index}
hf_ez_question_answering_api_inference <- function(question, context, flatten = TRUE, use_gpu = FALSE, use_cache = FALSE, wait_for_model = FALSE, use_auth_token = NULL, stop_on_error = FALSE, ...) {

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

  if (is.null(use_auth_token) && Sys.getenv("HUGGING_FACE_HUB_TOKEN") != "") use_auth_token <- Sys.getenv("HUGGING_FACE_HUB_TOKEN")

  response <-
    httr2::request(glue::glue("https://api-inference.huggingface.co/models/{model}")) %>%
    httr2::req_auth_bearer_token(token = use_auth_token) %>%
    httr2::req_body_json(
      payload
    ) %>%
    httr2::req_error(is_error = function(resp) stop_on_error) %>%
    httr2::req_perform()

  results <-
    response %>%
    httr2::resp_body_json(auto_unbox = TRUE)

  # Create an unnamed list by default.
  if(!is.null(names(results))){
    results <- list(results)
  }

  if(flatten){
    results %>%
      jsonlite::toJSON(auto_unbox = TRUE) %>%
      jsonlite::fromJSON(flatten = T)
  }else{
    results
  }
}



################# Table Question Answering ##################


#' EZ Table Question Answering
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
#' @param flatten Whether to flatten the results into a data frame. Default: TRUE (flatten the results)
#'
#' @returns The results of the inference
#' @seealso
#' \url{https://huggingface.co/docs/transformers/main/en/pipeline_tutorial}
hf_ez_table_question_answering_local_inference <- function(query, table, flatten = TRUE, ...) {

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

  if(flatten){
    results %>%
      jsonlite::toJSON(auto_unbox = TRUE) %>%
      jsonlite::fromJSON(flatten = T)
  }else{
    results
  }
}


#' Table Question Answering API Inference
#'
#' @param query The query in plain text that you want to ask the table
#' @param table A dataframe with all text columns.
#' @param flatten Whether to flatten the results into a data frame. Default: TRUE (flatten the results)
#' @param use_gpu Whether to use GPU for inference.
#' @param use_cache Whether to use cached inference results for previously seen inputs.
#' @param wait_for_model Whether to wait for the model to be ready instead of receiving a 503 error after a certain amount of time.
#' @param use_auth_token The token to use as HTTP bearer authorization for the Inference API. Defaults to HUGGING_FACE_HUB_TOKEN environment variable.
#' @param stop_on_error Whether to throw an error if an API error is encountered. Defaults to FALSE (do not throw error).
#'
#' @returns The results of the inference
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/index}
hf_ez_table_question_answering_api_inference <- function(query, table, flatten = TRUE, use_gpu = FALSE, use_cache = FALSE, wait_for_model = FALSE, use_auth_token = NULL, stop_on_error = FALSE, ...) {

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

  if (is.null(use_auth_token) && Sys.getenv("HUGGING_FACE_HUB_TOKEN") != "") use_auth_token <- Sys.getenv("HUGGING_FACE_HUB_TOKEN")

  response <-
    httr2::request(glue::glue("https://api-inference.huggingface.co/models/{model}")) %>%
    httr2::req_auth_bearer_token(token = use_auth_token) %>%
    httr2::req_body_json(
      payload
    ) %>%
    httr2::req_error(is_error = function(resp) stop_on_error) %>%
    httr2::req_perform()

  results <-
    response %>%
    httr2::resp_body_json(auto_unbox = TRUE)

  # Create an unnamed list by default.
  if(!is.null(names(results))){
    results <- list(results)
  }

  if(flatten){
    results %>%
      jsonlite::toJSON(auto_unbox = TRUE) %>%
      jsonlite::fromJSON(flatten = T)
  }else{
    results
  }
}



################# Sentence Similarity ##################


#' EZ Sentence Similarity
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
#' @param flatten Whether to flatten the results into a data frame. Default: TRUE (flatten the results)
#'
#' @returns The results of the inference
#' @seealso
#' \url{https://huggingface.co/docs/transformers/main/en/pipeline_tutorial}
hf_ez_sentence_similarity_local_inference <- function(source_sentence, sentences, flatten = TRUE, ...) {

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

  if(flatten){
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
#' @param flatten Whether to flatten the results into a data frame. Default: TRUE (flatten the results)
#' @param use_gpu Whether to use GPU for inference.
#' @param use_cache Whether to use cached inference results for previously seen inputs.
#' @param wait_for_model Whether to wait for the model to be ready instead of receiving a 503 error after a certain amount of time.
#' @param use_auth_token The token to use as HTTP bearer authorization for the Inference API. Defaults to HUGGING_FACE_HUB_TOKEN environment variable.
#' @param stop_on_error Whether to throw an error if an API error is encountered. Defaults to FALSE (do not throw error).
#'
#' @returns The results of the inference
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/index}
hf_ez_sentence_similarity_api_inference <- function(source_sentence, sentences, flatten = TRUE, use_gpu = FALSE, use_cache = FALSE, wait_for_model = FALSE, use_auth_token = NULL, stop_on_error = FALSE, ...) {

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

  if (is.null(use_auth_token) && Sys.getenv("HUGGING_FACE_HUB_TOKEN") != "") use_auth_token <- Sys.getenv("HUGGING_FACE_HUB_TOKEN")

  response <-
    httr2::request(glue::glue("https://api-inference.huggingface.co/models/{model}")) %>%
    httr2::req_auth_bearer_token(token = use_auth_token) %>%
    httr2::req_body_json(
      payload
    ) %>%
    httr2::req_error(is_error = function(resp) stop_on_error) %>%
    httr2::req_perform()

  similarities <-
    response %>%
    httr2::resp_body_json(auto_unbox = TRUE) %>%
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

  if(flatten){
    results %>%
      as.data.frame()
  }else{
    results
  }
}

