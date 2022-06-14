
hf_pipeline <- function(model_id, tokenizer = NULL, task = NULL, config = NULL,
                        feature_extractor = NULL, framework = NULL, revision = NULL,
                        use_fast = NULL, use_auth_token = NULL, model_kwargs = NULL,
                        pipeline_class = NULL){

  hf_load_pipeline()

  if(is.null(tokenizer)) tokenizer <- reticulate::py$AutoTokenizer$from_pretrained(model_id)

  reticulate::py$pipeline(task = task, model = model_id, tokenizer = tokenizer)
}


#' Load Model
#' Load Model from Huggingface
#'
#' @param model_id The id of the model given in the url by https://huggingface.co/model_name.
#' @param tokenizer The tokenizer function used to tokenize inputs. Defaults to NULL (one will be automatically loaded).
#' @param task The task the model will accomplish. Run hf_list_tasks() for options.
#' @returns A Huggingface model ready for prediction.
#' @export
#' @examples
#' model <- hf_load_model('facebook/bart-large-mnli')
#' model$task
#' model("Joe is eating a donut and enjoying himself.", c("happy", "neutral", "sad"))
#' @seealso
#' \url{https://huggingface.co/docs/transformers/main/en/pipeline_tutorial}
hf_load_model <- function(model_id,
                          tokenizer = NULL,
                          task = NULL){

  if(is.null(tokenizer)) hf_load_tokenizer(model_id)

  model <-
    hf_pipeline(model_id, tokenizer = tokenizer, task = task)

  message(glue::glue("\n\n{model_id} is ready for {model$task}", .trim = F))

  model
}


#' Load Tokenizer
#' Load Tokenizer for Huggingface Model
#'
#' @param model_id The id of the model given in the url by https://huggingface.co/model_name.
#' @returns A Huggingface model tokenizer.
#' @export
#' @examples
#' tokenizer <- hf_load_tokenizer('facebook/bart-large-mnli')
#' model <- hf_load_model('facebook/bart-large-mnli', tokenizer = tokenizer)
#' model("Joe is eating a donut and enjoying himself.", c("happy", "neutral", "sad"))
#' @seealso
#' \url{https://huggingface.co/docs/transformers/main/en/pipeline_tutorial}
hf_load_tokenizer <- function(model_id){

  hf_load_autotokenizer()

  tokenizer <- reticulate::py$AutoTokenizer$from_pretrained(model_id)

  tokenizer
}
