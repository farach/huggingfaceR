
hf_pipeline <- function(model_id, tokenizer = NULL, task = NULL, ...) {
  hf_import_pipeline()

  if (is.null(tokenizer)) tokenizer <- reticulate::py$AutoTokenizer$from_pretrained(model_id)

  reticulate::py$pipeline(task = task, model = model_id, tokenizer = tokenizer, ...)
}


#' Load a pipeline object from Hugging Face - pipelines usually include a model, tokenizer and task.
#'
#' Load Model from Hugging Face
#'
#' @param model_id The id of the model given in the url by https://huggingface.co/model_name.
#' @param tokenizer The tokenizer function used to tokenize inputs. Defaults to NULL (one will be automatically loaded).
#' @param task The task the model will accomplish. Run hf_list_tasks() for options.
#' @param ... Fed to the hf_pipeline function
#'
#' @returns A Hugging Face model ready for prediction.
#' @export
#' @seealso
#' \url{https://huggingface.co/docs/transformers/main/en/pipeline_tutorial}
hf_load_pipeline <- function(model_id,
                          tokenizer = NULL,
                          task = NULL, ...) {
  if (is.null(tokenizer)) tokenizer <- hf_load_tokenizer(model_id)

  pipeline <-
    hf_pipeline(model_id, tokenizer = tokenizer, task = task, ...)

  message(glue::glue("\n\n{model_id} is ready for {pipeline$task}", .trim = FALSE))

  return(pipeline)
}


#' Load an AutoTokenizer from a pre-tained model
#'
#' Load Tokenizer for Hugging Face Model
#'
#' @param model_id The id of the model given in the url by https://huggingface.co/model_name.
#' @param ... sent to the `AutoTokenizer.from_pretained()`, accepts named arguments e.g. use_fast \url{https://huggingface.co/docs/transformers/main_classes/tokenizer}
#'
#' @returns A Hugging Face model's tokenizer.
#' @export
#' @seealso
#' \url{https://huggingface.co/docs/transformers/main/en/pipeline_tutorial}
hf_load_tokenizer <- function(model_id, ...) {
  hf_import_autotokenizer()

  tokenizer <- reticulate::py$AutoTokenizer$from_pretrained(model_id, ...)

  return(tokenizer)
}


#' Try to set device to GPU for accelerated computation
#'
#' This function currently depends on having a working installation of torch for your GPU in this environment.
#' If running an Apple silicon GPU, you'll need the native mac M+ build (ARM binary). You will also need rust and other transformers dependencies.
#' As you need to make sure that everything that needs to be on the GPU (tensors, model, pipeline etc.), is on the GPU, we currently recommend this for advanced users only.
#' We will be working on integrating this fully with the installation and build of the huggingfaceR environment.
#'
#' @return a device that models, pipelines, and tensors can be sent to.
#' @export
#'
#' @examples
#' \dontrun{
#' device <- hf_set_device()
#' }
hf_set_device <- function(){

  result <-
    tryCatch({
      #Check that torch is imported, if not report an error - can re-work this to cater for ARM & x86 builds
      if(!"torch" %in% names(reticulate::py)){
        return("Attempt to set device failed. This function requires torch to be loaded.")
      }

    }, error = function(e) e)

  if ("error" %in% class(result)){
    return("You'll need a working version of torch in your environment to run this function.")
  } else if (reticulate::py$torch$has_cuda){
    return( reticulate::py$torch$device('cuda'))
  } else if (
    #Check we're on arm 64 machine, and that torch version has mps
    Sys.info()["machine"] == "arm64" &
    "has_mps" %in% names(reticulate::py$torch) &
    reticulate::py$torch$has_mps){
    return(reticulate::py$torch$device('mps'))

  } else {
    message("device is being set to CPU because neither CUDA nor MPS were detected")
    return(reticulate::py$torch$device('cpu'))}

}


#' Load a pre-trained AutoModel object from Hugging Face
#'
#' @param model_id model_id The id of the model given in the url by https://huggingface.co/model_name.
#' @param ... sent to AutoModel.from_pretrained()
#'
#' @return a pre-trained model object
#' @export
#'
#' @examples
#' \dontrun{
#' model <- hf_load_model("distilbert-base-uncased")
#' }
hf_load_model <- function(model_id, ...){
  hf_import_automodel()

  model <- reticulate::py$AutoModel$from_pretrained(model_id, ...)

  message(glue::glue("\n\n{model_id} is ready", .trim = FALSE))
  return(model)
}


#' Import a pre-trained model for a specific task.
#'
#' Function differs from `hf_load_model` in that `hf_load_model` currently only loads AutoModels i.e. not AutoModelsForX
#'
#' @param model_type The AutoModel's type passed as a string e.g. c("AutoModelForQuestionAnswering", "AutoModelForTokenClassification", "AutoModelForSequenceClassification")
#' @param model_id The model's name or id on the Hugging Face hub
#' @param use_auth_token For private models, copy and paste your auth token in as a string.
#'
#' @return an AutoMNodel object for a specific task
#' @export
#'
#' @seealso
#' \url{https://huggingface.co/transformers/v3.0.2/model_doc/auto.html}
hf_load_AutoModel <- function(model_type = "AutoModelForSequenceClassification", model_id, use_auth_token = NULL){
  hf_import_AutoModel(model_type)

  string_to_run <- paste0("reticulate::py$", model_type, "$from_pretrained('", model_id, "', use_auth_token ='", use_auth_token, "')")

  model <- eval(parse(text = string_to_run))

  return(model)


}



# ß#' examples
# ß#' model <- hf_load_model('facebook/bart-large-mnli')
# ß#' model$task
# ß#' model("Joe is eating a donut and enjoying himself.", c("happy", "neutral", "sad"))


##' examples
##' tokenizer <- hf_load_tokenizer('facebook/bart-large-mnli')
##' model <- hf_load_model('facebook/bart-large-mnli', tokenizer = tokenizer)
##' labels <- c("happy", "neutral", "sad")
##' model("Joe is eating a donut and enjoying himself.", labels)
