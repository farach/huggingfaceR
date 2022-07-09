
hf_pipeline <- function(model_id, tokenizer = NULL, task = NULL, config = NULL,
                        feature_extractor = NULL, framework = NULL, revision = NULL,
                        use_fast = NULL, use_auth_token = NULL, model_kwargs = NULL,
                        pipeline_class = NULL) {
  hf_load_pipeline()

  if (is.null(tokenizer)) tokenizer <- reticulate::py$AutoTokenizer$from_pretrained(model_id)

  reticulate::py$pipeline(task = task, model = model_id, tokenizer = tokenizer)
}


#' Load Model
#'
#' Load Model from Hugging Face
#'
#' @param model_id The id of the model given in the url by https://huggingface.co/model_name.
#' @param tokenizer The tokenizer function used to tokenize inputs. Defaults to NULL (one will be automatically loaded).
#' @param task The task the model will accomplish. Run hf_list_tasks() for options.
#' @param use_auth_token The token to use as HTTP bearer authorization for remote files. Unnecessary if HUGGING_FACE_HUB_TOKEN environment variable is set. If True, will use the token generated when running transformers-cli login (stored in ~/.huggingface).
#' @param ... Fed to the hf_pipeline function
#' @returns A Hugging Face model ready for prediction.
#' @export
#' @seealso
#' \url{https://huggingface.co/docs/transformers/main/en/pipeline_tutorial}
hf_load_model <- function(model_id,
                          tokenizer = NULL,
                          task = NULL,
                          use_auth_token = FALSE,
                          ...) {
  if (is.null(tokenizer)) hf_load_tokenizer(model_id)

  model <-
    hf_pipeline(model_id, tokenizer = tokenizer, task = task, use_auth_token = use_auth_token, ...)

  message(glue::glue("\n\n{model_id} is ready for {model$task}", .trim = FALSE))

  model
}


#' Load Tokenizer
#'
#' Load Tokenizer for Hugging Face Model
#'
#' @param model_id The id of the model given in the url by https://huggingface.co/model_name.
#' @returns A Hugging Face model tokenizer.
#' @export
#' @seealso
#' \url{https://huggingface.co/docs/transformers/main/en/pipeline_tutorial}
hf_load_tokenizer <- function(model_id) {
  hf_load_autotokenizer()

  tokenizer <- reticulate::py$AutoTokenizer$from_pretrained(model_id)

  tokenizer
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

# ß#' examples
# ß#' model <- hf_load_model('facebook/bart-large-mnli')
# ß#' model$task
# ß#' model("Joe is eating a donut and enjoying himself.", c("happy", "neutral", "sad"))


##' examples
##' tokenizer <- hf_load_tokenizer('facebook/bart-large-mnli')
##' model <- hf_load_model('facebook/bart-large-mnli', tokenizer = tokenizer)
##' labels <- c("happy", "neutral", "sad")
##' model("Joe is eating a donut and enjoying himself.", labels)
