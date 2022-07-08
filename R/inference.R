
#' Predict with Model
#'
#' Predict using Hugging Face Model. If a model_id is provided, the Inference API will be used to make the prediction.
#' If you wish to download a model rather than running your predictions through the Inference API, download the model first with the hf_load_model() function.
#'
#' @param model Either a downloaded model from the Hugging Face Hub (using hf_load_model()) or a model_id.
#' @param inputs The data to predict on.
#' @param parameters Model parameters distinct to the model being used.
#' @param use_gpu API Only - Whether to use GPU for inference.
#' @param use_cache API Only - Whether to use cached inference results for previously seen inputs.
#' @param wait_for_model API Only - Whether to wait for the model to be ready instead of receiving a 503 error after a certain amount of time.
#' @param use_auth_token API Only - The token to use as HTTP bearer authorization for the Inference API. Defaults to HUGGING_FACE_HUB_TOKEN environment variable.
#' @returns A Hugging Face model prediction.
#' @export
#' @seealso
#' \url{https://huggingface.co/docs/api-inference/index}
hf_predict <- function(model, inputs, parameters = NULL, use_gpu = FALSE, use_cache = FALSE, wait_for_model = FALSE, use_auth_token = NULL, ...) {

  # If model is a model_id, use Inference API
  if (is.character(model)) {
    if (is.null(use_auth_token) && Sys.getenv("HUGGING_FACE_HUB_TOKEN") != "") use_auth_token <- Sys.getenv("HUGGING_FACE_HUB_TOKEN")

    response <-
      httr2::request(glue::glue("https://api-inference.huggingface.co/models/{model}")) %>%
      httr2::req_auth_bearer_token(token = use_auth_token) %>%
      httr2::req_body_json(
        list(
          inputs = inputs,
          parameters = parameters,
          options = list(
            use_gpu = use_gpu,
            use_cache = use_cache,
            wait_for_model = wait_for_model
          )
        )
      ) %>%
      httr2::req_perform()

    if (response$status_code < 300) {
      result <-
        response %>%
        httr2::resp_body_json(auto_unbox = TRUE) %>%
        jsonlite::toJSON(auto_unbox = TRUE) %>%
        jsonlite::fromJSON()
    }
  } else {

    # If local model object is passed in to model, perform local inference.
    if ("python.builtin.object" %in% class(model)) {
      result <-
        model(inputs, parameters, ...)
    } else {
      stop("model must be a downloaded Huggingface model or model_id")
    }
  }

  result
}
