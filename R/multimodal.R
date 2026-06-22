#' Transcribe Audio
#'
#' Transcribe speech from an audio file, URL, or raw vector using automatic
#' speech recognition via the Hugging Face Inference Providers API.
#'
#' @param audio Audio input: a local file path, URL, raw vector, or vector/list of
#'   paths/URLs.
#' @param return_timestamps Logical or character. Use `FALSE` for text only,
#'   `TRUE` for chunk timestamps, or a model-supported value such as `"word"`.
#' @param model Character string. Model ID from Hugging Face Hub. Default:
#'   "openai/whisper-large-v3".
#' @param token Character string or NULL. API token for authentication.
#' @param endpoint_url Character string or NULL. A custom Inference Endpoint URL.
#' @param content_type Character string or NULL. MIME type to use for raw audio
#'   inputs. Paths and URLs are inferred when possible.
#' @param ... Additional arguments (currently unused).
#'
#' @returns A tibble with columns: audio, text, chunks.
#' @export
#' @seealso \url{https://huggingface.co/docs/inference-providers/tasks/automatic-speech-recognition}
#'
#' @examples
#' \dontrun{
#' hf_transcribe("interview.flac")
#' hf_transcribe("interview.flac", return_timestamps = "word")
#' }
hf_transcribe <- function(audio,
                          return_timestamps = FALSE,
                          model = hf_default_model("transcribe"),
                          token = NULL,
                          endpoint_url = NULL,
                          content_type = NULL,
                          ...) {
  audios <- hf_as_input_list(audio)
  if (length(audios) == 0) {
    return(tibble::tibble(audio = character(), text = character(), chunks = list()))
  }

  query <- if (isFALSE(return_timestamps)) {
    NULL
  } else {
    list(return_timestamps = if (isTRUE(return_timestamps)) "true" else return_timestamps)
  }

  purrr::map_dfr(audios, function(single_audio) {
    if (hf_is_missing_media(single_audio)) {
      return(tibble::tibble(audio = NA_character_, text = NA_character_, chunks = list(NULL)))
    }

    result <- hf_binary_task_request(
      model = model,
      input = single_audio,
      token = token,
      endpoint_url = endpoint_url,
      content_type = content_type,
      query = query
    )

    tibble::tibble(
      audio = hf_media_label(single_audio),
      text = result$text %||% NA_character_,
      chunks = list(result$chunks %||% list())
    )
  })
}


#' Convert Text to Speech
#'
#' Generate speech audio from text and write it to disk. The public
#' `hf-inference` provider did not expose a broadly available TTS model during
#' verification; use this with a compatible model/provider or dedicated
#' Inference Endpoint.
#'
#' @param text Character vector of text to synthesize.
#' @param output Character path(s) or NULL. When NULL, files are written to
#'   temporary paths with an extension inferred from the response content type.
#' @param model Character string. Model ID from Hugging Face Hub. Default:
#'   "facebook/mms-tts-eng".
#' @param token Character string or NULL. API token for authentication.
#' @param endpoint_url Character string or NULL. A custom Inference Endpoint URL.
#' @param overwrite Logical. If TRUE, overwrite existing output files.
#' @param ... Additional generation parameters passed to the model.
#'
#' @returns A tibble with columns: text, path, content_type, audio.
#' @export
#' @seealso \url{https://huggingface.co/docs/inference-providers/tasks/text-to-speech}
#'
#' @examples
#' \dontrun{
#' hf_text_to_speech("Hello from R.")
#' }
hf_text_to_speech <- function(text,
                              output = NULL,
                              model = hf_default_model("text_to_speech"),
                              token = NULL,
                              endpoint_url = NULL,
                              overwrite = FALSE,
                              ...) {
  hf_generate_binary_outputs(
    inputs = text,
    input_name = "text",
    raw_name = "audio",
    output = output,
    default_ext = ".wav",
    model = model,
    token = token,
    endpoint_url = endpoint_url,
    overwrite = overwrite,
    parameters = list(...)
  )
}


#' Generate an Image from Text
#'
#' Generate an image from a prompt and write it to disk using a text-to-image
#' model via the Hugging Face Inference Providers API.
#'
#' @param prompt Character vector of prompts.
#' @param output Character path(s) or NULL. When NULL, files are written to
#'   temporary paths with an extension inferred from the response content type.
#' @param seed Integer or NULL. Optional random seed for reproducibility when the
#'   provider/model supports it.
#' @param model Character string. Model ID from Hugging Face Hub. Default:
#'   "black-forest-labs/FLUX.1-schnell".
#' @param token Character string or NULL. API token for authentication.
#' @param endpoint_url Character string or NULL. A custom Inference Endpoint URL.
#' @param overwrite Logical. If TRUE, overwrite existing output files.
#' @param ... Additional generation parameters passed to the model.
#'
#' @returns A tibble with columns: prompt, path, content_type, image.
#' @export
#' @seealso \url{https://huggingface.co/docs/inference-providers/tasks/text-to-image}
#'
#' @examples
#' \dontrun{
#' img <- hf_text_to_image("a small red cube on a white background", seed = 42)
#' img$path
#' }
hf_text_to_image <- function(prompt,
                             output = NULL,
                             seed = NULL,
                             model = hf_default_model("text_to_image"),
                             token = NULL,
                             endpoint_url = NULL,
                             overwrite = FALSE,
                             ...) {
  hf_generate_binary_outputs(
    inputs = prompt,
    input_name = "prompt",
    raw_name = "image",
    output = output,
    default_ext = ".jpg",
    model = model,
    token = token,
    endpoint_url = endpoint_url,
    overwrite = overwrite,
    parameters = c(list(seed = seed), list(...))
  )
}


#' Classify Images
#'
#' Classify images using an image-classification model via the Hugging Face
#' Inference Providers API.
#'
#' @param image Image input: a local file path, URL, raw vector, or vector/list of
#'   paths/URLs.
#' @param top_k Integer. Maximum labels to return per image.
#' @param model Character string. Model ID from Hugging Face Hub. Default:
#'   "google/vit-base-patch16-224".
#' @param token Character string or NULL. API token for authentication.
#' @param endpoint_url Character string or NULL. A custom Inference Endpoint URL.
#' @param content_type Character string or NULL. MIME type to use for raw image
#'   inputs. Paths and URLs are inferred when possible.
#' @param ... Additional arguments (currently unused).
#'
#' @returns A tibble with columns: image, label, score.
#' @export
#' @seealso \url{https://huggingface.co/docs/inference-providers/tasks/image-classification}
#'
#' @examples
#' \dontrun{
#' hf_classify_image("cat.png", top_k = 3)
#' }
hf_classify_image <- function(image,
                              top_k = 5L,
                              model = hf_default_model("classify_image"),
                              token = NULL,
                              endpoint_url = NULL,
                              content_type = NULL,
                              ...) {
  images <- hf_as_input_list(image)
  if (length(images) == 0) {
    return(tibble::tibble(image = character(), label = character(), score = numeric()))
  }

  purrr::map_dfr(images, function(single_image) {
    if (hf_is_missing_media(single_image)) {
      return(tibble::tibble(image = NA_character_, label = NA_character_, score = NA_real_))
    }

    result <- hf_binary_task_request(
      model = model,
      input = single_image,
      token = token,
      endpoint_url = endpoint_url,
      content_type = content_type
    )
    preds <- utils::head(result, top_k)
    tibble::tibble(
      image = hf_media_label(single_image),
      label = vapply(preds, function(x) x$label %||% NA_character_, character(1)),
      score = vapply(preds, function(x) x$score %||% NA_real_, numeric(1))
    )
  })
}


#' Caption Images
#'
#' Generate short image captions. This uses a vision-capable chat model by
#' default because the public `hf-inference` provider did not expose a broadly
#' available image-to-text captioning model during verification.
#'
#' @param image Image input: a local file path, URL, raw vector, or vector/list of
#'   paths/URLs.
#' @param prompt Prompt used to request the caption.
#' @param model Character string. Vision-capable chat model ID. Default:
#'   "google/gemma-3-4b-it".
#' @param max_tokens Integer. Maximum tokens to generate.
#' @param token Character string or NULL. API token for authentication.
#' @param endpoint_url Character string or NULL. A custom Inference Endpoint URL.
#' @param ... Additional arguments passed to \code{hf_describe_image()}.
#'
#' @returns A tibble with columns: image, caption.
#' @export
#' @seealso \code{\link{hf_describe_image}}
#'
#' @examples
#' \dontrun{
#' hf_caption_image("cat.png")
#' }
hf_caption_image <- function(image,
                             prompt = "Write a short, factual caption for this image.",
                             model = hf_default_model("caption_image"),
                             max_tokens = 80,
                             token = NULL,
                             endpoint_url = NULL,
                             ...) {
  result <- hf_describe_image(
    image = image,
    prompt = prompt,
    model = model,
    max_tokens = max_tokens,
    token = token,
    endpoint_url = endpoint_url,
    ...
  )
  dplyr::rename(result, caption = "description")
}


#' Detect Objects in Images
#'
#' Detect objects and bounding boxes in images using an object-detection model
#' via the Hugging Face Inference Providers API.
#'
#' @param image Image input: a local file path, URL, raw vector, or vector/list of
#'   paths/URLs.
#' @param threshold Numeric or NULL. Optional minimum confidence score to keep.
#' @param model Character string. Model ID from Hugging Face Hub. Default:
#'   "facebook/detr-resnet-50".
#' @param token Character string or NULL. API token for authentication.
#' @param endpoint_url Character string or NULL. A custom Inference Endpoint URL.
#' @param content_type Character string or NULL. MIME type to use for raw image
#'   inputs. Paths and URLs are inferred when possible.
#' @param ... Additional arguments (currently unused).
#'
#' @returns A tibble with columns: image, label, score, xmin, ymin, xmax, ymax.
#' @export
#' @seealso \url{https://huggingface.co/docs/inference-providers/tasks/object-detection}
#'
#' @examples
#' \dontrun{
#' boxes <- hf_detect_objects("cat.png", threshold = 0.5)
#' boxes
#' }
hf_detect_objects <- function(image,
                              threshold = NULL,
                              model = hf_default_model("detect_objects"),
                              token = NULL,
                              endpoint_url = NULL,
                              content_type = NULL,
                              ...) {
  images <- hf_as_input_list(image)
  if (length(images) == 0) {
    return(hf_empty_object_detection())
  }

  purrr::map_dfr(images, function(single_image) {
    if (hf_is_missing_media(single_image)) {
      return(tibble::tibble(
        image = NA_character_,
        label = NA_character_,
        score = NA_real_,
        xmin = NA_real_,
        ymin = NA_real_,
        xmax = NA_real_,
        ymax = NA_real_
      ))
    }

    result <- hf_binary_task_request(
      model = model,
      input = single_image,
      token = token,
      endpoint_url = endpoint_url,
      content_type = content_type
    )
    if (!is.null(threshold)) {
      result <- result[vapply(result, function(x) (x$score %||% 0) >= threshold, logical(1))]
    }
    if (length(result) == 0) {
      return(hf_empty_object_detection())
    }

    tibble::tibble(
      image = hf_media_label(single_image),
      label = vapply(result, function(x) x$label %||% NA_character_, character(1)),
      score = vapply(result, function(x) x$score %||% NA_real_, numeric(1)),
      xmin = vapply(result, function(x) x$box$xmin %||% NA_real_, numeric(1)),
      ymin = vapply(result, function(x) x$box$ymin %||% NA_real_, numeric(1)),
      xmax = vapply(result, function(x) x$box$xmax %||% NA_real_, numeric(1)),
      ymax = vapply(result, function(x) x$box$ymax %||% NA_real_, numeric(1))
    )
  })
}


hf_generate_binary_outputs <- function(inputs, input_name, raw_name, output,
                                       default_ext, model, token, endpoint_url,
                                       overwrite, parameters) {
  if (length(inputs) == 0) {
    out <- tibble::tibble(
      path = character(),
      content_type = character()
    )
    out[[input_name]] <- character()
    out[[raw_name]] <- list()
    return(out[c(input_name, "path", "content_type", raw_name)])
  }
  if (!is.character(inputs)) {
    stop("`", input_name, "` must be a character vector.", call. = FALSE)
  }
  hf_validate_output_paths(output, length(inputs))

  purrr::map2_dfr(inputs, seq_along(inputs), function(single_input, i) {
    if (is.na(single_input)) {
      row <- tibble::tibble(path = NA_character_, content_type = NA_character_)
      row[[input_name]] <- single_input
      row[[raw_name]] <- list(NULL)
      return(row[c(input_name, "path", "content_type", raw_name)])
    }

    binary <- hf_binary_generation_request(
      model = model,
      inputs = single_input,
      parameters = parameters,
      token = token,
      endpoint_url = endpoint_url
    )
    path <- hf_write_binary_output(
      binary = binary,
      output = output,
      index = i,
      n = length(inputs),
      default_ext = default_ext,
      overwrite = overwrite
    )
    row <- tibble::tibble(path = path, content_type = binary$content_type)
    row[[input_name]] <- single_input
    row[[raw_name]] <- list(binary$raw)
    row[c(input_name, "path", "content_type", raw_name)]
  })
}


hf_as_input_list <- function(input) {
  if (is.raw(input)) list(input) else as.list(input)
}


hf_is_missing_media <- function(input) {
  is.character(input) && length(input) == 1L && is.na(input)
}


hf_media_label <- function(input) {
  if (is.raw(input)) "<raw>" else as.character(input)
}


hf_media_input <- function(input, content_type = NULL) {
  if (is.character(input) && length(input) == 1L &&
      grepl("^https?://", input, ignore.case = TRUE)) {
    resp <- httr2::request(input) |>
      httr2::req_perform()
    return(list(
      raw = httr2::resp_body_raw(resp),
      content_type = content_type %||%
        hf_clean_content_type(httr2::resp_header(resp, "content-type") %||%
                                "application/octet-stream")
    ))
  }

  if (is.character(input) && length(input) == 1L) {
    if (!file.exists(input)) {
      stop("Media file not found: ", input, call. = FALSE)
    }
    return(list(
      raw = readBin(input, what = "raw", n = file.info(input)$size),
      content_type = content_type %||% hf_media_content_type(input)
    ))
  }

  if (is.raw(input)) {
    return(list(
      raw = input,
      content_type = content_type %||% "application/octet-stream"
    ))
  }

  stop("Media input must be a URL, local file path, or raw vector.", call. = FALSE)
}


hf_media_content_type <- function(path) {
  ext <- tolower(tools::file_ext(path))
  switch(ext,
    jpg = "image/jpeg",
    jpeg = "image/jpeg",
    png = "image/png",
    gif = "image/gif",
    webp = "image/webp",
    wav = "audio/wav",
    wave = "audio/wav",
    flac = "audio/flac",
    mp3 = "audio/mpeg",
    m4a = "audio/mp4",
    ogg = "audio/ogg",
    opus = "audio/ogg",
    "application/octet-stream"
  )
}


hf_clean_content_type <- function(content_type) {
  sub(";.*$", "", content_type)
}


hf_validate_output_paths <- function(output, n) {
  if (is.null(output)) {
    return(invisible(NULL))
  }
  if (!is.character(output) || !length(output) %in% c(1L, n) || anyNA(output)) {
    stop("`output` must be NULL, one path, or one path per input.", call. = FALSE)
  }
  invisible(NULL)
}


hf_output_path <- function(output, index, n, content_type, default_ext) {
  ext <- hf_extension_from_content_type(content_type) %||% default_ext
  if (is.null(output)) {
    return(tempfile(fileext = ext))
  }
  if (length(output) == 1L && n == 1L) {
    return(output)
  }
  output[[index]]
}


hf_write_binary_output <- function(binary, output, index, n, default_ext,
                                   overwrite) {
  path <- hf_output_path(output, index, n, binary$content_type, default_ext)
  if (file.exists(path) && !isTRUE(overwrite)) {
    stop("Output file already exists: ", path, call. = FALSE)
  }
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  writeBin(binary$raw, path)
  normalizePath(path, winslash = "/", mustWork = FALSE)
}


hf_extension_from_content_type <- function(content_type) {
  switch(hf_clean_content_type(content_type),
    "image/jpeg" = ".jpg",
    "image/png" = ".png",
    "image/gif" = ".gif",
    "image/webp" = ".webp",
    "audio/wav" = ".wav",
    "audio/wave" = ".wav",
    "audio/x-wav" = ".wav",
    "audio/mpeg" = ".mp3",
    "audio/mp4" = ".m4a",
    "audio/flac" = ".flac",
    "audio/x-flac" = ".flac",
    NULL
  )
}


hf_empty_object_detection <- function() {
  tibble::tibble(
    image = character(),
    label = character(),
    score = numeric(),
    xmin = numeric(),
    ymin = numeric(),
    xmax = numeric(),
    ymax = numeric()
  )
}
