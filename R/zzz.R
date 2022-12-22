# Using suggestions from https://rstudio.github.io/reticulate/articles/package.html
.onAttach <- function(libname, pkgname) {

  huggingface_env <- Sys.getenv("HUGGINGFACE_ENV")

  if (huggingface_env == "") {
    huggingface_env <- "huggingfaceR"
  }

  result <-
    tryCatch(
      {
        reticulate::use_miniconda(huggingface_env, required = TRUE)
      },
      error = function(e) {
        e
      }
    )

  if ("error" %in% class(result)) {
    if (result$message %>% stringr::str_detect("Miniconda is not installed")) {
      stop(result$message)
    }

    if (result$message %>% stringr::str_detect("Unable to locate conda environment")) {
      packageStartupMessage(glue::glue("\nCreating environment {huggingface_env}\n", .trim = FALSE))

      reticulate::conda_create(
        envname = huggingface_env,
        conda = paste0(reticulate::miniconda_path(), "/condabin/conda")
      )

      packageStartupMessage(glue::glue("\nSuccessfully created environment {huggingface_env}\n", .trim = FALSE))
    }

    if(!('transformers' %in% reticulate::py_list_packages(huggingface_env)$package %>% unique())){
      warning("Missing needed Python libraries. Run hf_python_depends() to install needed dependencies.")
    }
  }

  # Force reticulate to use huggingface python path.
  python_path <-
    reticulate::conda_list() %>%
    dplyr::filter(name == huggingface_env) %>%
    dplyr::pull(python)

  Sys.setenv(RETICULATE_PYTHON = python_path)

  reticulate::use_condaenv(condaenv = huggingface_env, required = TRUE)

  invisible()
}

#' Install Python Dependencies
#'
#' Installs python packages needed to run huggingfaceR functions
#' @param packages Python libraries needed for local model usage. \cr
#' Defaults to transformers, sentencepiece, huggingface_hub, datasets, and sentence-transformers.
#' @export
hf_python_depends <- function(packages = c("transformers",
                                           "sentencepiece",
                                           "huggingface_hub",
                                           "datasets",
                                           "sentence-transformers")){

  huggingface_env <- Sys.getenv("HUGGINGFACE_ENV")

  if (huggingface_env == "") {
    huggingface_env <- "huggingfaceR"
  }

  reticulate::conda_install(
    huggingface_env,
    packages = packages)
}


.onUnload <- function(libpath) {


}

# get the current python environment
get_current_python_environment <- function() {
  if (Sys.info()["sysname"] == "Windows") {
    reticulate::py_config()$python %>%
      stringr::str_extract(".*(?<=/huggingfaceR)")
  } else {
    paste0(
      "/",
      reticulate::py_config()$python %>%
        stringr::str_extract("/.*(?<=/bin/python$)") %>%
        stringr::str_remove_all("/bin/python") %>%
        stringr::str_remove("/")
    )
  }
}

# Loads the Huggingface API into memory.
hf_load_api <- function() {
  if (!"hf_api" %in% names(reticulate::py)) {
    result <-
      tryCatch(
        {
          reticulate::py_run_string("from huggingface_hub import HfApi")
          reticulate::py_run_string("hf_api = HfApi()")
        },
        error = function(e) e
      )

    if ("error" %in% class(result)) {
      if (result$message %>% stringr::str_detect("No module named")) {
        env <- get_current_python_environment()

        stop(glue::glue("\nMissing Python library! Run hf_python_depends('huggingface_hub') to install the missing library, or run hf_python_depends() to install all needed libraries.\n", .trim = FALSE))
      }
    }
  }

  T
}

# Loads the model search arguments into memory.
hf_load_model_args <- function() {
  if (!"model_args" %in% names(reticulate::py)) {
    result <-
      tryCatch(
        {
          reticulate::py_run_string("from huggingface_hub import ModelSearchArguments")
          reticulate::py_run_string("model_args = ModelSearchArguments()")
        },
        error = function(e) e
      )

    if ("error" %in% class(result)) {
      if (result$message %>% stringr::str_detect("No module named")) {
        env <- get_current_python_environment()

        stop(glue::glue("\nMissing Python library! Run hf_python_depends('huggingface_hub') to install the missing library, or run hf_python_depends() to install all needed libraries.\n", .trim = FALSE))
      }
    }
  }

  T
}

# Loads the model filter into memory.
hf_load_model_filter <- function() {
  if (!"ModelFilter" %in% names(reticulate::py)) {
    result <-
      tryCatch(
        {
          reticulate::py_run_string("from huggingface_hub import ModelFilter")
        },
        error = function(e) e
      )

    if ("error" %in% class(result)) {
      if (result$message %>% stringr::str_detect("No module named")) {
        env <- get_current_python_environment()

        stop(glue::glue("\nMissing Python library! Run hf_python_depends('huggingface_hub') to install the missing library, or run hf_python_depends() to install all needed libraries.\n", .trim = FALSE))
      }
    }
  }

  T
}

# List searchable model attributes
hf_list_model_attributes <- function() {
  stopifnot(hf_load_model_args())

  reticulate::py$model_args %>% names()
}

# Return all or a matched subset of values for a given attribute.
hf_list_attribute_options <- function(attribute, pattern = NULL, ignore_case = TRUE) {
  stopifnot(hf_load_model_args())

  vals <- reticulate::py$model_args[attribute]

  if (is.null(pattern)) {
    #  purrr::map_dfr(vals %>% names(), function(val) tibble(term = val , value = vals[val]))
    purrr::map_chr(vals %>% names(), function(val) vals[val])
  } else {
    #  purrr::map_dfr(vals %>% names() %>% stringr::str_subset(stringr::regex(pattern, ignore_case = T)), function(val) tibble(term = val , value = vals[val]))
    purrr::map_chr(vals %>% names() %>% stringr::str_subset(stringr::regex(pattern %>% stringr::str_replace_all("-", "."), ignore_case = ignore_case)), function(val) vals[val])
  }
}

# install and load AutoTokenizer from the transformers python library
hf_import_autotokenizer <- function() {
  if (!"AutoTokenizer" %in% names(reticulate::py)) {
    result <-
      tryCatch(
        {
          reticulate::py_run_string("from transformers import AutoTokenizer")
        },
        error = function(e) e
      )

    if ("error" %in% class(result)) {
      if (result$message %>% stringr::str_detect("No module named")) {
        env <- get_current_python_environment()

        stop(glue::glue("\nMissing Python library! Run hf_python_depends('transformers') to install the missing library, or run hf_python_depends() to install all needed libraries.\n", .trim = FALSE))
      }
    }
  }

  T
}

# install and load AutoModel from the transformers python library
hf_import_automodel <- function() {
  if (!"AutoModel" %in% names(reticulate::py)) {
    result <-
      tryCatch(
        {
          reticulate::py_run_string("from transformers import AutoModel")
        },
        error = function(e) e
      )

    if ("error" %in% class(result)) {
      if (result$message %>% stringr::str_detect("No module named")) {
        env <- get_current_python_environment()

        stop(glue::glue("\nMissing Python library! Run hf_python_depends('transformers') to install the missing library, or run hf_python_depends() to install all needed libraries.\n", .trim = FALSE))
      }
    }
  }

  T
}

# install and load load pipeline from the transformers python library
hf_import_pipeline <- function() {
  if (!"pipeline" %in% names(reticulate::py)) {
    result <-
      tryCatch(
        {
          reticulate::py_run_string("from transformers import pipeline")
        },
        error = function(e) e
      )

    if ("error" %in% class(result)) {
      if (result$message %>% stringr::str_detect("No module named")) {
        env <- get_current_python_environment()

        stop(glue::glue("\nMissing Python library! Run hf_python_depends('transformers') to install the missing library, or run hf_python_depends() to install all needed libraries.\n", .trim = FALSE))
      }
    }
  }

  T
}


# install and load load SentenceTransformer from the sentence_transformers python library
hf_import_sentence_transformers <- function() {
  if (!"sentence_transformer" %in% names(reticulate::py) || reticulate::py_is_null_xptr(reticulate::py$sentence_transformer)) {
    result <-
      tryCatch(
        {
          reticulate::py_run_string("from sentence_transformers import SentenceTransformer as sentence_transformer")
        },
        error = function(e) e
      )

    if ("error" %in% class(result)) {
      if (result$message %>% stringr::str_detect("No module named")) {
        env <- get_current_python_environment()

        stop(glue::glue("\nMissing Python library! Run hf_python_depends('sentence-transformers') to install the missing library, or run hf_python_depends() to install all needed libraries.\n", .trim = FALSE))
      }
    }
  }

  T
}

# install and load load_dataset from the datasets python library
hf_import_datasets_transformers <- function() {
  if (!"load_dataset" %in% names(reticulate::py) || reticulate::py_is_null_xptr(reticulate::py$load_dataset)) {
    result <-
      tryCatch(
        {
          reticulate::py_run_string("from datasets import load_dataset")
        },
        error = function(e) e
      )

    if ("error" %in% class(result)) {
      if (result$message %>% stringr::str_detect("No module named")) {
        env <- get_current_python_environment()

        stop(glue::glue("\nMissing Python library! Run hf_python_depends('datasets') to install the missing library, or run hf_python_depends() to install all needed libraries.\n", .trim = FALSE))
      }
    }
  }

  TRUE
}


#Import a specific type of AutoModel.
hf_import_AutoModel <- function(model_type = "AutoModelForSequenceClassification"){
  if(!paste0(model_type) %in% names(reticulate::py)){

    reticulate::py_run_string(paste0("from transformers import ", model_type))

  } else if (paste0(model_type) %in% names(reticulate::py)) {

    message(paste0(model_type, " was already imported, loading your model"))
  }


}


hf_stop_token_spam <- function(){
  reticulate::py_run_string("import os")
  reticulate::py_run_string("os.environ['TOKENIZERS_PARALLELISM'] = 'false'")
}
