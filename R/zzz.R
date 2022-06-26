# Using suggestions from https://rstudio.github.io/reticulate/articles/package.html
.onLoad <- function(libname, pkgname) {

  huggingface_env <- Sys.getenv('HUGGINGFACE_ENV')

  if(huggingface_env == ''){
    huggingface_env <- 'huggingfaceR'
  }

  result <-
    tryCatch({

      reticulate::use_miniconda(huggingface_env, required = T)

    }, error = function(e){
      e
    })

  if('error' %in% class(result)){

    if(result$message %>% stringr::str_detect('Miniconda is not installed')){
      stop(result$message)
    }

    if(result$message %>% stringr::str_detect('Unable to locate conda environment')){

      message(glue::glue("\nCreating environment {huggingface_env}\n", .trim = F))

      reticulate::conda_create(
        envname = huggingface_env,
        packages = c("PyTorch", "Tensorflow", "transformers", "sentencepiece", "huggingface_hub"),
        conda = paste0(reticulate::miniconda_path(), "/condabin/conda")
      )

      message(glue::glue("\nSuccessfully created environment {huggingface_env}\n", .trim = F))
    }
  }

  # Force reticulate to use huggingface python path.
  python_path <-
    reticulate::conda_list() %>%
    dplyr::filter(name == huggingface_env) %>%
    dplyr::pull(python)

  Sys.setenv(RETICULATE_PYTHON = python_path)

  reticulate::use_condaenv(condaenv = huggingface_env, required = T)

  invisible()
}

.onUnload <- function(libpath){


}



get_current_python_environment <- function(){
  reticulate::py_config()$python %>%
    stringr::str_extract('/.*(?<=/bin/python$)') %>%
    stringr::str_remove_all('/bin/python') %>%
    stringr::str_remove('/')
}

# Loads the Huggingface API into memory.
hf_load_api <- function(){

  if(!'hf_api' %in% names(reticulate::py)){
    result <-
      tryCatch({
        reticulate::py_run_string("from huggingface_hub import HfApi")
        reticulate::py_run_string("hf_api = HfApi()")
      }, error = function(e) e)

    if('error' %in% class(result)){

      if(result$message %>% stringr::str_detect('No module named')){

        env <- get_current_python_environment()

        message(glue::glue("\nInstalling needed Python library huggingface_hub into env {env}\n", .trim = F))
        Sys.sleep(1)
        reticulate::py_install(packages = 'huggingface_hub', envname = env)

        reticulate::py_run_string("from huggingface_hub import HfApi")
        reticulate::py_run_string("hf_api = HfApi()")
      }
    }
  }

  T
}

# Loads the model search arguments into memory.
hf_load_model_args <- function(){

  if(!'model_args' %in% names(reticulate::py)){
    result <-
      tryCatch({
        reticulate::py_run_string("from huggingface_hub import ModelSearchArguments")
        reticulate::py_run_string("model_args = ModelSearchArguments()")
      }, error = function(e) e)

    if('error' %in% class(result)){

      if(result$message %>% stringr::str_detect('No module named')){

        env <- get_current_python_environment()

        message(glue::glue("\nInstalling needed Python library huggingface_hub into env {env}\n", .trim = F))
        reticulate::py_install(packages = 'huggingface_hub', envname = env)

        reticulate::py_run_string("from huggingface_hub import ModelSearchArguments")
        reticulate::py_run_string("model_args = ModelSearchArguments()")
      }
    }
  }

  T
}

# Loads the model filter into memory.
hf_load_model_filter <- function(){

  if(!'ModelFilter' %in% names(reticulate::py)){

    result <-
      tryCatch({
        reticulate::py_run_string("from huggingface_hub import ModelFilter")
      }, error = function(e) e)

    if('error' %in% class(result)){

      if(result$message %>% stringr::str_detect('No module named')){

        env <- get_current_python_environment()

        message(glue::glue("\nInstalling needed Python library huggingface_hub into env {env}\n", .trim = F))
        reticulate::py_install(packages = 'huggingface_hub', envname = env)

        reticulate::py_run_string("from huggingface_hub import ModelFilter")
      }
    }
  }

  T
}

# List searchable model attributes
hf_list_model_attributes <- function(){

  stopifnot(hf_load_model_args())

  reticulate::py$model_args %>% names()
}

# Return all or a matched subset of values for a given attribute.
hf_list_attribute_options <- function(attribute, pattern = NULL, ignore_case = T){

  stopifnot(hf_load_model_args())

  vals <- reticulate::py$model_args[attribute]

  if(is.null(pattern)){
    #  purrr::map_dfr(vals %>% names(), function(val) tibble(term = val , value = vals[val]))
    purrr::map_chr(vals %>% names(), function(val) vals[val])
  }else{
    #  purrr::map_dfr(vals %>% names() %>% stringr::str_subset(stringr::regex(pattern, ignore_case = T)), function(val) tibble(term = val , value = vals[val]))
    purrr::map_chr(vals %>% names() %>% stringr::str_subset(stringr::regex(pattern %>% stringr::str_replace_all("-", "."), ignore_case = ignore_case)), function(val) vals[val])
  }
}

hf_load_autotokenizer <- function(){

  if(!'AutoTokenizer' %in% names(reticulate::py)){

    result <-
      tryCatch({
        reticulate::py_run_string('from transformers import AutoTokenizer')
      }, error = function(e) e)

    if('error' %in% class(result)){

      if(result$message %>% stringr::str_detect('No module named')){

        env <- get_current_python_environment()

        message(glue::glue("\nInstalling needed Python library transformers into env {env}\n", .trim = F))
        reticulate::py_install(packages = 'transformers', envname = env)

        reticulate::py_run_string('from transformers import AutoTokenizer')
      }
    }
  }

  T
}



hf_load_pipeline <- function(){

  if(!'pipeline' %in% names(reticulate::py)){

    result <-
      tryCatch({
        reticulate::py_run_string('from transformers import pipeline')
      }, error = function(e) e)

    if('error' %in% class(result)){

      if(result$message %>% stringr::str_detect('No module named')){

        env <- get_current_python_environment()

        message(glue::glue("\nInstalling needed Python library transformers into env {env}\n", .trim = F))
        reticulate::py_install(packages = 'transformers', envname = env)

        reticulate::py_run_string('from transformers import pipeline')
      }
    }
  }

  T
}


# Installs and loads sentence-transformers
hf_load_sentence_transformers <- function(){

  if(!'sentence_transformer' %in% names(reticulate::py) || reticulate::py_is_null_xptr(reticulate::py$sentence_transformer)){
    result <-
      tryCatch({
        reticulate::py_run_string("from sentence_transformers import SentenceTransformer as sentence_transformer")
      }, error = function(e) e)

    if('error' %in% class(result)){

      if(result$message %>% stringr::str_detect('No module named')){

        env <- get_current_python_environment()

        message(glue::glue("\nInstalling needed Python library sentence-transformers into env {env}\n", .trim = F))
        Sys.sleep(1)
        reticulate::py_install(packages = 'sentence-transformers', envname = env)

        reticulate::py_run_string("from sentence_transformers import SentenceTransformer as sentence_transformer")
      }
    }
  }

  T
}
