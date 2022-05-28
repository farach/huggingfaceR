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

  reticulate::use_condaenv(condaenv = huggingface_env, required = T)

  invisible()
}
