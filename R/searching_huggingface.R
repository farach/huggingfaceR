# Loads the Huggingface API into memory.
hf_load_api <- function(){

  if(!'HfApi' %in% names(reticulate::py)){
    reticulate::py_run_string("from huggingface_hub import HfApi")
    reticulate::py_run_string("hf_api = HfApi()")
  }

  T
}

# Loads the model search arguments into memory.
hf_load_model_args <- function(){

  if(!'ModelSearchArguments' %in% names(reticulate::py)){
    reticulate::py_run_string("from huggingface_hub import ModelSearchArguments")
    reticulate::py_run_string("model_args = ModelSearchArguments()")
  }

  T
}

# Loads the model filter into memory.
hf_load_model_filter <- function(){

  if(!'ModelFilter' %in% names(reticulate::py)){
    reticulate::py_run_string("from huggingface_hub import ModelFilter")
  }

  T
}

# List possible
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

#' List Authors
#'
#' List Model Authors
#' @param pattern A search term or regular expression. Defaults to NULL (return all results).
#' @export
#'
#' @examples
#' hf_list_authors(pattern = '^sam')
hf_list_authors <- function(pattern = NULL){

  hf_list_attribute_options('author', pattern)
}

#' List Datasets
#'
#' List Model Datasets
#' @param pattern A search term or regular expression. Defaults to NULL (return all results).
#' @export
#'
#' @examples
#' hf_list_datasets('imdb')
hf_list_datasets <- function(pattern = NULL){

  hf_list_attribute_options('dataset', pattern)
}

#' List Languages
#'
#' List Model Languages
#' @param pattern A search term or regular expression. Defaults to NULL (return all results).
#' @export
#'
#' @examples
#' hf_list_languages('es')
hf_list_languages <- function(pattern = NULL){

  hf_list_attribute_options('language', pattern)
}

#' List Libraries
#'
#' List Model Libraries
#' @param pattern A search term or regular expression. Defaults to NULL (return all results).
#' @export
#'
#' @examples
#' hf_list_libraries('pytorch|tensorflow')
hf_list_libraries<- function(pattern = NULL){

  hf_list_attribute_options('library', pattern)
}

#' List Licenses
#'
#' List Model Licenses
#' @param pattern A search term or regular expression. Defaults to NULL (return all results).
#' @export
#'
#' @examples
#' hf_list_licenses('mit')
hf_list_licenses <- function(pattern = NULL){

  hf_list_attribute_options('license', pattern)
}

#' List Models
#'
#' List Model Names
#' @param pattern A search term or regular expression. Defaults to NULL (return all results).
#' @export
#'
#' @examples
#' hf_list_models('bert-base-cased')
hf_list_models <- function(pattern = NULL){

  hf_list_attribute_options('model_name', pattern)
}

#' List Tasks
#'
#' List Model Tasks
#' @param pattern A search term or regular expression. Defaults to NULL (return all results).
#' @export
#'
#' @examples
#' hf_list_tasks('^image.*tion')
hf_list_tasks <- function(pattern = NULL){

  hf_list_attribute_options('pipeline_tag', pattern)
}

#' Search Models
#'
#' Search Huggingface Models
#' @param author Filter by model author.
#' @param language Filter by the languages the model accommodates.
#' @param library Filter by the deep learning libraries which work with the model.
#' @param name Filter by model names.
#' @param tags Filter by model tags.
#' @param task Filter by tasks the model can accomplish.
#' @param dataset Filter by the datasets the model was trained on.
#' @export
#'
#' @examples
#' hf_search_models(library = "pytorch", dataset = 'mnli')
#' hf_search_models(author = "facebook", name = 'bart')
hf_search_models <- function(author = NULL, language = NULL, library = NULL, name = NULL, tags = NULL, task = NULL, dataset = NULL){

  stopifnot(hf_load_model_filter())

  model_filter <-
    reticulate::py$ModelFilter(author = author, language = language, library = library, model_name = name, tags = tags, task = task, trained_dataset = dataset)

  stopifnot(hf_load_api())

  reticulate::py$hf_api$list_models(filter = model_filter)
}
