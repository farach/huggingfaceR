#' Embedding Recipe Step
#'
#' Create text embeddings using a Hugging Face model as part of a tidymodels recipe.
#' This step converts text columns into embedding features for downstream modeling.
#'
#' @param recipe A recipe object.
#' @param ... One or more text column selectors (see recipes::selections()).
#' @param role Character string. Role for the new embedding variables. Default: "predictor".
#' @param trained Logical. Internal use only.
#' @param model Character string. Hugging Face model ID for embeddings.
#'   Default: "BAAI/bge-small-en-v1.5".
#' @param token Character string or NULL. API token for authentication.
#' @param embeddings List. Internal use only (stores embeddings during training).
#' @param skip Logical. Should step be skipped when baking? Default: FALSE.
#' @param id Character string. Unique ID for this step.
#'
#' @returns An updated recipe object.
#' @export
#'
#' @examples
#' \dontrun{
#' library(tidymodels)
#' library(dplyr)
#'
#' # Create a recipe with embeddings
#' rec <- recipe(sentiment ~ text, data = train_data) |>
#'   step_hf_embed(text, model = "BAAI/bge-small-en-v1.5")
#'
#' # Use in a workflow
#' wf <- workflow() |>
#'   add_recipe(rec) |>
#'   add_model(logistic_reg()) |>
#'   fit(data = train_data)
#' }
step_hf_embed <- function(recipe,
                          ...,
                          role = "predictor",
                          trained = FALSE,
                          model = "BAAI/bge-small-en-v1.5",
                          token = NULL,
                          embeddings = NULL,
                          skip = FALSE,
                          id = recipes::rand_id("hf_embed")) {
  
  if (!requireNamespace("recipes", quietly = TRUE)) {
    stop("Package 'recipes' is required. Install it with: install.packages('recipes')",
         call. = FALSE)
  }
  
  recipes::add_step(
    recipe,
    step_hf_embed_new(
      terms = recipes::ellipse_check(...),
      role = role,
      trained = trained,
      model = model,
      token = token,
      embeddings = embeddings,
      skip = skip,
      id = id
    )
  )
}


step_hf_embed_new <- function(terms, role, trained, model, token, embeddings, skip, id) {
  recipes::step(
    subclass = "hf_embed",
    terms = terms,
    role = role,
    trained = trained,
    model = model,
    token = token,
    embeddings = embeddings,
    skip = skip,
    id = id
  )
}


#' @exportS3Method recipes::prep
prep.step_hf_embed <- function(x, training, info = NULL, ...) {
  
  col_names <- recipes::recipes_eval_select(x$terms, training, info)
  
  recipes::check_type(training[, col_names], types = c("string", "character"))
  
  # During prep, we just store metadata
  # Actual embedding happens in bake
  step_hf_embed_new(
    terms = x$terms,
    role = x$role,
    trained = TRUE,
    model = x$model,
    token = x$token,
    embeddings = col_names,
    skip = x$skip,
    id = x$id
  )
}


#' @exportS3Method recipes::bake
bake.step_hf_embed <- function(object, new_data, ...) {
  
  col_names <- object$embeddings
  
  for (col_name in col_names) {
    # Generate embeddings for this column
    text_data <- new_data[[col_name]]
    
    embeddings_df <- hf_embed(
      text = text_data,
      model = object$model,
      token = object$token
    )
    
    # Filter out NULL embeddings and get valid indices
    valid_idx <- !sapply(embeddings_df$embedding, is.null)
    
    if (sum(valid_idx) == 0) {
      stop("No valid embeddings generated for column: ", col_name, call. = FALSE)
    }
    
    # Convert list-column of embeddings to separate columns
    valid_embeddings <- embeddings_df$embedding[valid_idx]
    emb_matrix <- do.call(rbind, valid_embeddings)
    n_dims <- ncol(emb_matrix)
    
    # Create column names
    emb_col_names <- paste0(col_name, "_emb_", seq_len(n_dims))
    
    # Initialize embedding columns with NA
    for (i in seq_len(n_dims)) {
      new_data[[emb_col_names[i]]] <- NA_real_
    }
    
    # Add embedding columns for valid rows
    for (i in seq_len(n_dims)) {
      new_data[[emb_col_names[i]]][valid_idx] <- emb_matrix[, i]
    }
    
    # Remove original text column
    new_data[[col_name]] <- NULL
  }
  
  new_data
}


#' @export
print.step_hf_embed <- function(x, width = max(20, options()$width - 30), ...) {
  title <- "HuggingFace embeddings for "
  recipes::print_step(x$embeddings, x$terms, x$trained, title, width)
  invisible(x)
}


#' @rdname step_hf_embed
#' @param x A step_hf_embed object
#' @exportS3Method generics::tidy
tidy.step_hf_embed <- function(x, ...) {
  if (recipes::is_trained(x)) {
    res <- tibble::tibble(
      terms = x$embeddings,
      model = x$model
    )
  } else {
    term_names <- recipes::sel2char(x$terms)
    res <- tibble::tibble(
      terms = term_names,
      model = x$model
    )
  }
  res$id <- x$id
  res
}


#' @rdname step_hf_embed
#' @exportS3Method generics::tunable
tunable.step_hf_embed <- function(x, ...) {
  tibble::tibble(
    name = "model",
    call_info = list(list(pkg = "huggingfaceR", fun = "step_hf_embed")),
    source = "recipe",
    component = "step_hf_embed",
    component_id = x$id
  )
}


#' @exportS3Method recipes::required_pkgs
required_pkgs.step_hf_embed <- function(x, ...) {
  c("huggingfaceR")
}
