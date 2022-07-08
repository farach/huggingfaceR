## code to prepare `models_with_downloads`

# Grab all of the models' data
all_models <- hf_search_models()

# Get the names of all models for filtering
get_names <- function(x) map(x, ~ names(.x))
all_models_names <- get_names(all_models)
id_col <- 1:length(all_models_names)

# Create tibble for pipeline
all_models_data <- tibble(id_col, all_models_names)

# Get models with data for downloads - or mapping will fail, then pull ids to filte
filter_ids <- all_models_data %>%
  unnest(all_models_names) %>%
  group_by(id_col) %>%
  filter("downloads" %in% all_models_names) %>%
  ungroup() %>%
  distinct(id_col) %>%
  pull(id_col)

# Keep only models with downloads info
all_models <- all_models[filter_ids]

# Extract the data we're interested in
models_with_downloads <- all_models %>%
  map_dfr(~ tibble(
    model = .x$modelId,
    downloads = .x$downloads,
    task = .x$pipeline_tag,
    sha = .x$sha,
    private = .x$private
  ))

usethis::use_data(models_with_downloads, overwrite = TRUE)
