#' Detect Default Config for a Dataset
#'
#' Query the splits endpoint to find the default config for a given dataset and split.
#'
#' @param dataset Character string. Dataset name.
#' @param split Character string. The split to find a config for.
#' @param token Character string or NULL. API token.
#'
#' @returns Character string with the config name.
#' @keywords internal
hf_detect_config <- function(dataset, split, token = NULL) {
  split <- hf_split_base(split)

  req <- httr2::request("https://datasets-server.huggingface.co/splits") |>
    httr2::req_url_query(dataset = dataset)

  if (!is.null(token)) {
    req <- httr2::req_auth_bearer_token(req, token)
  }

  resp <- tryCatch(
    req |>
      httr2::req_error(body = function(resp) {
        paste0("Failed to detect config for dataset '", dataset, "'")
      }) |>
      httr2::req_perform(),
    error = function(e) NULL
  )

  if (is.null(resp)) {
    # Dataset not found - try resolving the full ID via Hub API
    resolved <- hf_resolve_dataset(dataset, token)
    if (!is.null(resolved)) {
      return(hf_detect_config(resolved, split, token))
    }
    stop(
      paste0("Could not auto-detect config for dataset '", dataset,
             "'. Please specify the full dataset ID (e.g., 'org/dataset') ",
             "and/or the 'config' parameter explicitly."),
      call. = FALSE
    )
  }

  result <- httr2::resp_body_json(resp)

  if (is.null(result$splits) || length(result$splits) == 0) {
    stop(
      paste0("No configs found for dataset '", dataset,
             "'. Please specify the 'config' parameter explicitly."),
      call. = FALSE
    )
  }

  # Find configs that match the requested split
  matching <- purrr::keep(result$splits, function(s) s$split == split)

  if (length(matching) > 0) {
    return(matching[[1]]$config)
  }

  # If no match for the split, use the first available config
  result$splits[[1]]$config
}


hf_split_base <- function(split) {
  sub("\\[.*$", "", split)
}


hf_split_num_rows <- function(dataset, config, split, token = NULL) {
  req <- httr2::request("https://datasets-server.huggingface.co/size") |>
    httr2::req_url_query(dataset = dataset, config = config)

  if (!is.null(token)) {
    req <- httr2::req_auth_bearer_token(req, token)
  }

  resp <- req |>
    httr2::req_error(body = function(resp) {
      paste0("Failed to get split size for dataset '", dataset, "'")
    }) |>
    httr2::req_perform()

  result <- httr2::resp_body_json(resp)
  matching <- purrr::keep(result$size$splits %||% list(), function(s) s$split == split)

  if (length(matching) == 0 || is.null(matching[[1]]$num_rows)) {
    stop(
      paste0("Could not determine row count for split '", split,
             "'. Use a numeric slice or explicit limit/offset instead."),
      call. = FALSE
    )
  }

  matching[[1]]$num_rows
}


hf_parse_split_bound <- function(bound, num_rows, default, is_end = FALSE) {
  if (identical(bound, "")) {
    return(default)
  }

  if (grepl("%$", bound)) {
    pct <- suppressWarnings(as.numeric(sub("%$", "", bound)))
    if (is.na(pct) || pct < 0 || pct > 100) {
      stop("Split percentage bounds must be between 0% and 100%.", call. = FALSE)
    }

    value <- pct / 100 * num_rows
    return(if (is_end) ceiling(value) else floor(value))
  }

  value <- suppressWarnings(as.integer(bound))
  if (is.na(value) || value < 0) {
    stop("Split bounds must be non-negative integers or percentages.", call. = FALSE)
  }

  value
}


hf_parse_split <- function(split, dataset, config, token = NULL) {
  if (!grepl("\\[|\\]", split)) {
    return(list(split = split, offset = 0L, limit = Inf))
  }

  has_valid_brackets <- grepl("^[^[]+\\[", split) && grepl("\\]$", split)
  bounds <- if (has_valid_brackets) {
    sub("^[^[]+\\[", "", sub("\\]$", "", split))
  } else {
    ""
  }
  bounds <- strsplit(bounds, ":", fixed = TRUE)[[1]]

  if (!has_valid_brackets || length(bounds) != 2) {
    stop(
      "Split slices must use the form 'split[start:end]', e.g. 'train[:10%]' or 'train[100:200]'.",
      call. = FALSE
    )
  }

  base_split <- sub("\\[.*$", "", split)
  start_bound <- bounds[[1]]
  end_bound <- bounds[[2]]
  needs_num_rows <- grepl("%$", start_bound) || grepl("%$", end_bound) || identical(end_bound, "")
  num_rows <- if (needs_num_rows) hf_split_num_rows(dataset, config, base_split, token) else NA_integer_

  start <- hf_parse_split_bound(start_bound, num_rows, default = 0L, is_end = FALSE)
  end <- hf_parse_split_bound(end_bound, num_rows, default = num_rows, is_end = TRUE)

  if (!is.na(num_rows)) {
    start <- min(start, num_rows)
    end <- min(end, num_rows)
  }

  if (end < start) {
    stop("Split slice end must be greater than or equal to start.", call. = FALSE)
  }

  list(split = base_split, offset = start, limit = end - start)
}


#' Convert Dataset Server Rows to a Tibble
#'
#' Internal helper converting the `rows` payload from the Hugging Face datasets
#' server into a tibble.
#'
#' Row fields are not always scalars: many datasets contain variable-length
#' fields (for example `reference_files` on `openai/gdpval`). Converting such a
#' row directly with [tibble::as_tibble()] recycles the row to the length of its
#' longest field, silently multiplying rows. This helper detects fields that are
#' ever non-scalar and represents them as list-columns, so one source row always
#' yields exactly one output row.
#'
#' @param rows A list of row payloads, each a named list of field values.
#'
#' @returns A tibble with one row per element of `rows`.
#' @keywords internal
hf_rows_to_tibble <- function(rows) {
  if (length(rows) == 0) {
    return(tibble::tibble())
  }

  field_names <- unique(unlist(lapply(rows, names)))
  if (is.null(field_names)) {
    return(tibble::tibble())
  }

  # Rows parsed from JSON are lists. Anything else is coerced to a list so each
  # field can be read with the `[[` primitive below.
  row_is_list <- vapply(rows, is.list, logical(1))
  if (!all(row_is_list)) {
    rows[!row_is_list] <- lapply(rows[!row_is_list], as.list)
  }

  columns <- lapply(field_names, function(nm) {
    # `.subset2()` is the primitive behind `[[`, so a field is read from every
    # row without an interpreted function call per cell. Absent fields are NULL.
    hf_rows_column(lapply(rows, .subset2, nm))
  })

  names(columns) <- field_names
  tibble::as_tibble(columns)
}


# Combines one field's values, one per row, into a column.
#
# A field needs a list-column if any row holds a value that is not a length-one
# atomic scalar. A null is treated as missing rather than as a variable-length
# value, so it does not by itself force a list-column.
hf_rows_column <- function(values) {
  sizes <- lengths(values)
  missing <- sizes == 0L
  if (any(missing)) {
    missing[missing] <- vapply(values[missing], is.null, logical(1))
  }

  if (all(sizes[!missing] == 1L)) {
    # Scalar column: every value is a length-one atomic or absent. A non-recursive
    # unlist() stays a list if any length-one value is itself a list.
    filled <- values
    filled[missing] <- list(NA)
    column <- unlist(filled, recursive = FALSE, use.names = FALSE)
    if (!is.list(column)) {
      return(column)
    }
  }

  # One cell per row. An absent or null field becomes an empty list rather than
  # NA so the column stays a well-formed list-column.
  values[missing] <- list(list())
  values
}


#' Resolve a Short Dataset Name to Full ID
#'
#' Uses the Hub API to find the full org/name ID for a dataset.
#'
#' @param dataset Character string. Short dataset name.
#' @param token Character string or NULL. API token.
#'
#' @returns Character string with full dataset ID, or NULL if not found.
#' @keywords internal
hf_resolve_dataset <- function(dataset, token = NULL) {
  req <- httr2::request("https://huggingface.co/api/datasets") |>
    httr2::req_url_query(search = dataset, limit = 5) |>
    hf_req_retry(max_tries = 3) |>
    httr2::req_timeout(20)

  if (!is.null(token)) {
    req <- httr2::req_auth_bearer_token(req, token)
  }

  resp <- tryCatch(
    req |> httr2::req_perform(),
    error = function(e) NULL
  )

  if (is.null(resp)) return(NULL)

  results <- httr2::resp_body_json(resp)

  if (length(results) == 0) return(NULL)

  # Look for an exact match on the dataset name portion
  for (r in results) {
    id <- r$id %||% ""
    # Match if the name after "/" equals the search term
    name_part <- sub("^.*/", "", id)
    if (tolower(name_part) == tolower(dataset)) {
      return(id)
    }
  }

  NULL
}


hf_viewer_url <- "https://datasets-server.huggingface.co"


# Expands a short dataset name such as "imdb" to its full "owner/name" ID.
# `unresolved` records a failed lookup so later errors can say so.
hf_viewer_dataset <- function(dataset, token = NULL) {
  if (grepl("/", dataset, fixed = TRUE)) {
    return(list(dataset = dataset, unresolved = FALSE))
  }
  resolved <- hf_resolve_dataset(dataset, token)
  if (is.null(resolved)) {
    list(dataset = dataset, unresolved = TRUE)
  } else {
    list(dataset = resolved, unresolved = FALSE)
  }
}


# Builds the error message for a failed dataset viewer request.
hf_viewer_error_body <- function(dataset, split = NULL, unresolved_short_name = FALSE) {
  function(resp) {
    if (identical(httr2::resp_status(resp), 429L)) {
      return(hf_rate_limit_message(paste0(
        "The dataset viewer returns at most 100 rows per request, so for large ",
        "loads use hf_load_dataset(format = \"parquet\"), which needs far fewer. "
      )))
    }
    body <- tryCatch(
      httr2::resp_body_json(resp),
      error = function(e) list(error = "Unknown error")
    )
    error_msg <- body$error %||% "Unknown error"
    renamed <- grepl("renamed", error_msg, ignore.case = TRUE)
    not_found <- grepl("not found", error_msg, ignore.case = TRUE)

    if (unresolved_short_name && (renamed || not_found)) {
      paste0(
        "Could not expand the short dataset name '", dataset, "' to its full ",
        "'owner/name' form, because the Hub lookup did not succeed. Pass the ",
        "full dataset ID (for example \"stanfordnlp/imdb\" rather than ",
        "\"imdb\"), or retry. Original API message: ", error_msg
      )
    } else if (renamed) {
      paste0(
        "Dataset '", dataset, "' has been renamed on the Hub. Use its current ",
        "'owner/name' ID; hf_search_datasets(\"", sub("^.*/", "", dataset),
        "\") will show it. Original API message: ", error_msg
      )
    } else if (not_found && !is.null(split)) {
      paste0("Dataset '", dataset, "' or split '", split, "' not found")
    } else if (not_found) {
      paste0("Dataset '", dataset, "' not found")
    } else {
      paste0("API error: ", error_msg)
    }
  }
}


# Builds a GET request to a dataset viewer endpoint. NULL query values are
# dropped.
hf_viewer_request <- function(endpoint, query, token = NULL, error_body = NULL) {
  req <- httr2::request(paste0(hf_viewer_url, "/", endpoint)) |>
    httr2::req_url_query(!!!query) |>
    hf_req_retry(max_tries = 3)

  if (!is.null(token)) {
    req <- httr2::req_auth_bearer_token(req, token)
  }
  if (!is.null(error_body)) {
    req <- httr2::req_error(req, body = error_body)
  }
  req
}


# Fetches up to `limit` rows from a paged dataset viewer endpoint (/rows,
# /search, or /filter), starting at `offset`.
#
# The first page is fetched on its own because it reports how many rows exist.
# Every later page then has a known offset, so those pages are requested
# together, up to `max_active` at a time. Pages are combined in offset order.
# If the total is not reported, the remaining pages are fetched one by one.
#
# Returns a list with the raw row payloads, the reported total, and whether the
# viewer only indexed part of the dataset.
hf_viewer_pages <- function(endpoint, query, offset, limit, token = NULL,
                            error_body = NULL, max_active = 1L,
                            page_size = 100L) {
  if (limit <= 0) {
    return(list(rows = list(), num_rows_total = NA_real_, partial = FALSE))
  }

  page_request <- function(start, size) {
    hf_viewer_request(
      endpoint,
      c(query, list(offset = start, length = size)),
      token = token,
      error_body = error_body
    )
  }
  fetch <- function(start, size) {
    httr2::resp_body_json(httr2::req_perform(page_request(start, size)))
  }

  first_size <- min(page_size, limit)
  first <- fetch(offset, first_size)
  pages <- list(first$rows %||% list())
  fetched <- length(pages[[1L]])
  total <- first$num_rows_total

  if (fetched == first_size && fetched < limit) {
    if (is.numeric(total) && length(total) == 1L && !is.na(total)) {
      start <- offset + fetched
      end <- min(offset + limit, total)
      if (start < end) {
        starts <- seq(start, end - 1, by = page_size)
        items <- Map(c, starts, pmin(page_size, end - starts))
        more <- hf_map_requests(
          items,
          perform_one = function(x) fetch(x[[1L]], x[[2L]]),
          build_one = function(x) page_request(x[[1L]], x[[2L]]),
          parse = httr2::resp_body_json,
          max_active = max_active
        )
        pages <- c(pages, lapply(more, function(page) page$rows %||% list()))
      }
    } else {
      repeat {
        size <- min(page_size, limit - fetched)
        rows <- fetch(offset + fetched, size)$rows %||% list()
        if (length(rows) == 0L) {
          break
        }
        pages[[length(pages) + 1L]] <- rows
        fetched <- fetched + length(rows)
        if (length(rows) < size || fetched >= limit) {
          break
        }
      }
    }
  }

  list(
    rows = unlist(pages, recursive = FALSE, use.names = FALSE),
    num_rows_total = if (is.numeric(total)) total else NA_real_,
    partial = isTRUE(first$partial)
  )
}


# Converts row payloads from /search or /filter to a tibble that keeps each
# row's position in the full split.
hf_viewer_rows_tibble <- function(rows, dataset, split) {
  if (length(rows) == 0L) {
    return(tibble::tibble(
      .row_idx = integer(),
      .dataset = character(),
      .split = character()
    ))
  }

  df <- hf_rows_to_tibble(lapply(rows, .subset2, "row"))
  df$.row_idx <- as.integer(vapply(
    rows,
    function(row) as.numeric(row$row_idx %||% NA_real_),
    numeric(1)
  ))
  df$.dataset <- dataset
  df$.split <- split
  df
}


# Checks that `split` names a whole split. Slices only make sense for
# hf_load_dataset().
hf_check_plain_split <- function(split) {
  if (!is.character(split) || length(split) != 1L || is.na(split) ||
      grepl("[][]", split)) {
    stop(
      "`split` must be a single split name such as \"train\". Slice syntax ",
      "is only supported by hf_load_dataset().",
      call. = FALSE
    )
  }
  invisible(split)
}


# Downloads each URL to the matching path, up to `max_active` at a time. Real
# responses stream to disk. Mocked responses keep the body in memory, so it is
# written to the path here.
hf_download_files <- function(urls, paths, token = NULL, max_active = 1L) {
  reqs <- lapply(urls, function(url) {
    req <- httr2::request(url)
    if (!is.null(token)) {
      req <- httr2::req_auth_bearer_token(req, token)
    }
    req
  })

  batch <- hf_perform_batch(
    reqs,
    input_indices = seq_along(reqs),
    max_active = max_active,
    progress = rlang::is_interactive(),
    paths = paths
  )
  failed <- which(batch$.error)
  if (length(failed) > 0L) {
    stop(batch$response[[failed[[1L]]]])
  }

  for (i in seq_along(paths)) {
    body <- batch$response[[i]]$body
    if (is.raw(body)) {
      writeBin(body, paths[[i]])
    }
  }
  invisible(paths)
}


# Reads rows of one split from the dataset viewer's Parquet export.
#
# Shards are downloaded in order and stop once enough rows have been read to
# cover `offset + limit`. A whole split downloads every shard, up to
# `max_active` at a time. The Parquet decoding, concatenation, and slicing all
# happen in Arrow's C++ library.
hf_parquet_split <- function(dataset, config, split, offset, limit,
                             token = NULL, max_active = 1L, error_body = NULL) {
  if (!requireNamespace("arrow", quietly = TRUE)) {
    stop(
      "Package 'arrow' is required for format = \"parquet\". ",
      "Install with: install.packages('arrow')",
      call. = FALSE
    )
  }

  listing <- hf_viewer_request(
    "parquet",
    list(dataset = dataset, config = config),
    token = token,
    error_body = error_body
  ) |>
    httr2::req_perform() |>
    httr2::resp_body_json()

  files <- Filter(
    function(file) identical(file$split, split),
    listing$parquet_files %||% list()
  )
  if (length(files) == 0L) {
    stop(
      "No Parquet export is available for split '", split, "' of dataset '",
      dataset, "' (config '", config, "'). Use format = \"rows\" instead.",
      call. = FALSE
    )
  }
  if (isTRUE(listing$partial)) {
    cli::cli_inform(c(
      "i" = "The Parquet export of {.val {dataset}} is partial, so large splits are cut short."
    ))
  }

  filenames <- vapply(files, function(file) file$filename %||% "", character(1))
  files <- files[order(filenames)]
  urls <- vapply(files, function(file) file$url, character(1))

  if (limit <= 0) {
    return(tibble::tibble())
  }

  needed <- offset + limit
  paths <- tempfile(rep("hf-parquet-", length(urls)), fileext = ".parquet")
  on.exit(unlink(paths), add = TRUE)

  tables <- list()
  rows_read <- 0
  next_shard <- 1L
  while (next_shard <= length(urls) && rows_read < needed) {
    # The listing has no row counts, so a finite request reads the first shard
    # alone and then sizes each batch from the rows per shard seen so far.
    # Shards hold about 500 MB, so this avoids downloading any it won't use.
    n_shards <- if (!is.finite(needed)) {
      length(urls)
    } else if (rows_read == 0) {
      1L
    } else {
      per_shard <- rows_read / (next_shard - 1L)
      min(max_active, max(1L, ceiling((needed - rows_read) / per_shard)))
    }
    last_shard <- min(length(urls), next_shard + n_shards - 1L)
    shards <- seq.int(next_shard, last_shard)
    hf_download_files(urls[shards], paths[shards], token = token, max_active = max_active)

    for (k in shards) {
      table <- arrow::read_parquet(paths[[k]], as_data_frame = FALSE, mmap = FALSE)
      tables[[length(tables) + 1L]] <- table
      rows_read <- rows_read + table$num_rows
      unlink(paths[[k]])
    }
    next_shard <- last_shard + 1L
  }

  table <- if (length(tables) == 1L) tables[[1L]] else do.call(arrow::concat_tables, tables)
  start <- min(offset, table$num_rows)
  table <- table$Slice(start, min(limit, table$num_rows - start))
  tibble::as_tibble(as.data.frame(table))
}


#' Load Dataset via Hugging Face Datasets Server API
#'
#' Load a dataset from Hugging Face Hub using the Datasets Server API.
#' This is an API-first approach that doesn't require Python.
#' For local dataset loading with Python, see the legacy function or advanced vignette.
#'
#' Rows are requested in pages of 100. The first page reports the size of the
#' split, so the remaining pages are requested together, up to
#' \code{max_active} at a time, and combined in order.
#'
#' With \code{format = "parquet"}, the split's Parquet export is downloaded
#' and read with the arrow package instead. Shards of the export hold about
#' 500 MB each and are read whole, so this format suits large or complete
#' splits (\code{limit = Inf}). For a few rows the default \code{"rows"}
#' format is faster. Column types follow the Parquet schema, so they can
#' differ from \code{"rows"}: images and audio arrive as data-frame columns of
#' raw bytes and file names rather than URLs.
#'
#' Each \code{"rows"} request returns at most 100 rows, and Hugging Face caps
#' how many requests a user or IP address can send in each 5-minute window.
#' Anonymous requests get the lowest cap, so set a token with
#' [hf_set_token()]. To load more than a few thousand rows, use
#' \code{format = "parquet"}, which needs far fewer requests. Requests that hit
#' the cap are retried after the wait the server asks for.
#'
#' @param dataset Character string. Dataset name (e.g., "imdb", "squad").
#' @param split Character string. Dataset split: "train", "test", "validation", etc.
#'   Supports Hugging Face slice syntax such as `"train[100:200]"`.
#'   Percentage slices like `"train[:10\\%]"` are also supported. Default:
#'   "train".
#' @param config Character string or NULL. Dataset configuration/subset name.
#'   If NULL (default), auto-detected from the dataset's available configs.
#' @param limit Integer. Maximum number of rows to fetch. Default: 1000.
#'   Set to Inf to fetch all rows (may be slow for large datasets).
#' @param offset Integer. Row offset for pagination. Default: 0.
#' @param token Character string or NULL. API token for private datasets.
#' @param format Either \code{"rows"} (default), which reads JSON pages from
#'   the dataset viewer, or \code{"parquet"}, which reads the split's Parquet
#'   export with the arrow package.
#' @param max_active Integer. The most page or shard requests sent at once.
#'   Dataset viewer reads are not billed, so the default is 4. Use 1 to send
#'   requests one at a time.
#'
#' @returns A tibble with dataset rows, plus .dataset and .split columns.
#' @seealso [hf_dataset_search()], [hf_dataset_filter()], and
#'   [hf_dataset_statistics()] to query a split without loading it.
#' @export
#'
#' @examples
#' \dontrun{
#' # Load first 1000 rows of IMDB train set
#' imdb <- hf_load_dataset("imdb", split = "train", limit = 1000)
#'
#' # Load test set
#' imdb_test <- hf_load_dataset("imdb", split = "test", limit = 500)
#'
#' # Load a slice of a split
#' imdb_sample <- hf_load_dataset("imdb", split = "train[100:200]", limit = Inf)
#'
#' # Load a whole split from its Parquet export (requires arrow)
#' imdb_all <- hf_load_dataset("imdb", split = "train", limit = Inf,
#'                             format = "parquet")
#' }
hf_load_dataset <- function(dataset,
                            split = "train",
                            config = NULL,
                            limit = 1000,
                            offset = 0,
                            token = NULL,
                            format = c("rows", "parquet"),
                            max_active = 4L) {

  format <- match.arg(format)
  max_active <- hf_max_active(max_active)
  token <- hf_get_token(token, required = FALSE)

  # Resolve short dataset names (e.g., "imdb" -> "stanfordnlp/imdb")
  target <- hf_viewer_dataset(dataset, token)
  dataset <- target$dataset

  if (is.null(config)) {
    config <- hf_detect_config(dataset, hf_split_base(split), token)
  }

  split_info <- hf_parse_split(split, dataset, config, token)
  offset <- offset + split_info$offset
  if (is.finite(split_info$limit)) {
    limit <- min(limit, split_info$limit)
  }

  error_body <- hf_viewer_error_body(dataset, split, target$unresolved)

  df <- if (format == "parquet") {
    hf_parquet_split(
      dataset, config, split_info$split, offset, limit,
      token = token, max_active = max_active, error_body = error_body
    )
  } else {
    page <- hf_viewer_pages(
      "rows",
      list(dataset = dataset, config = config, split = split_info$split),
      offset = offset,
      limit = limit,
      token = token,
      error_body = error_body,
      max_active = max_active
    )
    hf_rows_to_tibble(lapply(page$rows, .subset2, "row"))
  }

  if (nrow(df) == 0L) {
    cli::cli_warn("No rows returned. Dataset may be empty or unavailable via API.")
    return(tibble::tibble())
  }

  # Add metadata columns
  df$.dataset <- dataset
  df$.split <- split

  df
}


#' Search Dataset Rows by Text
#'
#' Full-text search over the string columns of one dataset split, run by the
#' Hugging Face dataset viewer. Only the matching rows are downloaded.
#'
#' Search works for datasets that have a Parquet export. Use
#' [hf_dataset_validity()] to check before searching. For large datasets the
#' viewer indexes only the first part of the split, and a message says so.
#'
#' @param dataset Character string. Dataset ID (e.g., "stanfordnlp/imdb").
#'   Short names are resolved as in [hf_load_dataset()].
#' @param query Character string. The text to search for.
#' @param split Character string. Split name. Default: "train". Slice syntax
#'   is not supported here.
#' @param config Character string or NULL. Dataset configuration. If NULL
#'   (default), it is detected from the dataset's available configs.
#' @param limit Integer. Maximum number of matching rows to return. Default:
#'   100. Use Inf for all matches.
#' @param offset Integer. Number of matching rows to skip. Default: 0.
#' @param max_active Integer. The most page requests sent at once. Default: 4.
#' @param token Character string or NULL. API token. The search and filter
#'   endpoints can refuse anonymous requests, so a token is recommended.
#'
#' @returns A tibble with one row per match, plus \code{.row_idx} (the row's
#'   0-based position in the split), \code{.dataset}, and \code{.split}.
#' @seealso [hf_dataset_filter()] to select rows with a condition.
#' @export
#'
#' @examples
#' \dontrun{
#' hf_dataset_search("cornell-movie-review-data/rotten_tomatoes", "dull",
#'                   limit = 20)
#' }
hf_dataset_search <- function(dataset, query, split = "train", config = NULL,
                              limit = 100, offset = 0, max_active = 4L,
                              token = NULL) {
  if (!is.character(query) || length(query) != 1L || is.na(query) || !nzchar(query)) {
    stop("`query` must be a single non-empty string.", call. = FALSE)
  }
  hf_viewer_query_rows(
    "search", dataset, split, config, list(query = query),
    limit = limit, offset = offset, max_active = max_active, token = token
  )
}


#' Filter Dataset Rows with a Condition
#'
#' Select rows of one dataset split with a SQL-style condition, run by the
#' Hugging Face dataset viewer. Only the matching rows are downloaded.
#'
#' The \code{where} condition compares columns with \code{=}, \code{<>},
#' \code{>}, \code{>=}, \code{<}, and \code{<=}, and joins comparisons with
#' \code{AND}, \code{OR}, and \code{NOT}. Put column names in double quotes
#' and text values in single quotes, for example
#' \code{"\"label\" = 1 AND \"lang\" = 'en'"}. Filtering works for datasets
#' that have a Parquet export.
#'
#' @inheritParams hf_dataset_search
#' @param where Character string. The condition rows must meet.
#' @param orderby Character string or NULL. Sort order, for example
#'   \code{"\"score\" DESC"}. Default: NULL (split order).
#'
#' @returns A tibble with one row per match, plus \code{.row_idx},
#'   \code{.dataset}, and \code{.split}.
#' @seealso [hf_dataset_search()] for full-text search.
#' @export
#'
#' @examples
#' \dontrun{
#' hf_dataset_filter("cornell-movie-review-data/rotten_tomatoes",
#'                   where = "\"label\" = 1", limit = 50)
#' }
hf_dataset_filter <- function(dataset, where, orderby = NULL, split = "train",
                              config = NULL, limit = 100, offset = 0,
                              max_active = 4L, token = NULL) {
  if (!is.character(where) || length(where) != 1L || is.na(where) || !nzchar(where)) {
    stop("`where` must be a single non-empty string.", call. = FALSE)
  }
  if (!is.null(orderby) &&
      (!is.character(orderby) || length(orderby) != 1L || is.na(orderby))) {
    stop("`orderby` must be NULL or a single string.", call. = FALSE)
  }
  hf_viewer_query_rows(
    "filter", dataset, split, config, list(where = where, orderby = orderby),
    limit = limit, offset = offset, max_active = max_active, token = token
  )
}


# Shared path for the /search and /filter endpoints.
hf_viewer_query_rows <- function(endpoint, dataset, split, config, params,
                                 limit, offset, max_active, token) {
  hf_check_plain_split(split)
  max_active <- hf_max_active(max_active)
  token <- hf_get_token(token, required = FALSE)

  target <- hf_viewer_dataset(dataset, token)
  dataset <- target$dataset
  config <- config %||% hf_detect_config(dataset, split, token)

  page <- hf_viewer_pages(
    endpoint,
    c(list(dataset = dataset, config = config, split = split), params),
    offset = offset,
    limit = limit,
    token = token,
    error_body = hf_viewer_error_body(dataset, split, target$unresolved),
    max_active = max_active
  )
  if (page$partial) {
    cli::cli_inform(c(
      "i" = "Results cover only the first part of {.val {dataset}}, because the viewer indexes only the start of large datasets."
    ))
  }

  hf_viewer_rows_tibble(page$rows, dataset, split)
}


#' Get Column Statistics for a Dataset Split
#'
#' Retrieve the summary statistics that the Hugging Face dataset viewer
#' computes for each column of a split, without downloading the data.
#'
#' Which statistics are filled in depends on the column type. Numeric columns
#' have \code{min}, \code{max}, \code{mean}, \code{median}, \code{std}, and a
#' histogram. Text and list columns describe their lengths in the same way.
#' Label and boolean columns have \code{n_unique} and \code{frequencies}. The
#' \code{statistics} column keeps every field the API returned, including
#' summaries with no numeric form such as datetime ranges.
#'
#' @inheritParams hf_dataset_search
#'
#' @returns A tibble with one row per column: \code{column}, \code{type},
#'   \code{nan_count}, \code{nan_proportion}, \code{n_unique}, \code{min},
#'   \code{max}, \code{mean}, \code{median}, \code{std}, \code{histogram} (a
#'   list of tibbles with \code{bin_start}, \code{bin_end}, and \code{count}),
#'   \code{frequencies} (a list of tibbles with \code{value} and
#'   \code{count}), and \code{statistics} (the raw API fields). The number of
#'   rows in the split is stored in the \code{"num_examples"} attribute.
#' @export
#'
#' @examples
#' \dontrun{
#' stats <- hf_dataset_statistics("cornell-movie-review-data/rotten_tomatoes")
#' stats
#' stats$frequencies[[1]]
#' attr(stats, "num_examples")
#' }
hf_dataset_statistics <- function(dataset, split = "train", config = NULL,
                                  token = NULL) {
  hf_check_plain_split(split)
  token <- hf_get_token(token, required = FALSE)

  target <- hf_viewer_dataset(dataset, token)
  dataset <- target$dataset
  config <- config %||% hf_detect_config(dataset, split, token)

  result <- hf_viewer_request(
    "statistics",
    list(dataset = dataset, config = config, split = split),
    token = token,
    error_body = hf_viewer_error_body(dataset, split, target$unresolved)
  ) |>
    httr2::req_perform() |>
    httr2::resp_body_json()

  if (isTRUE(result$partial)) {
    cli::cli_inform(c(
      "i" = "Statistics for {.val {dataset}} cover only the first part of the split."
    ))
  }

  columns <- result$statistics %||% list()
  stats <- lapply(columns, function(column) column$column_statistics %||% list())
  number <- function(name) {
    vapply(stats, function(s) {
      value <- s[[name]]
      if (is.numeric(value) && length(value) == 1L) as.numeric(value) else NA_real_
    }, numeric(1))
  }
  text <- function(name) {
    vapply(columns, function(column) as.character(column[[name]] %||% NA_character_),
           character(1))
  }

  out <- tibble::tibble(
    column = text("column_name"),
    type = text("column_type"),
    nan_count = as.integer(number("nan_count")),
    nan_proportion = number("nan_proportion"),
    n_unique = as.integer(number("n_unique")),
    min = number("min"),
    max = number("max"),
    mean = number("mean"),
    median = number("median"),
    std = number("std"),
    histogram = lapply(stats, function(s) hf_stats_histogram(s$histogram)),
    frequencies = lapply(stats, function(s) hf_stats_frequencies(s$frequencies)),
    statistics = stats
  )
  attr(out, "num_examples") <- result$num_examples
  out
}


# Converts a viewer histogram (counts plus one more bin edge) to a tibble.
hf_stats_histogram <- function(histogram) {
  if (is.null(histogram) || length(histogram$hist) == 0L) {
    return(NULL)
  }
  edges <- unlist(histogram$bin_edges, use.names = FALSE)
  counts <- as.integer(unlist(histogram$hist, use.names = FALSE))
  if (length(edges) != length(counts) + 1L) {
    return(NULL)
  }
  tibble::tibble(
    bin_start = edges[-length(edges)],
    bin_end = edges[-1L],
    count = counts
  )
}


# Converts viewer frequencies, a named list of counts, to a tibble.
hf_stats_frequencies <- function(frequencies) {
  if (length(frequencies) == 0L) {
    return(NULL)
  }
  tibble::tibble(
    value = names(frequencies),
    count = as.integer(unlist(frequencies, use.names = FALSE))
  )
}


#' Check Which Dataset Viewer Features Work for a Dataset
#'
#' Report whether the Hugging Face dataset viewer can preview, page through,
#' search, filter, and summarize each dataset. Use it before calling
#' [hf_load_dataset()], [hf_dataset_search()], [hf_dataset_filter()], or
#' [hf_dataset_statistics()].
#'
#' @param dataset Character vector of dataset IDs. Short names are resolved
#'   as in [hf_load_dataset()].
#' @param token Character string or NULL. API token for private or gated
#'   datasets.
#'
#' @returns A tibble with one row per dataset and logical columns
#'   \code{viewer} (rows can be loaded), \code{preview}, \code{search},
#'   \code{filter}, and \code{statistics}.
#' @export
#'
#' @examples
#' \dontrun{
#' hf_dataset_validity(c("stanfordnlp/imdb", "openai/gdpval"))
#' }
hf_dataset_validity <- function(dataset, token = NULL) {
  if (!is.character(dataset) || length(dataset) == 0L || anyNA(dataset)) {
    stop("`dataset` must be a character vector of dataset IDs.", call. = FALSE)
  }
  token <- hf_get_token(token, required = FALSE)

  targets <- lapply(dataset, hf_viewer_dataset, token = token)
  build_one <- function(target) {
    hf_viewer_request(
      "is-valid",
      list(dataset = target$dataset),
      token = token,
      error_body = hf_viewer_error_body(target$dataset, NULL, target$unresolved)
    )
  }
  results <- hf_map_requests(
    targets,
    perform_one = function(target) {
      httr2::resp_body_json(httr2::req_perform(build_one(target)))
    },
    build_one = build_one,
    parse = httr2::resp_body_json,
    max_active = 4L
  )

  flag <- function(name) {
    vapply(results, function(result) isTRUE(result[[name]]), logical(1))
  }
  tibble::tibble(
    dataset = vapply(targets, function(target) target$dataset, character(1)),
    viewer = flag("viewer"),
    preview = flag("preview"),
    search = flag("search"),
    filter = flag("filter"),
    statistics = flag("statistics")
  )
}


#' Get Dataset Information
#'
#' Retrieve metadata about a dataset from Hugging Face Hub.
#'
#' @param dataset Character string. Dataset name.
#' @param token Character string or NULL. API token for private datasets.
#'
#' @returns A list with dataset information.
#' @export
#'
#' @examples
#' \dontrun{
#' hf_dataset_info("imdb")
#' }
hf_dataset_info <- function(dataset, token = NULL) {

  token <- hf_get_token(token, required = FALSE)
  target <- hf_viewer_dataset(dataset, token)

  hf_viewer_request(
    "info",
    list(dataset = target$dataset),
    token = token,
    error_body = function(resp) {
      paste0("Dataset '", target$dataset, "' not found or unavailable")
    }
  ) |>
    httr2::req_perform() |>
    httr2::resp_body_json()
}
