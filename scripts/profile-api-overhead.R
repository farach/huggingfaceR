# Usage: Rscript profile-api-overhead.R baseline|current <R-library> <output-dir> <commit>
#
# Measures the R-side cost of huggingfaceR's hosted-API workflows with the
# network removed. Requests are answered by mocks with precomputed payloads, so
# every timing is time spent in R and in the compiled code it calls (the JSON
# parser and BLAS), never network I/O or server-side inference. Run each
# version in a fresh process on the same runner, then compare the results with
# compare-api-profiles.R, which also checks that both versions return the same
# outputs.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4L || !args[1] %in% c("baseline", "current")) {
  stop("Usage: profile-api-overhead.R baseline|current <R-library> <output-dir> <commit>")
}
label <- args[1]
library_dir <- normalizePath(args[2], mustWork = TRUE)
output_dir <- args[3]
source_commit <- args[4]
dir.create(file.path(output_dir, "outputs"), recursive = TRUE, showWarnings = FALSE)
.libPaths(c(library_dir, .libPaths()))
Sys.unsetenv(c("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_BILL_TO"))
library(huggingfaceR)
stopifnot(
  identical(normalizePath(find.package("huggingfaceR")),
            normalizePath(file.path(library_dir, "huggingfaceR")))
)
for (pkg in c("bench", "testthat", "recipes", "jsonlite", "httr2", "tibble")) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop("Profiling requires the package '", pkg, "'.")
  }
}
profile_memory <- isTRUE(capabilities("profmem"))
hf <- asNamespace("huggingfaceR")

# A request that no scenario mock answers is a bug in this script. It must
# never reach the network.
options(httr2_mock = function(req) stop("Unmocked request: ", req$url, call. = FALSE))

# The explicit provider suffix routes without a Hub metadata lookup.
model <- "BAAI/bge-small-en-v1.5:hf-inference"
dims <- 384L

texts_for <- function(n, prefix = "text") sprintf("%s-%06d", prefix, seq_len(n))
json_raw <- function(x, ...) {
  charToRaw(enc2utf8(as.character(jsonlite::toJSON(x, digits = NA, ...))))
}
embedding_values <- function(index) round(sin(index * 0.37 + seq_len(dims) * 0.11), 6)
json_response <- function(req, body, headers = list()) {
  httr2::response(
    status_code = 200L,
    url = req$url,
    headers = c(list(`Content-Type` = "application/json"), headers),
    body = body
  )
}

# Precomputed response bodies keyed by the first input of each request. The
# keys mirror batch_vector(), which splits inputs into contiguous batches.
batch_bodies <- function(texts, batch_size, make_body) {
  bodies <- new.env(hash = TRUE)
  n <- length(texts)
  for (start in seq.int(1L, n, by = batch_size)) {
    rows <- start:min(start + batch_size - 1L, n)
    assign(texts[[start]], make_body(rows), envir = bodies)
  }
  bodies
}
embedding_body <- function(rows) {
  json_raw(t(vapply(rows, embedding_values, numeric(dims))))
}
classification_body <- function(rows) {
  items <- data.frame(
    label = ifelse(rows %% 3L == 0L, "NEGATIVE", "POSITIVE"),
    score = round(0.5 + (rows %% 50L) / 100, 4)
  )
  if (length(rows) == 1L) {
    items <- rbind(items, data.frame(label = "OTHER", score = 0.01))
  }
  json_raw(list(items), dataframe = "rows")
}

# Stands in for hf_perform_batch(), the package's only parallel network seam.
mock_perform_batch <- function(bodies, failing = character()) {
  force(bodies)
  force(failing)
  function(reqs, input_indices, max_active = 10L, progress = TRUE) {
    responses <- lapply(reqs, function(req) {
      key <- as.character(req$body$data$inputs[[1]])
      if (key %in% failing) {
        return(errorCondition(
          "HTTP 503 Service Unavailable.",
          class = c("httr2_http_503", "httr2_http", "httr2_error")
        ))
      }
      json_response(req, get(key, envir = bodies, inherits = FALSE))
    })
    failed <- vapply(responses, inherits, logical(1), "error")
    tibble::tibble(
      .input_idx = input_indices,
      response = responses,
      .error = failed,
      .error_msg = vapply(responses, function(resp) {
        if (inherits(resp, "error")) conditionMessage(resp) else NA_character_
      }, character(1))
    )
  }
}
with_batch_mock <- function(bodies, code, failing = character()) {
  testthat::local_mocked_bindings(
    hf_perform_batch = mock_perform_batch(bodies, failing),
    .package = "huggingfaceR"
  )
  code
}
with_http_mock <- function(mock, code) {
  httr2::local_mocked_responses(mock)
  code
}

metrics <- list()
outputs <- list()
measure <- function(scenario, n, run, iterations = 5L, keep_output = TRUE) {
  key <- paste0(scenario, "_", n)
  value <- run()
  if (keep_output) {
    outputs[[key]] <<- value
  }
  rm(value)
  invisible(gc())
  measured <- bench::mark(
    run(), iterations = iterations, check = FALSE,
    filter_gc = FALSE, memory = profile_memory
  )
  times <- as.numeric(measured$time[[1]])
  metrics[[key]] <<- data.frame(
    version = label, scenario = scenario, n = n,
    iterations = length(times),
    median_s = stats::median(times),
    min_s = min(times),
    r_allocated_mib = if (profile_memory) {
      as.numeric(measured$mem_alloc[1]) / 1024^2
    } else {
      NA_real_
    },
    stringsAsFactors = FALSE
  )
  cat(sprintf("%-32s n = %-7d median %9.4f s\n", scenario, n, stats::median(times)))
}

# Pairwise similarity: one dimension, then mixed dimensions with missing and
# zero-norm embeddings.
similarity_input <- function(n, mixed = FALSE) {
  embedding <- lapply(seq_len(n), embedding_values)
  if (mixed) {
    short <- seq(5L, n, by = 11L)
    embedding[short] <- lapply(short, function(i) embedding_values(i)[1:256])
    embedding[seq(6L, n, by = 13L)] <- list(numeric(dims))
    embedding[seq(3L, n, by = 7L)] <- list(NULL)
  }
  tibble::tibble(text = texts_for(n), embedding = embedding)
}
for (n in c(100L, 1000L, 2000L)) {
  data <- similarity_input(n)
  measure("similarity", n, function() hf_similarity(data),
          iterations = if (n > 1000L) 3L else 5L)
}
data <- similarity_input(1000L, mixed = TRUE)
measure("similarity_mixed", 1000L, function() hf_similarity(data), iterations = 3L)

# Compiled floor: the BLAS cross-product alone for 2,000 unit vectors.
unit <- vapply(seq_len(2000L), embedding_values, numeric(dims))
unit <- unit / rep(sqrt(colSums(unit * unit)), each = dims)
measure("floor_blas_similarity", 2000L, function() crossprod(unit),
        keep_output = FALSE)
rm(unit, data)

# Nearest neighbours against a mocked query embedding.
query_bodies <- new.env(hash = TRUE)
assign("find similar text", json_raw(matrix(embedding_values(0.5), nrow = 1L)),
       envir = query_bodies)
with_batch_mock(query_bodies, {
  for (n in c(1000L, 50000L)) {
    data <- tibble::tibble(
      text = texts_for(n),
      embedding = lapply(seq_len(n), embedding_values)
    )
    measure("nearest_neighbors", n, function() {
      hf_nearest_neighbors(data, "find similar text", k = 10, model = model)
    })
  }
})
corpus <- vapply(seq_len(50000L), embedding_values, numeric(dims))
query <- embedding_values(0.5)
measure("floor_blas_nearest", 50000L, function() crossprod(corpus, query),
        keep_output = FALSE)
rm(corpus, data)

# Feature extraction in batches of 100, plus one request for hf_embed().
for (n in c(1L, 100L, 5000L)) {
  texts <- texts_for(n)
  bodies <- batch_bodies(texts, 100L, embedding_body)
  with_batch_mock(bodies, measure("embed_batch", n, function() {
    hf_embed_batch(texts, model = model, progress = FALSE)
  }))
}
texts <- texts_for(1000L)
bodies <- batch_bodies(texts, 1000L, embedding_body)
with_batch_mock(bodies, measure("embed", 1000L, function() hf_embed(texts, model = model)))
bodies <- batch_bodies(texts, 100L, embedding_body)
with_batch_mock(
  bodies,
  measure("embed_batch_errors", 1000L, function() {
    hf_embed_batch(texts, model = model, progress = FALSE)
  }),
  failing = texts[c(201L, 701L)]
)
named <- stats::setNames(texts_for(250L), paste0("id", seq_len(250L)))
bodies <- batch_bodies(unname(named), 100L, embedding_body)
with_batch_mock(bodies, measure("embed_batch_named", 250L, function() {
  hf_embed_batch(named, model = model, progress = FALSE)
}, iterations = 1L))

# Compiled floor: jsonlite's C parser alone on the same 5,000-text payloads.
texts <- texts_for(5000L)
bodies <- batch_bodies(texts, 100L, embedding_body)
payloads <- mget(texts[seq.int(1L, 5000L, by = 100L)], envir = bodies)
measure("floor_json_parse_embed", 5000L, function() {
  lapply(payloads, function(raw) jsonlite::fromJSON(rawToChar(raw), simplifyVector = FALSE))
}, keep_output = FALSE)
rm(payloads, bodies)

# Text classification in batches, and through hf_classify().
for (n in c(100L, 5000L)) {
  texts <- texts_for(n)
  bodies <- batch_bodies(texts, 100L, classification_body)
  with_batch_mock(bodies, measure("classify_batch", n, function() {
    hf_classify_batch(texts, model = model, progress = FALSE)
  }))
}
texts <- texts_for(1000L)
bodies <- batch_bodies(texts, 1000L, classification_body)
with_batch_mock(bodies, measure("classify", 1000L, function() {
  hf_classify(texts, model = model)
}))
bodies <- batch_bodies(texts, 100L, classification_body)
with_batch_mock(
  bodies,
  measure("classify_batch_errors", 1000L, function() {
    hf_classify_batch(texts, model = model, progress = FALSE)
  }),
  failing = texts[c(101L, 901L)]
)

# Zero-shot classification: one request per text, mixing current, legacy,
# empty, integer-scored, and failed responses.
zero_shot_labels <- c("technology", "sports", "food")
zero_shot_body <- function(i) {
  scores <- round(c(0.6, 0.3, 0.1) + (i %% 5L) / 100 * c(-1, 0.5, 0.5), 4)
  body <- if (i %% 11L == 0L) {
    "[]"
  } else if (i %% 7L == 0L) {
    as.character(jsonlite::toJSON(list(labels = zero_shot_labels, scores = scores), digits = NA))
  } else if (i %% 17L == 0L) {
    '[{"label":"technology","score":1},{"label":"sports","score":0},{"label":"food","score":0}]'
  } else {
    as.character(jsonlite::toJSON(
      data.frame(label = zero_shot_labels, score = scores),
      dataframe = "rows", digits = NA
    ))
  }
  charToRaw(body)
}
for (n in c(100L, 1000L)) {
  texts <- texts_for(n)
  bodies <- batch_bodies(texts, 1L, zero_shot_body)
  with_batch_mock(
    bodies,
    measure("zero_shot_batch", n, function() {
      hf_classify_zero_shot_batch(texts, zero_shot_labels, model = model, progress = FALSE)
    }, iterations = if (n > 100L) 3L else 5L),
    failing = texts[seq(13L, n, by = 13L)]
  )
}
named <- stats::setNames(texts_for(60L), paste0("id", seq_len(60L)))
bodies <- batch_bodies(unname(named), 1L, zero_shot_body)
with_batch_mock(bodies, measure("zero_shot_batch_named", 60L, function() {
  hf_classify_zero_shot_batch(named, zero_shot_labels, model = model, progress = FALSE)
}, iterations = 1L))

# Unusual embedding response shapes, checked for identical output only.
shape_bodies <- new.env(hash = TRUE)
assign("token-a", json_raw(list(
  list(c(0.1, 0.2), c(0.3, 0.4)),
  list(c(0.5, 0.6), c(0.7, 0.8)),
  list(c(0.9, 1.1), c(1.2, 1.3))
)), envir = shape_bodies)
assign("flat-a", json_raw(c(0.1, 0.2, 0.3)), envir = shape_bodies)
assign("short-a", json_raw(list(c(0.1, 0.2))), envir = shape_bodies)
with_batch_mock(shape_bodies, {
  measure("embed_shape_tokens", 3L, function() {
    hf_embed_batch(c("token-a", "token-b", "token-c"), model = model, progress = FALSE)
  }, iterations = 1L)
  measure("embed_shape_flat", 1L, function() {
    hf_embed_batch("flat-a", model = model, progress = FALSE)
  }, iterations = 1L)
  measure("embed_shape_short", 3L, function() {
    suppressWarnings(hf_embed_batch(c("short-a", "short-b", "short-c"),
                                    model = model, progress = FALSE))
  }, iterations = 1L)
})

# Dataset rows: paginated datasets-server responses with scalar, missing,
# null, variable-length, and nested fields.
dataset_row <- function(i) {
  row <- list(
    id = i,
    text = paste("Synthetic dataset row", i),
    label = i %% 3L,
    score = round(i / 7, 4),
    tags = as.list(rep(c("a", "b", "c"), length.out = i %% 4L)),
    meta = list(source = "synthetic", position = i)
  )
  if (i %% 5L == 0L) {
    row["note"] <- list(NULL)
  } else if (i %% 5L == 1L) {
    row$note <- paste("note", i)
  }
  row
}
dataset_pages <- function(n) {
  pages <- new.env(hash = TRUE)
  for (offset in seq.int(0L, n - 1L, by = 100L)) {
    ids <- offset + seq_len(min(100L, n - offset)) - 1L
    payload <- list(rows = lapply(ids, function(i) {
      list(row_idx = i, row = dataset_row(i), truncated_cells = list())
    }))
    assign(as.character(offset), json_raw(payload, auto_unbox = TRUE, null = "null"),
           envir = pages)
  }
  pages
}
for (n in c(1000L, 10000L)) {
  pages <- dataset_pages(n)
  with_http_mock(function(req) {
    stopifnot(startsWith(req$url, "https://datasets-server.huggingface.co/rows?"))
    offset <- httr2::url_parse(req$url)$query$offset
    json_response(req, get(offset, envir = pages, inherits = FALSE))
  }, measure("load_dataset", n, function() {
    hf_load_dataset("synthetic/rows", split = "train", config = "default", limit = n)
  }, iterations = 3L))
}
rows_json <- rawToChar(json_raw(lapply(0:49999, dataset_row), auto_unbox = TRUE, null = "null"))
rows <- jsonlite::fromJSON(rows_json, simplifyVector = FALSE)
measure("rows_to_tibble", 50000L, function() hf$hf_rows_to_tibble(rows), iterations = 3L)
measure("floor_json_parse_rows", 50000L, function() {
  jsonlite::fromJSON(rows_json, simplifyVector = FALSE)
}, iterations = 3L, keep_output = FALSE)
rm(rows, rows_json)

# Hub listings: cursor-paginated model search and one large file tree.
model_record <- function(i) {
  list(
    `_id` = sprintf("%024d", i),
    id = sprintf("org-%d/model-%06d", i %% 50L, i),
    author = if (i %% 2L == 0L) sprintf("org-%d", i %% 50L),
    likes = i %% 97L,
    downloads = if (i %% 101L == 0L) 3e9 + i else i * 7L,
    tags = as.list(c("transformers", "pytorch", if (i %% 2L == 1L) "en")),
    pipeline_tag = if (i %% 5L != 0L) "text-classification",
    library_name = "transformers",
    createdAt = "2024-01-01T00:00:00.000Z"
  )
}
for (n in c(1000L, 5000L)) {
  pages <- lapply(seq_len(ceiling(n / 100)), function(page) {
    ids <- ((page - 1L) * 100L + 1L):min(page * 100L, n)
    json_raw(lapply(ids, model_record), auto_unbox = TRUE, null = "null")
  })
  with_http_mock(function(req) {
    stopifnot(startsWith(req$url, "https://huggingface.co/api/models?"))
    cursor <- httr2::url_parse(req$url)$query$cursor
    page <- if (is.null(cursor)) 1L else as.integer(cursor)
    link <- if (page < length(pages)) {
      list(Link = paste0("<https://huggingface.co/api/models?cursor=", page + 1L,
                         '>; rel="next"'))
    } else {
      list()
    }
    json_response(req, pages[[page]], link)
  }, measure("search_models", n, function() hf_search_models(limit = n), iterations = 3L))
}
for (n in c(5000L, 20000L)) {
  files <- json_raw(lapply(seq_len(n), function(i) {
    if (i %% 50L == 0L) {
      list(type = "directory", oid = sprintf("%040d", i), path = sprintf("data/dir-%06d", i))
    } else {
      list(
        type = "file", oid = sprintf("%040d", i),
        size = if (i %% 333L == 0L) 5e9 + i else i * 1024L,
        path = sprintf("data/file-%06d.parquet", i)
      )
    }
  }), auto_unbox = TRUE)
  with_http_mock(function(req) {
    stopifnot(startsWith(req$url, "https://huggingface.co/api/models/org/big-repo/tree/"))
    json_response(req, files)
  }, measure("list_repo_files", n, function() hf_list_repo_files("org/big-repo"),
             iterations = 3L))
}

# Recipe step: bake 1,000 texts into 384 embedding columns.
texts <- texts_for(1000L)
bodies <- batch_bodies(texts, 1000L, embedding_body)
train <- data.frame(text = texts, y = seq_along(texts), stringsAsFactors = FALSE)
with_batch_mock(bodies, {
  rec <- huggingfaceR::step_hf_embed(recipes::recipe(y ~ text, data = train), text,
                                     model = model)
  prepped <- recipes::prep(rec, training = train)
  measure("recipe_bake", 1000L, function() recipes::bake(prepped, new_data = train))
})

saveRDS(outputs, file.path(output_dir, "outputs", paste0(label, "-outputs.rds")))
metrics <- do.call(rbind, unname(metrics))
stopifnot(all(is.finite(metrics$median_s)), all(metrics$median_s > 0))
utils::write.csv(metrics, file.path(output_dir, paste0(label, "-metrics.csv")),
                 row.names = FALSE)
packages <- c("huggingfaceR", "httr2", "jsonlite", "tibble", "dplyr", "purrr", "vctrs",
              "recipes", "bench")
jsonlite::write_json(
  list(
    version = label,
    source_commit = source_commit,
    package_path = find.package("huggingfaceR"),
    R_version = as.character(getRversion()),
    BLAS = unname(extSoftVersion()["BLAS"]),
    LAPACK = La_library(),
    packages = as.list(vapply(packages, function(pkg) {
      as.character(utils::packageVersion(pkg))
    }, character(1))),
    system = as.list(Sys.info()[c("sysname", "release", "machine")]),
    timing_scope = paste(
      "Warm R-side cost with HTTP mocked: request construction, JSON parsing,",
      "result shaping, and local numeric work. Network I/O and server-side",
      "inference are excluded by design."
    ),
    floor_scope = paste(
      "floor_* rows time only the compiled routine a C client would also run",
      "(jsonlite's C parser or the BLAS cross-product) on the same data."
    ),
    allocation_scope = "R allocations only, not peak RSS.",
    interpretation = "Descriptive paired-run measurements on a shared runner."
  ),
  file.path(output_dir, paste0(label, "-metadata.json")),
  pretty = TRUE, auto_unbox = TRUE, na = "null", null = "null"
)
print(metrics, row.names = FALSE)
