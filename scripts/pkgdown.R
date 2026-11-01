pkgdown_validate_version <- function(
  version,
  target,
  tag = "",
  cran_version = ""
) {
  if (!target %in% c("dev", "stable")) {
    stop("Documentation target must be dev or stable.")
  }
  if (target == "dev") {
    if (!grepl("^[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+$", version)) {
      stop("Development publishing requires a four-component package version.")
    }
  } else {
    if (!grepl("^[0-9]+\\.[0-9]+\\.[0-9]+$", version)) {
      stop("Stable publishing requires a three-component release version.")
    }
    if (!identical(tag, paste0("v", version))) {
      stop("Stable tag must be v followed by the exact DESCRIPTION version.")
    }
    if (!identical(version, cran_version)) {
      stop(
        "Stable version must equal the current CRAN version; refusing a rollback or unreleased build."
      )
    }
  }
  invisible(version)
}

pkgdown_cran_version <- function() {
  packages <- available.packages(
    contriburl = contrib.url("https://cloud.r-project.org", type = "source"),
    filters = list()
  )
  if (!"huggingfaceR" %in% rownames(packages)) {
    stop("huggingfaceR is missing from the CRAN package index.")
  }
  unname(packages["huggingfaceR", "Version"])
}

pkgdown_prepare <- function(
  source,
  destination,
  target,
  tag = "",
  cran_version = ""
) {
  version <- unname(read.dcf(file.path(source, "DESCRIPTION"))[1, "Version"])
  pkgdown_validate_version(version, target, tag, cran_version)
  config <- yaml::read_yaml(file.path(source, "_pkgdown.yml"))
  publishing <- yaml::read_yaml("_pkgdown.yml")
  # Keep the release's reference and article indexes, not main's newer API.
  config$development <- publishing$development
  config$navbar$components$versions <- publishing$navbar$components$versions
  config$navbar$structure$right <- unique(c(
    "versions",
    config$navbar$structure$right
  ))
  config$template$includes$before_body <- publishing$template$includes$before_body
  config$destination <- normalizePath(
    destination,
    winslash = "/",
    mustWork = FALSE
  )
  yaml::write_yaml(config, file.path(source, "_pkgdown.yml"))
  pkg <- pkgdown::as_pkgdown(source)
  expected <- file.path(config$destination, if (target == "dev") "dev" else "")
  if (
    !identical(
      normalizePath(pkg$dst_path, winslash = "/", mustWork = FALSE),
      normalizePath(expected, winslash = "/", mustWork = FALSE)
    )
  ) {
    stop(
      "pkgdown destination does not match the requested documentation target."
    )
  }
  pkg
}

pkgdown_site_version <- function(index) {
  if (!file.exists(index)) {
    stop("Documentation has no index.html.")
  }
  html <- paste(readLines(index, warn = FALSE), collapse = "\n")
  badge <- regmatches(
    html,
    regexpr("<small[^>]*class=\"nav-text[^>]*>[0-9.]+</small>", html)
  )
  version <- sub(".*>([0-9.]+)</small>$", "\\1", badge)
  if (length(version) != 1L) {
    stop("Cannot verify the documentation navbar version.")
  }
  version
}

pkgdown_check_site <- function(pkg, target) {
  index <- file.path(pkg$dst_path, "index.html")
  if (!identical(pkgdown_site_version(index), pkg$version)) {
    stop("Built documentation navbar version does not match DESCRIPTION.")
  }
  html <- paste(readLines(index, warn = FALSE), collapse = "\n")
  required <- c(
    pkg$version,
    "CRAN stable",
    "Development (main)",
    "https://farach.github.io/huggingfaceR/dev/",
    if (target == "dev") {
      "Development documentation.</strong>"
    } else {
      "CRAN stable documentation.</strong>"
    }
  )
  if (
    !all(vapply(required, \(text) grepl(text, html, fixed = TRUE), logical(1)))
  ) {
    stop("Built homepage is missing version information or version links.")
  }
  noindex <- grepl('<meta name="robots" content="noindex', html, fixed = TRUE)
  if (!identical(noindex, target == "dev")) {
    stop("Search engine indexing does not match the documentation target.")
  }
  if (target == "dev") {
    topics <- c(
      "hf_dataset_search",
      "hf_dataset_filter",
      "hf_dataset_statistics",
      "hf_dataset_validity"
    )
    if (
      !all(file.exists(file.path(
        pkg$dst_path,
        "reference",
        paste0(topics, ".html")
      )))
    ) {
      stop(
        "Development documentation is missing dataset viewer reference pages."
      )
    }
  }
  invisible(index)
}

pkgdown_check_seed <- function(path, cran_version) {
  version <- pkgdown_site_version(file.path(path, "index.html"))
  pkgdown_validate_version(
    version,
    "stable",
    paste0("v", version),
    cran_version
  )
  invisible(version)
}
