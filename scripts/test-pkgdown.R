source("scripts/pkgdown.R")

expect_failure <- function(expr, message) {
  error <- tryCatch(force(expr), error = identity)
  stopifnot(
    inherits(error, "error"),
    grepl(message, conditionMessage(error), fixed = TRUE)
  )
}

stopifnot(identical(
  pkgdown_validate_version("2.4.0.9000", "dev"),
  "2.4.0.9000"
))
stopifnot(identical(
  pkgdown_validate_version("2.4.0", "stable", "v2.4.0", "2.4.0"),
  "2.4.0"
))
expect_failure(pkgdown_validate_version("2.4.0", "dev"), "four-component")
expect_failure(
  pkgdown_validate_version("2.4.0.9000", "stable", "v2.4.0.9000", "2.4.0"),
  "three-component"
)
expect_failure(
  pkgdown_validate_version("2.4.0", "stable", "v2.3.0", "2.4.0"),
  "exact DESCRIPTION"
)
expect_failure(
  pkgdown_validate_version("2.5.0", "stable", "v2.5.0", "2.4.0"),
  "current CRAN"
)
expect_failure(
  pkgdown_validate_version("2.3.0", "stable", "v2.3.0", "2.4.0"),
  "current CRAN"
)
expect_failure(pkgdown_validate_version("2.4.0.9000", "other"), "dev or stable")

local({
  root <- tempfile("pkgdown-guards-")
  dir.create(root)
  on.exit(unlink(root, recursive = TRUE))
  source <- file.path(root, "source")
  dir.create(source)
  stopifnot(file.copy(c("DESCRIPTION", "_pkgdown.yml", "NAMESPACE"), source))
  dir.create(file.path(source, "man"))
  stopifnot(all(file.copy(
    list.files("man", full.names = TRUE),
    file.path(source, "man"),
    recursive = TRUE
  )))
  desc <- read.dcf(file.path(source, "DESCRIPTION"))
  desc[1, "Version"] <- "2.4.0.9000"
  write.dcf(desc, file.path(source, "DESCRIPTION"))
  destination <- file.path(root, "site")
  pkg <- pkgdown_prepare(source, destination, "dev")
  stopifnot(identical(basename(pkg$dst_path), "dev"))
  dir.create(pkg$dst_path, recursive = TRUE)
  writeLines("development sentinel", file.path(pkg$dst_path, "sentinel"))
  desc <- read.dcf(file.path(source, "DESCRIPTION"))
  desc[1, "Version"] <- "2.4.0"
  write.dcf(desc, file.path(source, "DESCRIPTION"))
  missing_destination <- file.path(root, "not-created")
  stable <- pkgdown_prepare(
    source,
    missing_destination,
    "stable",
    "v2.4.0",
    "2.4.0"
  )
  stopifnot(
    !dir.exists(missing_destination),
    identical(basename(stable$dst_path), "not-created")
  )
  pkg <- pkgdown_prepare(source, destination, "stable", "v2.4.0", "2.4.0")
  stopifnot(identical(basename(pkg$dst_path), "site"))
  pkgdown::clean_site(pkg, quiet = TRUE)
  stopifnot(identical(
    readLines(file.path(destination, "dev", "sentinel")),
    "development sentinel"
  ))
  writeLines(
    '<small class="nav-text text-muted">2.4.0</small>',
    file.path(destination, "index.html")
  )
  stopifnot(identical(pkgdown_check_seed(destination, "2.4.0"), "2.4.0"))
  expect_failure(pkgdown_check_seed(destination, "2.5.0"), "current CRAN")
  pkg$version <- "2.5.0"
  expect_failure(
    pkgdown_check_site(pkg, "stable"),
    "does not match DESCRIPTION"
  )
  writeLines("No navbar version", file.path(destination, "index.html"))
  expect_failure(pkgdown_check_seed(destination, "2.4.0"), "navbar version")
  docs <- file.path(source, "docs")
  dir.create(file.path(docs, "articles", "figures"), recursive = TRUE)
  writeLines(
    '<small class="nav-text text-muted">2.4.0</small>',
    file.path(docs, "index.html")
  )
  writeBin(
    as.raw(0:255),
    file.path(docs, "articles", "figures", "executed.png")
  )
  writeLines("", file.path(docs, ".nojekyll"))
  artifact <- file.path(root, "stable-artifact")
  stopifnot(identical(
    pkgdown_stage_stable(source, artifact, "v2.4.0", "2.4.0"),
    "2.4.0"
  ))
  stopifnot(
    identical(
      readBin(
        file.path(artifact, "articles", "figures", "executed.png"),
        "raw",
        256
      ),
      as.raw(0:255)
    ),
    file.exists(file.path(artifact, ".nojekyll"))
  )
  expect_failure(
    pkgdown_stage_stable(source, artifact, "v2.4.0", "2.4.0"),
    "already exist"
  )
  expect_failure(
    pkgdown_stage_stable(
      source,
      file.path(root, "invalid-artifact"),
      "v2.4.0",
      "2.5.0"
    ),
    "current CRAN"
  )
})

cat("pkgdown publishing guards passed.\n")
