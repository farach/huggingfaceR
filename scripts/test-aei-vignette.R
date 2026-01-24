# Test Script for anthropic-economic-index.Rmd Vignette
#
# Executes all code from the vignette with reduced sample sizes to verify
# correctness without excessive API calls or runtime.
#
# Usage:
#   Rscript scripts/test-aei-vignette.R
#
# Prerequisites:
#   - A valid Hugging Face token in .Renviron (HUGGING_FACE_HUB_TOKEN)
#   - Packages: huggingfaceR, dplyr, tidyr, ggplot2, readr, uwot, tidytext

devtools::load_all()
library(dplyr)
library(tidyr)
library(ggplot2)
library(readr)

# --- Token discovery ---
if (Sys.getenv("HUGGING_FACE_HUB_TOKEN") == "") {
  renviron_paths <- c(
    file.path(Sys.getenv("HOME"), ".Renviron"),
    file.path(Sys.getenv("USERPROFILE"), "Documents", ".Renviron"),
    file.path(Sys.getenv("USERPROFILE"),
              "OneDrive - Microsoft", "Documents", ".Renviron"),
    file.path(path.expand("~"), ".Renviron")
  )
  for (p in renviron_paths) {
    if (file.exists(p)) {
      readRenviron(p)
      if (Sys.getenv("HUGGING_FACE_HUB_TOKEN") != "") break
    }
  }
  if (Sys.getenv("HUGGING_FACE_HUB_TOKEN") == "") {
    stop("No HUGGING_FACE_HUB_TOKEN found.")
  }
}

# --- Test infrastructure ---
results <- list()
test <- function(name, expr) {
  cat(sprintf("  %-55s", name))
  result <- tryCatch(
    {
      val <- eval(expr)
      cat("[PASS]\n")
      results[[name]] <<- list(status = "PASS", value = val)
      val
    },
    error = function(e) {
      cat(sprintf("[FAIL] %s\n", conditionMessage(e)))
      results[[name]] <<- list(status = "FAIL", error = conditionMessage(e))
      NULL
    }
  )
  invisible(result)
}

check <- function(condition, msg = "assertion failed") {
  if (!isTRUE(condition)) stop(msg, call. = FALSE)
}

cat("=== AEI Vignette Test (anthropic-economic-index.Rmd) ===\n\n")

# ============================================================
# Section: Loading the Dataset
# ============================================================
cat("Loading the Dataset\n")

test("Load task_statements from HF", {
  base_url <- paste0(
    "https://huggingface.co/datasets/Anthropic/EconomicIndex/",
    "resolve/main/release_2025_02_10/"
  )
  task_statements <<- read_csv(paste0(base_url, "onet_task_statements.csv"),
                               show_col_types = FALSE)
  check(tibble::is_tibble(task_statements), "expected tibble")
  check(nrow(task_statements) > 19000, "expected 19000+ rows")
  check(all(c("O*NET-SOC Code", "Title", "Task ID", "Task", "Task Type")
            %in% names(task_statements)))
  task_statements
})

test("Load task_usage from HF", {
  task_usage <<- read_csv(paste0(base_url, "onet_task_mappings.csv"),
                          show_col_types = FALSE)
  check(tibble::is_tibble(task_usage), "expected tibble")
  check(nrow(task_usage) > 3000, "expected 3000+ rows")
  check(all(c("task_name", "pct") %in% names(task_usage)))
  task_usage
})

test("Load auto_augment from HF", {
  auto_augment <<- read_csv(paste0(base_url, "automation_vs_augmentation.csv"),
                            show_col_types = FALSE)
  check(tibble::is_tibble(auto_augment), "expected tibble")
  check(nrow(auto_augment) == 6, "expected 6 rows")
  check(all(c("interaction_type", "pct") %in% names(auto_augment)))
  auto_augment
})

test("Load wages from HF", {
  wages <<- read_csv(paste0(base_url, "wage_data.csv"),
                     show_col_types = FALSE)
  check(tibble::is_tibble(wages), "expected tibble")
  check(nrow(wages) > 1000, "expected 1000+ rows")
  check(all(c("SOCcode", "JobName", "MedianSalary") %in% names(wages)))
  wages
})

# ============================================================
# Section: Semantic Embeddings of Occupational Tasks
# ============================================================
cat("\nSemantic Embeddings\n")

test("Join with case normalization produces matches", {
  tasks_with_usage <<- task_statements |>
    select(task_id = `Task ID`, task = Task, title = Title,
           soc_code = `O*NET-SOC Code`, task_type = `Task Type`) |>
    mutate(task_lower = tolower(task)) |>
    inner_join(
      task_usage |> mutate(task_lower = tolower(task_name)),
      by = "task_lower"
    ) |>
    select(-task_lower, -task_name) |>
    rename(ai_usage_pct = pct)
  check(nrow(tasks_with_usage) > 3000,
        sprintf("expected 3000+ rows, got %d", nrow(tasks_with_usage)))
  tasks_with_usage
})

test("Sample tasks across quartiles", {
  set.seed(42)
  # Use n=5 per quartile for testing (vignette uses 50)
  sample_tasks <<- tasks_with_usage |>
    mutate(usage_quartile = ntile(ai_usage_pct, 4)) |>
    group_by(usage_quartile) |>
    slice_sample(n = 5) |>
    ungroup()
  check(nrow(sample_tasks) == 20, "expected 20 sampled rows")
  check(length(unique(sample_tasks$usage_quartile)) == 4, "expected 4 quartiles")
  sample_tasks
})

test("hf_embed on sample tasks", {
  task_embeddings <<- hf_embed(sample_tasks$task)
  check(tibble::is_tibble(task_embeddings), "expected tibble")
  check(nrow(task_embeddings) == 20, "expected 20 rows")
  check(all(c("text", "embedding", "n_dims") %in% names(task_embeddings)))
  check(!is.na(task_embeddings$n_dims[1]), "expected non-NA n_dims")
  check(task_embeddings$n_dims[1] == 384, "expected 384 dimensions")
  task_embeddings
})

# ============================================================
# Section: Measuring Task Similarity
# ============================================================
cat("\nMeasuring Task Similarity\n")

test("hf_similarity on subset", {
  analytical_tasks <- task_embeddings |> slice(1:5)
  sim_result <- hf_similarity(analytical_tasks)
  check(tibble::is_tibble(sim_result), "expected tibble")
  # 5 choose 2 = 10 pairs

  check(nrow(sim_result) == 10, sprintf("expected 10 pairs, got %d", nrow(sim_result)))
  check(all(c("text_1", "text_2", "similarity") %in% names(sim_result)))
  check(all(sim_result$similarity >= -1 & sim_result$similarity <= 1),
        "similarity should be in [-1, 1]")
  sim_result
})

# ============================================================
# Section: Nearest Neighbor Search
# ============================================================
cat("\nNearest Neighbor Search\n")

test("hf_embed_text on sample_tasks", {
  task_docs <<- sample_tasks |>
    select(task, ai_usage_pct, title) |>
    hf_embed_text(task)
  check(tibble::is_tibble(task_docs), "expected tibble")
  check("embedding" %in% names(task_docs), "expected embedding column")
  check(nrow(task_docs) == 20, "expected 20 rows")
  check(!is.null(task_docs$embedding[[1]]), "first embedding should not be NULL")
  task_docs
})

test("hf_nearest_neighbors: writing and editing", {
  nn_result <- hf_nearest_neighbors(task_docs, "writing and editing documents", k = 3)
  check(tibble::is_tibble(nn_result), "expected tibble")
  check(nrow(nn_result) == 3, "expected 3 neighbors")
  check("similarity" %in% names(nn_result), "expected similarity column")
  check(all(nn_result$similarity > 0), "expected positive similarities")
  nn_result
})

test("hf_nearest_neighbors: quantitative data analysis", {
  nn_result <- hf_nearest_neighbors(task_docs, "quantitative data analysis", k = 3)
  check(nrow(nn_result) == 3, "expected 3 neighbors")
  nn_result
})

test("hf_nearest_neighbors: interpersonal communication", {
  nn_result <- hf_nearest_neighbors(task_docs, "interpersonal communication", k = 3)
  check(nrow(nn_result) == 3, "expected 3 neighbors")
  nn_result
})

# ============================================================
# Section: Clustering Tasks by Semantic Content
# ============================================================
cat("\nClustering Tasks\n")

test("hf_cluster_texts k=3", {
  # Use k=3 (smaller than vignette's k=6) since we only have 20 texts
  clustered_tasks <<- hf_cluster_texts(task_docs, k = 3)
  check("cluster" %in% names(clustered_tasks), "expected cluster column")
  check(length(unique(clustered_tasks$cluster)) == 3, "expected 3 clusters")
  check(nrow(clustered_tasks) == 20, "expected 20 rows")
  clustered_tasks
})

test("Cluster summary statistics", {
  cluster_summary <- clustered_tasks |>
    group_by(cluster) |>
    summarize(
      n_tasks = n(),
      mean_ai_usage = mean(ai_usage_pct, na.rm = TRUE),
      example_task = first(task),
      .groups = "drop"
    ) |>
    arrange(desc(mean_ai_usage))
  check(nrow(cluster_summary) == 3, "expected 3 cluster rows")
  check(all(c("n_tasks", "mean_ai_usage", "example_task") %in% names(cluster_summary)))
  cat("\n")
  print(cluster_summary |> select(cluster, n_tasks, mean_ai_usage))
  cat("  ")
  cluster_summary
})

test("hf_extract_topics", {
  # Requires tidytext
  if (!requireNamespace("tidytext", quietly = TRUE)) {
    stop("tidytext not installed")
  }
  topics <- task_docs |>
    hf_extract_topics(text_col = "task", k = 3)
  check(tibble::is_tibble(topics), "expected tibble")
  check("topic_terms" %in% names(topics), "expected topic_terms column")
  check("cluster" %in% names(topics), "expected cluster column")
  check(nrow(topics) == 3, "expected 3 topic rows")
  cat("\n")
  print(topics)
  cat("  ")
  topics
})

# ============================================================
# Section: Zero-Shot Classification
# ============================================================
cat("\nZero-Shot Classification\n")

test("Cognitive demand classification", {
  cognitive_labels <- c(
    "routine procedural work",
    "analytical reasoning",
    "creative problem solving",
    "interpersonal judgment"
  )
  high_usage_tasks <- sample_tasks |>
    filter(ai_usage_pct > quantile(ai_usage_pct, 0.75)) |>
    pull(task)
  check(length(high_usage_tasks) >= 5,
        sprintf("expected 5+ high-usage tasks, got %d", length(high_usage_tasks)))

  # Classify first 5 (vignette uses 20)
  cognitive_classes <- hf_classify_zero_shot(
    high_usage_tasks[1:5],
    labels = cognitive_labels
  )
  check(nrow(cognitive_classes) == 20, "expected 5 texts x 4 labels = 20 rows")
  check(all(c("text", "label", "score") %in% names(cognitive_classes)))

  label_counts <- cognitive_classes |>
    group_by(text) |>
    slice_max(score, n = 1) |>
    ungroup() |>
    count(label, sort = TRUE)
  check(nrow(label_counts) > 0, "expected at least 1 label")
  cat("\n")
  print(label_counts)
  cat("  ")
  label_counts
})

test("Automation potential classification", {
  automation_labels <- c(
    "fully automatable by AI",
    "partially automatable with human oversight",
    "requires significant human judgment",
    "cannot be performed by AI"
  )

  # Classify first 5 tasks (vignette uses 30)
  automation_classes <- hf_classify_zero_shot(
    sample_tasks$task[1:5],
    labels = automation_labels
  )
  check(nrow(automation_classes) == 20, "expected 5 texts x 4 labels = 20 rows")

  automation_summary <- automation_classes |>
    group_by(text) |>
    slice_max(score, n = 1) |>
    ungroup() |>
    left_join(
      sample_tasks |> select(task, ai_usage_pct),
      by = c("text" = "task")
    )
  check(nrow(automation_summary) == 5, "expected 5 rows after join")

  usage_by_label <- automation_summary |>
    group_by(label) |>
    summarize(
      n = n(),
      mean_actual_usage = mean(ai_usage_pct, na.rm = TRUE),
      .groups = "drop"
    ) |>
    arrange(desc(mean_actual_usage))
  check(nrow(usage_by_label) > 0, "expected results")
  cat("\n")
  print(usage_by_label)
  cat("  ")
  usage_by_label
})

# ============================================================
# Section: Visualizing the Task Embedding Space
# ============================================================
cat("\nVisualization (UMAP)\n")

test("UMAP projection with uwot", {
  if (!requireNamespace("uwot", quietly = TRUE)) {
    stop("uwot not installed")
  }
  library(uwot)

  emb_matrix <- do.call(rbind, task_docs$embedding)
  check(is.matrix(emb_matrix), "expected matrix")
  check(nrow(emb_matrix) == 20, "expected 20 rows")
  check(ncol(emb_matrix) == 384, "expected 384 cols")

  umap_coords <- umap(emb_matrix, n_neighbors = 15, min_dist = 0.1)
  check(is.matrix(umap_coords), "expected matrix from umap")
  check(ncol(umap_coords) == 2, "expected 2 columns")
  check(nrow(umap_coords) == 20, "expected 20 rows")

  plot_data <<- task_docs |>
    mutate(
      umap_1 = umap_coords[, 1],
      umap_2 = umap_coords[, 2]
    )
  check("umap_1" %in% names(plot_data))
  check("umap_2" %in% names(plot_data))
  plot_data
})

test("ggplot: AI usage colored UMAP", {
  p <- ggplot(plot_data, aes(x = umap_1, y = umap_2, color = ai_usage_pct)) +
    geom_point(alpha = 0.7, size = 2) +
    scale_color_viridis_c(
      name = "AI Usage %",
      labels = scales::percent_format(scale = 100)
    ) +
    labs(
      title = "Semantic Map of O*NET Tasks by AI Usage",
      subtitle = "UMAP projection of task embeddings colored by AI adoption rate",
      x = NULL, y = NULL
    ) +
    theme_minimal() +
    theme(
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      panel.grid = element_blank()
    )
  check(inherits(p, "ggplot"), "expected ggplot object")
  p
})

test("ggplot: Cluster colored UMAP", {
  p <- ggplot(
    plot_data |> left_join(clustered_tasks |> select(task, cluster), by = "task"),
    aes(x = umap_1, y = umap_2, color = factor(cluster))
  ) +
    geom_point(alpha = 0.7, size = 2) +
    labs(
      title = "Task Clusters in Embedding Space",
      subtitle = "K-means clusters projected via UMAP",
      color = "Cluster",
      x = NULL, y = NULL
    ) +
    theme_minimal() +
    theme(
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      panel.grid = element_blank()
    )
  check(inherits(p, "ggplot"), "expected ggplot object")
  p
})

# ============================================================
# Section: Linking AI Usage to Wage Data
# ============================================================
cat("\nWage Data Analysis\n")

test("Wage data join and plot", {
  occupation_wages <- wages |>
    filter(MedianSalary > 0, ChanceAuto >= 0) |>
    select(soc_code = SOCcode, job_name = JobName, job_family = JobFamily,
           median_salary = MedianSalary, chance_auto = ChanceAuto,
           job_zone = JobZone)
  check(nrow(occupation_wages) > 0, "expected wage rows after filter")

  occupation_usage <- tasks_with_usage |>
    group_by(soc_code, title) |>
    summarize(
      mean_ai_usage = mean(ai_usage_pct, na.rm = TRUE),
      n_tasks = n(),
      .groups = "drop"
    )
  check(nrow(occupation_usage) > 0, "expected usage rows")

  occupation_analysis <- occupation_usage |>
    inner_join(occupation_wages, by = "soc_code")
  check(nrow(occupation_analysis) > 100,
        sprintf("expected 100+ joined rows, got %d", nrow(occupation_analysis)))

  p <- ggplot(occupation_analysis, aes(x = median_salary, y = mean_ai_usage)) +
    geom_point(alpha = 0.4) +
    geom_smooth(method = "loess", se = TRUE) +
    scale_x_continuous(labels = scales::dollar_format()) +
    scale_y_continuous(labels = scales::percent_format(scale = 100)) +
    labs(
      title = "AI Usage by Median Salary",
      x = "Median Annual Salary",
      y = "Mean AI Usage Rate (across tasks)"
    ) +
    theme_minimal()
  check(inherits(p, "ggplot"), "expected ggplot object")
  cat(sprintf(" (%d occupations)", nrow(occupation_analysis)))
  p
})

# ============================================================
# Section: Collaboration Patterns
# ============================================================
cat("\nCollaboration Patterns\n")

test("auto_augment has expected structure", {
  check(nrow(auto_augment) == 6, "expected 6 rows")
  check(all(c("interaction_type", "pct") %in% names(auto_augment)))
  check("directive" %in% auto_augment$interaction_type)
  check("task iteration" %in% auto_augment$interaction_type)
  auto_augment
})

test("Interaction pattern zero-shot classification", {
  interaction_labels <- c(
    "giving direct instructions",
    "iterative refinement and feedback",
    "learning and understanding",
    "building upon previous output",
    "checking and validating work"
  )
  # Classify 3 tasks (vignette uses 20)
  interaction_classes <- hf_classify_zero_shot(
    sample_tasks$task[1:3],
    labels = interaction_labels,
    multi_label = TRUE
  )
  check(nrow(interaction_classes) > 0, "expected classification results")
  check(all(c("text", "label", "score") %in% names(interaction_classes)))
  interaction_classes
})

# ============================================================
# Section: Geographic Analysis (v3 Release)
# ============================================================
cat("\nGeographic Analysis (v3)\n")

test("Load enriched geographic data", {
  v3_url <- paste0(
    "https://huggingface.co/datasets/Anthropic/EconomicIndex/",
    "resolve/main/release_2025_09_15/data/output/"
  )
  geo_data <<- read_csv(
    paste0(v3_url, "aei_enriched_claude_ai_2025-08-04_to_2025-08-11.csv"),
    show_col_types = FALSE
  )
  check(tibble::is_tibble(geo_data), "expected tibble")
  check(nrow(geo_data) > 10000, "expected 10000+ rows")
  check(all(c("geo_id", "geography", "facet", "variable", "cluster_name", "value")
            %in% names(geo_data)))
  geo_data
})

test("Filter country-level task usage", {
  country_usage <- geo_data |>
    filter(
      geography == "country",
      facet == "onet_task",
      variable == "onet_task_pct",
      level == 0
    ) |>
    select(geo_id, cluster_name, value)
  check(nrow(country_usage) > 0,
        sprintf("expected country usage rows, got %d", nrow(country_usage)))

  top_tasks_by_country <<- country_usage |>
    group_by(geo_id) |>
    slice_max(value, n = 5) |>
    ungroup()
  check(nrow(top_tasks_by_country) > 0, "expected top tasks")
  cat(sprintf(" (%d countries)", length(unique(top_tasks_by_country$geo_id))))
  top_tasks_by_country
})

test("Embed unique geographic tasks", {
  unique_geo_tasks <- top_tasks_by_country |>
    distinct(cluster_name) |>
    slice(1:5) |>  # Limit to 5 for testing
    pull(cluster_name)
  check(length(unique_geo_tasks) > 0, "expected unique tasks")

  geo_task_embeddings <- hf_embed(unique_geo_tasks)
  check(nrow(geo_task_embeddings) == length(unique_geo_tasks))
  check(!is.na(geo_task_embeddings$n_dims[1]))

  # Test the join
  country_profiles <- top_tasks_by_country |>
    filter(cluster_name %in% unique_geo_tasks) |>
    left_join(
      geo_task_embeddings |> select(text, embedding),
      by = c("cluster_name" = "text")
    )
  check("embedding" %in% names(country_profiles))
  country_profiles
})

# ============================================================
# Section: Comparison with Conversational Approaches
# (Use tiny subset instead of all 19,000+ tasks)
# ============================================================
cat("\nComparison Section (reduced)\n")

test("Embed subset and find creative writing neighbors", {
  # Use only 20 tasks (vignette uses all 4000+)
  small_subset <- tasks_with_usage |> slice(1:20)
  all_embeddings <- small_subset |>
    hf_embed_text(task)
  check("embedding" %in% names(all_embeddings))
  check(nrow(all_embeddings) == 20)

  creative_tasks <- hf_nearest_neighbors(all_embeddings, "creative writing", k = 5)
  check(nrow(creative_tasks) == 5, "expected 5 neighbors")
  check("similarity" %in% names(creative_tasks))

  usage_stats <- creative_tasks |>
    summarize(
      mean_usage = mean(ai_usage_pct),
      median_usage = median(ai_usage_pct),
      sd_usage = sd(ai_usage_pct)
    )
  check(nrow(usage_stats) == 1)
  check(!is.na(usage_stats$mean_usage))
  cat(sprintf(" (mean_usage=%.4f)", usage_stats$mean_usage))
  usage_stats
})

# ============================================================
# Section: hf_embed_umap (alternative call)
# ============================================================
cat("\nhf_embed_umap alternative\n")

test("hf_embed_umap with small sample", {
  # Vignette uses 50 texts; use 5 for testing with reduced n_neighbors
  umap_result <- hf_embed_umap(sample_tasks$task[1:5], n_neighbors = 3)
  check(tibble::is_tibble(umap_result), "expected tibble")
  check(nrow(umap_result) == 5, "expected 5 rows")
  check(all(c("text", "umap_1", "umap_2") %in% names(umap_result)))
  check(!any(is.na(umap_result$umap_1)), "expected no NA in umap_1")
  umap_result
})

# ============================================================
# Summary
# ============================================================
cat("\n=== Results ===\n\n")

pass <- sum(sapply(results, function(r) r$status == "PASS"))
fail <- sum(sapply(results, function(r) r$status == "FAIL"))
total <- length(results)

cat(sprintf("  PASS: %d / %d\n", pass, total))
cat(sprintf("  FAIL: %d / %d\n", fail, total))

if (fail > 0) {
  cat("\nFailed tests:\n")
  for (name in names(results)) {
    if (results[[name]]$status == "FAIL") {
      cat(sprintf("  - %s: %s\n", name, results[[name]]$error))
    }
  }
  cat("\n")
  quit(status = 1)
} else {
  cat("\nAll tests passed.\n")
}
