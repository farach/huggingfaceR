
<!-- README.md is generated from README.Rmd. Please edit that file -->

# huggingfaceR

<!-- badges: start -->
<!-- badges: end -->

The goal of `huggingfaceR` is to to bring state-of-the-art NLP models to
R. `huggingfaceR` is built on top of Hugging Face’s
[transformer](https://huggingface.co/docs/transformers/index) library.

## Installation

Prior to installing `huggingfaceR` please be sure to have your python
environment set up correctly.

``` r
install.packages("reticulate")
library(reticulate)

install_miniconda()
```

If you are having issues, more detailed instructions on how to install
and configure python can be found
[here](https://support.rstudio.com/hc/en-us/articles/360023654474-Installing-and-Configuring-Python-with-RStudio).

After that you can install the development version of huggingfaceR from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("farach/huggingfaceR")
```

## Example

`huggingfaceR` makes use of the `transformer` `pipline()` function to
quickly make pre-trained models available for use in R. In this example
we will load the `distilbert-base-uncased-finetuned-sst-2-english` model
to obtain sentiment scores.

``` r
library(huggingfaceR)

distilBERT <- hf_load_pipeline("distilbert-base-uncased-finetuned-sst-2-english")
#> 
#> 
#> distilbert-base-uncased-finetuned-sst-2-english is ready for text-classification
```

With the model now loaded, we can begin using the model.

``` r
distilBERT("I like you. I love you")
#> [[1]]
#> [[1]]$label
#> [1] "POSITIVE"
#> 
#> [[1]]$score
#> [1] 0.9998739
```

We can use this model in a typical tidyverse processing chunk. First we
load the `tidyverse`.

``` r
library(tidyverse)
#> ── Attaching packages ─────────────────────────────────────── tidyverse 1.3.1 ──
#> ✔ ggplot2 3.3.6     ✔ purrr   0.3.4
#> ✔ tibble  3.1.7     ✔ dplyr   1.0.9
#> ✔ tidyr   1.2.0     ✔ stringr 1.4.0
#> ✔ readr   2.1.2     ✔ forcats 0.5.1
#> ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
#> ✖ dplyr::filter() masks stats::filter()
#> ✖ dplyr::lag()    masks stats::lag()
```

We can use the `huggingfaceR` `hf_load_dataset()` function to pull in
the [emotion](https://huggingface.co/datasets/emotion) Hugging Face
dataset. This dataset contains English Twitter messages with six basic
emotions: anger, fear, love, sadness, and surprise. We are interested in
how well the Distilbert model classifies these emotions as either a
positive or a negative sentiment.

``` r
emotion <- hf_load_dataset(
  "emotion", 
  split = "test", 
  as_tibble = TRUE, 
  label_name = "int2str"
  )

emotion_model <- emotion |>
  transmute(
    text,
    emotion_id = label,
    emotion_name = label_name,
    distilBERT_sent = distilBERT(text)
  ) |>
  unnest_wider(distilBERT_sent)

glimpse(emotion_model)
#> Rows: 2,000
#> Columns: 5
#> $ text         <chr> "im feeling rather rotten so im not very ambitious right …
#> $ emotion_id   <dbl> 0, 0, 0, 1, 0, 4, 3, 1, 1, 3, 4, 0, 4, 1, 2, 0, 1, 0, 3, …
#> $ emotion_name <chr> "sadness", "sadness", "sadness", "joy", "sadness", "fear"…
#> $ label        <chr> "NEGATIVE", "NEGATIVE", "POSITIVE", "POSITIVE", "NEGATIVE…
#> $ score        <dbl> 0.9998109, 0.9994603, 0.9993082, 0.9868246, 0.9996492, 0.…
```

We can use `ggplot2` to visualize the results.

``` r
emotion_model |>
  mutate(
    label = paste0("Distilbert class:\n", label),
    emotion_name = str_to_title(emotion_name)
  ) |>
  ggplot(aes(x = emotion_name, y = score, color = label)) +
  geom_boxplot(show.legend = FALSE, outlier.alpha = 0.4, ) +
  scale_color_manual(values = c("#D55E00", "#6699CC")) +
  facet_wrap(~ label) +
  labs(
    title = "Reviewing Distilbert classification predictions",
    x = "Original label",
    y = "Model score",
    caption = "source:\nhttps://huggingface.co/datasets/emotion"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    axis.text.x = element_text(angle = 45),
    axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)),
    axis.title.x = element_text(margin = margin(t = 10, r = 0, b = 0, l = 0))
  )
```

<img src="man/figures/README-unnamed-chunk-6-1.png" width="100%" />
