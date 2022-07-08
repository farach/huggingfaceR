
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

distilBERT <- hf_load_model("distilbert-base-uncased-finetuned-sst-2-english")
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
load some libraries.

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
library(janeaustenr)
```

Here we get the sentiment score assigned to the text in “Sense &
Sensibility”.

``` r
austen_books() |>
  filter(
    book == "Sense & Sensibility",
    text != ""
    ) %>%
  sample_n(20) |>
  mutate(
    distilBERT_sent = distilBERT(text),
    .before = text
  ) |>
  unnest_wider(distilBERT_sent)
#> # A tibble: 20 × 4
#>    label    score text                                                     book 
#>    <chr>    <dbl> <chr>                                                    <fct>
#>  1 POSITIVE 1.00  "as good friends as ever.  Look, she made me this bow t… Sens…
#>  2 NEGATIVE 0.979 "\"Had he been only in a violent fever, you would not h… Sens…
#>  3 NEGATIVE 0.996 "\"Elinor,\" cried Marianne, \"is this fair? is this ju… Sens…
#>  4 POSITIVE 1.00  "exultation; \"we came post all the way, and had a very… Sens…
#>  5 NEGATIVE 1.00  "constant and painful exertion;--they did not spring up… Sens…
#>  6 NEGATIVE 0.971 "be as much as possible with Charlotte, she went thithe… Sens…
#>  7 NEGATIVE 0.795 "the daughter of a private gentleman with no more than … Sens…
#>  8 POSITIVE 1.00  "exquisite power of enjoyment.  She was perfectly dispo… Sens…
#>  9 NEGATIVE 0.981 "\"Perhaps, Miss Marianne,\" cried Lucy, eager to take … Sens…
#> 10 NEGATIVE 0.999 "and so much of his ill-will was done away, that when w… Sens…
#> 11 NEGATIVE 0.999 "father was by this arrangement rendered impracticable.… Sens…
#> 12 NEGATIVE 0.850 "not be spared; Sir John would not hear of their going;… Sens…
#> 13 NEGATIVE 0.989 "composure, she seemed scarcely to notice it.  I could … Sens…
#> 14 POSITIVE 0.973 "returning to town, procured the forgiveness of Mrs. Fe… Sens…
#> 15 POSITIVE 1.00  "saloon.'  Lady Elliott was delighted with the thought.… Sens…
#> 16 NEGATIVE 0.970 "never have reason to repent it.  Your case is a very u… Sens…
#> 17 POSITIVE 0.999 "picturesque beauty was.  I detest jargon of every kind… Sens…
#> 18 NEGATIVE 0.892 "but he did not know that any more was required: to be … Sens…
#> 19 POSITIVE 0.905 "of the party to get her a good hand.  If dancing forme… Sens…
#> 20 NEGATIVE 0.998 "your difficulties will soon vanish.\""                  Sens…
```
