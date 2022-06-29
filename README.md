
<!-- README.md is generated from README.Rmd. Please edit that file -->

# huggingfaceR

<!-- badges: start -->
<!-- badges: end -->

The goal of `huggingfaceR` is to bring state-of-the-art NLP models to
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

Once you have python installed and configured you need to ensure that
you have the `keras` python library installed.

``` r
py_install("keras")
```

After that you can install the development version of huggingfaceR from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("farach/huggingfaceR")
```

## Example

`huggingfaceR` makes use of the `transformer` `pipleine()` function to
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

We can use this model in a typical tidyverse processing chunk.

``` r
library(janeaustenr)
library(tidytext)
suppressMessages(library(tidyverse))

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
#>  1 NEGATIVE 0.939 "she had never been informed by themselves of the terms… Sens…
#>  2 NEGATIVE 0.998 "connections, and probably inferior in fortune to herse… Sens…
#>  3 NEGATIVE 0.725 "sister, who watched, with unremitting attention her co… Sens…
#>  4 POSITIVE 0.607 "Barton Park was about half a mile from the cottage.  T… Sens…
#>  5 NEGATIVE 0.992 "Mrs. Ferrars; and such ill-timed praise of another, at… Sens…
#>  6 NEGATIVE 0.999 "destroys the bloom for ever!  Hers has been a very sho… Sens…
#>  7 NEGATIVE 0.950 "the puppyism of his manner in deciding on all the diff… Sens…
#>  8 POSITIVE 1.00  "\"Indeed!\""                                            Sens…
#>  9 NEGATIVE 0.990 "objection was made against Edward's taking orders for … Sens…
#> 10 NEGATIVE 0.924 "\"Your poor mother, too!--doting on Marianne.\""        Sens…
#> 11 NEGATIVE 0.988 "Willoughby's letter, and, after shuddering over every … Sens…
#> 12 POSITIVE 1.00  "say, that understanding you mean to take orders, he ha… Sens…
#> 13 NEGATIVE 0.724 "into the room, were officiously handed by him to Colon… Sens…
#> 14 POSITIVE 0.999 "which though it did not give actual elegance or grace,… Sens…
#> 15 POSITIVE 0.954 "step into his carriage, and in a minute it was out of … Sens…
#> 16 POSITIVE 0.999 "between Marianne and Eliza already acknowledged, and n… Sens…
#> 17 NEGATIVE 0.991 "in his proper situation, and would have wanted for not… Sens…
#> 18 POSITIVE 0.982 "enjoyment only by the entrance of her four noisy child… Sens…
#> 19 POSITIVE 0.998 "You will be setting your cap at him now, and never thi… Sens…
#> 20 POSITIVE 0.767 "held her hand only for a moment.  During all this time… Sens…
```
