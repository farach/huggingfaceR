% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/ez.R
\name{hf_ez_zero_shot_classification_local_inference}
\alias{hf_ez_zero_shot_classification_local_inference}
\title{Zero Shot Classification Local Inference}
\usage{
hf_ez_zero_shot_classification_local_inference(
  string,
  candidate_labels,
  multi_label = FALSE,
  tidy = TRUE,
  ...
)
}
\arguments{
\item{string}{a string or list of strings}

\item{candidate_labels}{a list of strings that are potential classes for inputs. (max 10 candidate_labels, for more, simply run multiple requests, results are going to be misleading if using too many candidate_labels anyway. If you want to keep the exact same, you can simply run multi_label=True and do the scaling on your end. )}

\item{multi_label}{(Default: false) Boolean that is set to True if classes can overlap}
}
\value{
The results of the inference
}
\description{
Zero Shot Classification Local Inference
}
\seealso{
\url{https://huggingface.co/docs/transformers/main/en/pipeline_tutorial}
}
