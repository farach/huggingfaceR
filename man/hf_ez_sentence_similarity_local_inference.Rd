% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/ez.R
\name{hf_ez_sentence_similarity_local_inference}
\alias{hf_ez_sentence_similarity_local_inference}
\title{Sentence Similarity Local Inference}
\usage{
hf_ez_sentence_similarity_local_inference(
  source_sentence,
  sentences,
  tidy = TRUE,
  ...
)
}
\arguments{
\item{source_sentence}{The string that you wish to compare the other strings with. This can be a phrase, sentence, or longer passage, depending on the model being used.}

\item{sentences}{A list of strings which will be compared against the source_sentence.}

\item{tidy}{Whether to tidy the results into a tibble. Default: TRUE (tidy the results)}
}
\value{
The results of the inference
}
\description{
Sentence Similarity Local Inference
}
\seealso{
\url{https://huggingface.co/docs/transformers/main/en/pipeline_tutorial}
}
