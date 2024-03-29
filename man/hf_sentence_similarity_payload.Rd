% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/inference.R
\name{hf_sentence_similarity_payload}
\alias{hf_sentence_similarity_payload}
\title{Sentence Similarity Payload}
\usage{
hf_sentence_similarity_payload(source_sentence, sentences)
}
\arguments{
\item{source_sentence}{The string that you wish to compare the other strings with. This can be a phrase, sentence, or longer passage, depending on the model being used.}

\item{sentences}{A list of strings which will be compared against the source_sentence.}
}
\value{
An inference payload
}
\description{
Calculate the semantic similarity between one text and a list of other sentences by comparing their embeddings.
}
\seealso{
\url{https://huggingface.co/docs/api-inference/detailed_parameters#sentence-similarity-task}
}
