% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/inference.R
\name{hf_text2text_generation_payload}
\alias{hf_text2text_generation_payload}
\title{Text2Text Generation Payload}
\usage{
hf_text2text_generation_payload(string)
}
\arguments{
\item{string}{a string containing a question or task and a sentence from which the answer is derived}
}
\value{
An inference payload
}
\description{
takes an input containing the sentence including the task and returns the output of the accomplished task.
}
\seealso{
\url{https://huggingface.co/docs/api-inference/detailed_parameters#text2text-generation-task}
}
