% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/inference.R
\name{hf_summarization_payload}
\alias{hf_summarization_payload}
\title{Summarization Task Payload}
\usage{
hf_summarization_payload(
  string,
  min_length = NULL,
  max_length = NULL,
  top_k = NULL,
  top_p = NULL,
  temperature = 1,
  repetition_penalty = NULL,
  max_time = NULL
)
}
\arguments{
\item{string}{a string to be summarized}

\item{min_length}{Integer to define the minimum length in tokens of the output summary. Default: NULL}

\item{max_length}{Integer to define the maximum length in tokens of the output summary. Default: NULL}

\item{top_k}{Integer to define the top tokens considered within the sample operation to create new text. Default: NULL}

\item{top_p}{Float to define the tokens that are within the sample operation of text generation. Add tokens in the sample for more probable to least probable until the sum of the probabilities is greater than top_p. Default: NULL}

\item{temperature}{Float (0.0-100.0). The temperature of the sampling operation. 1 means regular sampling, 0 means always take the highest score, 100.0 is getting closer to uniform probability. Default: 1.0}

\item{repetition_penalty}{Float (0.0-100.0). The more a token is used within generation the more it is penalized to not be picked in successive generation passes. Default: NULL}

\item{max_time}{Float (0-120.0). The amount of time in seconds that the query should take maximum. Network can cause some overhead so it will be a soft limit. Default: NULL}
}
\value{
An inference payload
}
\description{
This task is well known to summarize longer text into shorter text. Be careful, some models have a maximum length of input. That means that the summary cannot handle full books for instance.
}
\seealso{
\url{https://huggingface.co/docs/api-inference/detailed_parameters#summarization-task}
}
