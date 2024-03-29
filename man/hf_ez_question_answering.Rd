% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/ez.R
\name{hf_ez_question_answering}
\alias{hf_ez_question_answering}
\title{Answer Questions about a Text based on Context}
\usage{
hf_ez_question_answering(
  model_id = "deepset/roberta-base-squad2",
  use_api = FALSE
)
}
\arguments{
\item{model_id}{A model_id. Run hf_search_models(...) for model_ids. Defaults to 'deepset/roberta-base-squad2'.}

\item{use_api}{Whether to use the Inference API to run the model (TRUE) or download and run the model locally (FALSE). Defaults to FALSE}
}
\value{
A question answering object
}
\description{
Want to have a nice know-it-all bot that can answer any question?
}
\examples{
\dontrun{
# Load the default model and use local inference
ez <- hf_ez_question_answering()
ez$infer(question = "What's my name?", context = "My name is Clara and I live in Berkeley.")

# Use the api for inference.
ez <- hf_ez_fill_mask(use_api = TRUE)
ez$infer(question = "What's my name?", context = "My name is Clara and I live in Berkeley.")
}
}
\seealso{
\url{https://huggingface.co/docs/api-inference/detailed_parameters#question-answering-task}
}
