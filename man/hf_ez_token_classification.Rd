% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/ez.R
\name{hf_ez_token_classification}
\alias{hf_ez_token_classification}
\title{Classify parts of a Text}
\usage{
hf_ez_token_classification(
  model_id = "dbmdz/bert-large-cased-finetuned-conll03-english",
  use_api = FALSE
)
}
\arguments{
\item{model_id}{A model_id. Run hf_search_models(...) for model_ids. Defaults to 'dbmdz/bert-large-cased-finetuned-conll03-english'.}

\item{use_api}{Whether to use the Inference API to run the model (TRUE) or download and run the model locally (FALSE). Defaults to FALSE}
}
\value{
A text2text generation object
}
\description{
Usually used for sentence parsing, either grammatical, or Named Entity Recognition (NER) to understand keywords contained within text.
}
\examples{
\dontrun{
# Load the default named entity recognition model
ez <- hf_ez_token_classification()

# Run NER. Note how the full name is aggregated into one named entity.
ez$infer(string = "My name is Sarah Jessica Parker but you can call me Jessica", aggregation_strategy = 'simple')

# Run NER without aggregation. Note how the full name is separated into distinct named entities.
ez$infer(string = "My name is Sarah Jessica Parker but you can call me Jessica", aggregation_strategy = 'none')
}
}
\seealso{
\url{https://huggingface.co/docs/api-inference/detailed_parameters#token-classification-task}
}
