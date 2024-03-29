% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/huggingface_hub.R
\name{hf_list_models}
\alias{hf_list_models}
\title{List Models}
\usage{
hf_list_models(pattern = NULL)
}
\arguments{
\item{pattern}{A search term or regular expression. Defaults to NULL (return all results).}
}
\description{
List Model Names
}
\examples{
hf_list_models("bert-base-cased")
}
\seealso{
\url{https://huggingface.co/docs/hub/searching-the-hub}
}
