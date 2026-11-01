# Try to set device to GPU for accelerated computation

This function currently depends on having a working installation of
torch for your GPU in this environment. If running an Apple silicon GPU,
you'll need the native mac M+ build (ARM binary). You will also need
rust and other transformers dependencies. As you need to make sure that
everything that needs to be on the GPU (tensors, model, pipeline etc.),
is on the GPU, we currently recommend this for advanced users only. We
will be working on integrating this fully with the installation and
build of the huggingfaceR environment.

## Usage

``` r
hf_set_device()
```

## Value

a device that models, pipelines, and tensors can be sent to.

## Examples

``` r
if (FALSE) { # \dontrun{
device <- hf_set_device()
} # }
```
