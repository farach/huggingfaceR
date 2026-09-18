# unknown constructor interfaces fail rather than guessing an argument

    Code
      hf_local_load_backend("snapshot", "embed", "cpu")
    Condition
      Error:
      ! The installed SentenceTransformer constructor exposes neither processor_kwargs nor tokenizer_kwargs.

