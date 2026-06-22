# CLAUDE.md — Project conventions for huggingfaceR

Guidance for working on this package. Keep it short and current.

## Design principles

### Defaults must be beginner-friendly and broadly known

Every user-facing default — model IDs, example datasets, parameters — should be
the **easiest, most broadly understood option that gets a newcomer running with
the least friction**. The mental model is base R's `mtcars`: a default everyone
recognizes, that just works, and that lets people start experimenting in one
line.

Concretely, when choosing a default model:

- **Prefer broadly known, widely used models** over the technically "best" or
  newest one. Recognition and reliability beat marginal quality for a default.
- **Prefer the option that needs no extra arguments.** A default should work from
  a bare call, e.g. `hf_translate("Hello")`, without the user first learning
  codes, flags, or schemas. (This is why `hf_translate()` defaults to the
  Helsinki-NLP `opus-mt-en-fr` language-pair model — direction is in the model ID
  — rather than NLLB, which requires FLORES-200 codes.)
- **Prefer small, fast, low-cost models** for defaults so first calls are quick
  and cheap. Power users can always pass a larger `model`.
- Document the default's identity and behavior in the roxygen `@param`, and point
  to the heavier/more-capable alternative in `@details`/`@examples`.

The goal is the quickest path to people *actively using* the functions and
experimenting — easy onboarding first, depth second.

## R and docs verification

R is available in the Windows worktree used for the 2026-06 package expansion.
Before running package checks or docs builds, set:

```powershell
$env:LC_CTYPE = 'English_United States.UTF-8'
```

Useful commands that were actually run successfully:

```powershell
Rscript -e "devtools::document()"
Rscript -e "devtools::test()"
Rscript scripts/test-integration.R
Rscript scripts/test-readme.R
Rscript scripts/test-getting-started.R
Rscript scripts/test-multimodal-images-audio-speech.R
Rscript scripts/test-structured-extraction-and-tools.R
Rscript scripts/test-hub-download-upload-share.R
Rscript -e "rmarkdown::render('README.Rmd', clean = TRUE)"
Rscript -e "pkgdown::build_site()"
Rscript -e "pkgdown::check_pkgdown()"
Rscript -e "devtools::check(args = '--no-manual')"
```

Token lookup in the live scripts checks common `.Renviron` locations, including
`Documents/.Renviron` and `OneDrive - Microsoft/Documents/.Renviron`.

## Package-specific gotchas

- `air format .` is recommended by the R-package skill, but `air` was not
  installed in this environment. Do not burn time retrying it unless it has been
  installed.
- `docs/` is ignored in `.gitignore` but many pkgdown files are tracked. Use
  `git add -f docs` when a site rebuild intentionally changes tracked or new
  site files.
- `pkgdown::build_site()` renders arbitrary top-level Markdown files such as
  `CLAUDE.md` into `docs/CLAUDE.html`. `.Rbuildignore` does **not** stop this.
  Remove `docs/CLAUDE.html` after builds unless the file should be public.
- `pkgdown::build_site()` can fail transiently on this OneDrive-backed path with
  `xml2::write_html(): Error closing file`. A retry succeeded both times this
  happened.
- `gh pr merge` may report a local git checkout failure because `main` is already
  checked out in the separate main worktree. Check `gh pr view <n>` before
  retrying; PR #56 merged remotely even though the local cleanup step failed.
- After a squash merge, a follow-up branch based on pre-squash history will show
  a huge three-dot diff until `origin/main` is merged into it. If shorthand fetch
  misbehaves, restore the tracking ref with:

```powershell
git fetch origin refs/heads/main:refs/remotes/origin/main
```

## Live Hugging Face findings

- Default chat/streaming works with `meta-llama/Llama-3.1-8B-Instruct`, but that
  model rejected tool calls. Use `Qwen/Qwen2.5-72B-Instruct` for tool examples.
- `hf_extract()` works live with the default Llama model through the fallback from
  `json_schema` to `json_object`.
- `google/gemma-3-4b-it` worked for vision descriptions, but captioning can hit
  provider-capacity 429s. Keep README quick starts off that dependency.
- `openai/whisper-large-v3` worked for ASR. Public starter audio used in docs:
  `https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac`.
- `black-forest-labs/FLUX.1-schnell` worked for text-to-image with a small
  `num_inference_steps = 2` example.
- The Hugging Face docs cat image worked for object detection but gave confusing
  image-classification labels. For beginner docs, use the MS COCO cat image:
  `http://images.cocodataset.org/val2017/000000039769.jpg`.
- Public hosted TTS candidates tested in June 2026 were not supported by the
  public `hf-inference` provider. Leave TTS examples non-executed unless a
  compatible provider or dedicated endpoint is supplied.

## LinkedIn/social assets

The local post kit lives in `docs/linkedin/`. It was originally kept ignored, then
packaged so it can be force-added if the user wants it preserved in the repo.

Key files:

- `docs/linkedin/huggingfaceR-linkedin-carousel.pdf` — best LinkedIn upload
- `docs/linkedin/slides/huggingfaceR-carousel-01.png` through `06.png`
- `docs/linkedin/huggingfaceR-linkedin-cover-1200x627.png`
- `docs/linkedin/huggingfaceR-linkedin-square-1080.png`
- `docs/linkedin/post-copy.md`
- `docs/linkedin/carousel.html`
- `docs/linkedin/export-assets.mjs`

The first export looked broken because the responsive preview scaling applied to
single-slide screenshots. The CSS now scopes that scaling to gallery preview mode:
`body:not(.single):not(.asset):not(.print)`.
