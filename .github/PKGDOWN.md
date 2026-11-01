# Stable and development documentation

The root site is the current CRAN release. `/dev/` is the four-component
development version on `main`. `_pkgdown.yml` uses pkgdown's supported
`development: mode: auto` routing and retains the existing theme. Both rebuilt
sites include a Versions menu, an explicit version notice, and the appropriate
installation command. pkgdown marks development pages `noindex`.

The dedicated `pkgdown.yaml` workflow follows the r-lib/actions pkgdown Pages
template and stages to `gh-pages`. The combined branch is uploaded with
`actions/upload-pages-artifact` and published with `actions/deploy-pages` once
Pages is configured for GitHub Actions. Development deployment cleans only
`gh-pages:dev/`. Stable deployment cleans the root but excludes `dev/`.
Publication jobs share a concurrency group and do not cancel one another.
Pull requests build an artifact only, with no deployment.

## First deployment and migration

The existing Pages source is legacy `main:/docs`. Do not change it until a
successful publishing run has populated `gh-pages`.

1. After the workflow is accepted on `main`, a main push or a manual `dev` run
   builds the complete development site, including the dataset viewer pages.
2. If `gh-pages` does not exist, the workflow verifies that the committed
   `docs/index.html` version equals the current CRAN version, then copies
   `docs/` to the new branch without rebuilding the stable root. The current
   committed root is labeled 2.4.0 and can be preserved this way. A CRAN mismatch
   fails explicitly rather than seeding stale or development docs.
3. The same run publishes the new development build to `gh-pages:dev/`.
4. With explicit deployment authorization, set repository Settings > Pages >
   Build and deployment > Source to **GitHub Actions**, then rerun **dev**.
   The workflow deploys the combined stable root and development subdirectory
   with GitHub's official Pages actions. The public URLs do not change.
   No Pages setting change is performed by the workflow.

If the `github-pages` environment restricts deployment branches/tags, its rules
must allow `main` and accepted `v*` release tags. Changing those rules requires
separate authorization. Pull requests never enter the deployment job.

Until that setting changes, the workflow only stages the new branch and emits
a migration notice. The legacy site remains served from `main:/docs`. Simply
changing the legacy source to `gh-pages` is not sufficient: branch commits made
with `GITHUB_TOKEN` do not trigger Pages builds. Official Pages artifact
deployment avoids a new PAT or deployment key and its maintenance burden.

References: [pkgdown development mode](https://pkgdown.r-lib.org/reference/build_site.html#development-mode),
[r-lib/actions pkgdown template](https://github.com/r-lib/actions/blob/v2/examples/pkgdown.yaml),
and [GitHub Pages publishing sources](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

The first copied stable root keeps its existing HTML, styling, and content.
It will not acquire the new version menu/notice until a stable rebuild. It must
not acquire post-release APIs such as the new dataset viewer functions.
The old committed `main:docs/` stays intact as a migration fallback but is no
longer the publishing source after the Pages setting changes.

## Stable releases

A published, non-draft, non-prerelease GitHub release triggers a stable build.
The workflow checks out its tag, requires `vX.Y.Z` to match the package version,
and verifies that version against the current CRAN source package index.
Publishing before CRAN accepts the release fails. Run the workflow again after
CRAN acceptance, using **target: stable** and the exact accepted **tag**.
Old tags cannot roll the root back to an older CRAN version.

For the initial 2.4.0 rebuild, manually run the workflow on **main** with
**target: stable**, **tag: v2.4.0**, after authorizing publication. It uses the
tag's package code, README, reference index, articles, and styling. Only the
development routing and shared version navigation/notice are taken from the
workflow checkout, so functions added on main do not leak into stable docs.
That manual route also works for older release tags that lack the new workflow.

Main pushes and manual **dev** runs always build main, and reject a
three-component version. Manual runs must select main as the workflow branch.
Version or CRAN lookup failures stop publication. No release, tag, merge, or
Pages configuration is created by this workflow.

Before live deployment, the combined site's root navbar version is checked
against CRAN again. If CRAN advances before stable docs are rebuilt, a dev run
cannot publish the stale root. Publish the newly accepted stable tag first,
then rerun dev.

## Local checks

Run `Rscript -e 'source("scripts/test-pkgdown.R")'` for offline publishing guards.
Build into a disposable source copy, never over the committed stable `docs/`:
source `scripts/pkgdown.R`, call `pkgdown_prepare()` with that copy and a scratch
destination, then `pkgdown::build_site_github_pages(pkg, examples = FALSE)` and
`pkgdown_check_site(pkg, target)`. Stable builds require the verified CRAN version
and exact release tag as arguments.

For an offline local build, set `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN` to empty
strings and set `HF_RUN_LOCAL_EXAMPLES` and `HF_CASE_STUDY_RUN_*` flags to false
in the build process. `examples = FALSE` disables reference examples, but
vignettes have their own opt-in execution guards. The workflow explicitly
disables both metered inference and live case-study/local-model execution.
