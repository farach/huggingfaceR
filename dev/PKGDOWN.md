# Stable and development documentation

The root site is the accepted CRAN release. `/dev/` is the
four-component development version on `main`. `_pkgdown.yml` uses
pkgdown’s supported `development: mode: auto` routing and retains the
existing theme. Newly built sites include a Versions menu, version
notice, and installation instructions. pkgdown marks development pages
`noindex`.

The dedicated `pkgdown.yaml` workflow stages both trees on `gh-pages`,
then uploads the combined branch with `actions/upload-pages-artifact`
and publishes with `actions/deploy-pages` when Pages is configured for
GitHub Actions. Development publishing cleans only `gh-pages:dev/`.
Stable publishing cleans the root but excludes `/dev/`. The workflow
concurrency group uses `queue: max`: one run executes at a time and up
to 100 pending runs wait without replacing one another. Runs beyond that
queue capacity are canceled.

The build job has read-only repository permissions and neither checkout
persists credentials. Pull requests produce build artifacts only. A
separate publish job consumes those artifacts with write permissions and
never runs for pull requests.

## First deployment and migration

The existing Pages source is legacy `main:/docs`. Do not change it until
a successful publishing run has populated `gh-pages`.

1.  After the workflow is accepted on `main`, a main push or manual
    `dev` run builds development documentation, including the dataset
    viewer pages.
2.  If `gh-pages` does not exist, the publish job verifies the committed
    `docs/index.html` version against CRAN and copies the stable seed
    artifact unchanged. The current committed root is labeled 2.4.0. A
    mismatch fails explicitly rather than seeding stale or development
    docs.
3.  The same run stages development documentation at `gh-pages:dev/`.
4.  With explicit authorization, set Settings \> Pages \> Build and
    deployment \> Source to **GitHub Actions**, then rerun **dev**.
    Official Pages actions publish the combined root and development
    directory at the existing URLs. The workflow does not change Pages
    settings.

The current publication scope is **dev only**. Preserve the live 2.4.0
root, including executed article output and figure assets, unchanged
until the next release. Do not dispatch stable as part of this
migration. The copied root does not gain the new version menu or notice.
Those arrive when a maintainer builds and commits the next stable
release site locally.

Until migration, the workflow stages the branch and emits a notice while
the legacy site remains served from `main:/docs`. Legacy branch
publishing from `gh-pages:/(root)` is also a valid alternative:
generated sites with `.nojekyll` can trigger GitHub’s
`pages-build-deployment` workflow when pushed by the deployment action
using `GITHUB_TOKEN`. The chosen design uses explicit Pages artifact
deployment instead. No PAT or deployment key is required. Keep
`main:docs/` intact as a migration fallback.

The `github-pages` environment currently allows only the `main` branch.
Before publishing from release events, add an allowed **tag** rule `v*`.
That permission change needs separate authorization. Manual stable
dispatches on main can publish accepted tags without changing that rule.

## Stable releases: publish committed docs, never rebuild in CI

Maintainers build release documentation locally with the required
credentials, local models, and vignette execution flags. Check the
three-component release version, reference index, rendered article
outputs, and figure assets, then commit the complete executed `docs/`
tree **before tagging the release**. Keep development-only APIs out of
that release source and documentation. Normalize generated text files to
LF on Windows before committing.

A published, non-draft, non-prerelease GitHub release triggers stable
staging. The workflow checks out `refs/tags/vX.Y.Z`, requires that tag
to match DESCRIPTION, and verifies both DESCRIPTION and the committed
documentation navbar version against the current CRAN source index. It
copies the tag’s committed `docs/` verbatim and verifies file names and
hashes. It never renders stable articles in CI. Existing executed output
and images are preserved. Any tag-level `docs/dev/` tree is excluded;
the published sibling dev tree remains untouched.

Publication before CRAN acceptance fails. After CRAN acceptance,
manually run the workflow on **main**, **target: stable**, with the
exact accepted **tag**. Old tags cannot roll the root back to an older
CRAN version. Older tags that lack the workflow can also be staged
through a main dispatch, provided their committed docs pass the same
checks.

Main pushes and manual dev runs build main and reject a three-component
version. Manual dispatches must select main as the workflow branch.
Before Actions deployment, the combined root is checked against CRAN
again. If CRAN advances, stage the new release’s committed stable docs
before publishing dev. No package release, tag, merge, or Pages
configuration change is performed by this workflow.

## Local checks

Run `Rscript -e 'source("scripts/test-pkgdown.R")'` for offline guards,
including nonexistent stable destinations, execution from
release-version checkouts, and verbatim stable asset copying. CI runs
the same guards on Linux.

For a disposable dev build, source `scripts/pkgdown.R`, call
`pkgdown_prepare()` with a source copy and scratch destination, then
`pkgdown::build_site_github_pages(pkg, examples = FALSE, install = TRUE)`
and `pkgdown_check_site(pkg, "dev")`. Never build over the committed
stable `docs/`. For stable staging, call `pkgdown_stage_stable()` with
the release source copy, a nonexistent destination, the exact tag, and
the verified CRAN version.

For offline dev builds, clear `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN`,
and set `HF_RUN_LOCAL_EXAMPLES` and `HF_CASE_STUDY_RUN_*` flags to false
in the build process. Reference `examples = FALSE` does not disable
vignette evaluation. CI has no HF token secret, so getting-started,
hub-download-upload-share, local-models, multimodal-images-audio-speech,
and structured-extraction-and-tools show code without token/local-gated
output on dev. That is accepted for development, not a reason to replace
stable HTML.

References: [pkgdown development
mode](https://pkgdown.r-lib.org/reference/build_site.html#development-mode),
[r-lib/actions pkgdown
template](https://github.com/r-lib/actions/blob/v2/examples/pkgdown.yaml),
[concurrency
queues](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency),
and [Pages publishing
sources](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).
