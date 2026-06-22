# huggingfaceR LinkedIn post kit

Assets for the June 2026 `huggingfaceR` feature-expansion post.

## What to upload

Use `huggingfaceR-linkedin-carousel.pdf` as the primary LinkedIn document upload.
It contains six portrait slides at 1080 x 1350.

Backup options:

- `slides/huggingfaceR-carousel-01.png` through `06.png` for image upload
- `huggingfaceR-linkedin-cover-1200x627.png` for a wide single-image post
- `huggingfaceR-linkedin-square-1080.png` for a square single-image post
- `huggingfaceR-linkedin-post-kit.zip` for sharing the whole bundle

Copy, first-comment links, alt text, and posting checklist live in
`post-copy.md`.

## Source files

- `carousel.html` is the editable source for the carousel and single-image assets.
- `export-assets.mjs` exports the PNGs and PDF with Playwright.
- `package.json` and `package-lock.json` pin the local export dependency.

To regenerate:

```powershell
Set-Location docs\linkedin
npm install
npm run export
```

Delete `node_modules/` after export. It is not part of the kit.

## What went wrong

- The first Playwright export produced tiny slides in the top-left of each PNG.
  Cause: the mobile/gallery preview media query also applied to `?slide=1`
  single-slide exports. Fix: scope preview scaling to pages that are not
  `.single`, `.asset`, or `.print`.
- Browser-canvas screenshots timed out on the full gallery because it is a large
  page. The faster review path is the generated contact sheet:
  `huggingfaceR-linkedin-contact-sheet.png`.
- `docs/` is ignored by git. If this kit should be preserved in the repo, use
  `git add -f docs/linkedin`.
- If this folder is committed, it will be served by GitHub Pages under
  `/huggingfaceR/linkedin/`. Do not put private notes or secrets here.

## Design notes

- Primary format: portrait carousel, 1080 x 1350.
- Visual style: warm Hugging Face yellow, editorial cards, large type, visible
  hierarchy for mobile feeds.
- Target audience: R users working with text, documents, interviews, images,
  survey responses, tickets, reviews, or Hub files.
- Main reach tactic: upload the PDF as a carousel, not a link-only post.
