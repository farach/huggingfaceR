import { chromium } from "playwright";
import { fileURLToPath, pathToFileURL } from "node:url";
import path from "node:path";
import fs from "node:fs/promises";
import JSZip from "jszip";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const htmlPath = path.join(__dirname, "carousel.html");
const slidesDir = path.join(__dirname, "slides");
const stickerSourcePath = path.resolve(__dirname, "..", "..", "man", "figures", "logo.svg");
const stickerPath = path.join(__dirname, "huggingfaceR-hex-sticker.svg");
const htmlUrl = pathToFileURL(htmlPath).href;

async function ensureDirs() {
  await fs.mkdir(slidesDir, { recursive: true });
}

async function syncSticker() {
  try {
    const sticker = await fs.readFile(stickerSourcePath);
    await fs.writeFile(stickerPath, sticker);
    return;
  } catch (error) {
    if (error.code !== "ENOENT") {
      throw error;
    }
  }

  try {
    await fs.access(stickerPath);
    console.warn(`Using existing ${path.basename(stickerPath)}; source sticker was not found at ${stickerSourcePath}`);
  } catch {
    throw new Error(`Could not find package hex sticker at ${stickerSourcePath} or ${stickerPath}`);
  }
}

async function launchBrowser() {
  const launchOptions = [
    { channel: "msedge" },
    { channel: "chrome" },
    {}
  ];

  for (const options of launchOptions) {
    try {
      return await chromium.launch(options);
    } catch {
      // Try the next available browser channel.
    }
  }

  throw new Error("Could not launch Chromium, Chrome, or Edge with Playwright.");
}

async function screenshot(page, url, output, viewport) {
  await page.setViewportSize(viewport);
  await page.goto(url, { waitUntil: "networkidle" });
  await page.screenshot({ path: output, fullPage: false });
}

async function exportContactSheet(page) {
  const contactSheetPath = path.join(__dirname, "huggingfaceR-linkedin-contact-sheet.png");
  const contactSheetHtmlPath = path.join(__dirname, ".contact-sheet.html");
  const slideImages = Array.from({ length: 6 }, (_, index) => {
    const num = String(index + 1).padStart(2, "0");
    const filePath = path.join(slidesDir, `huggingfaceR-carousel-${num}.png`);
    return {
      num,
      url: pathToFileURL(filePath).href
    };
  });

  const contactSheetHtml = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>huggingfaceR LinkedIn carousel contact sheet</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 42px;
      background: linear-gradient(135deg, #fff8dc 0%, #fffdf2 48%, #eef7ff 100%);
      color: #111827;
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    }
    h1 {
      margin: 0 0 26px;
      font-size: 36px;
      letter-spacing: -0.04em;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 24px;
    }
    figure {
      margin: 0;
      overflow: hidden;
      border-radius: 22px;
      background: rgba(255, 255, 255, 0.76);
      border: 1px solid rgba(17, 24, 39, 0.12);
      box-shadow: 0 18px 46px rgba(17, 24, 39, 0.14);
    }
    img {
      display: block;
      width: 100%;
      height: auto;
    }
    figcaption {
      padding: 12px 16px 14px;
      font-size: 16px;
      font-weight: 700;
      color: #384151;
    }
  </style>
</head>
<body>
  <h1>huggingfaceR LinkedIn carousel review sheet</h1>
  <div class="grid">
    ${slideImages.map((slide) => `<figure><img src="${slide.url}" alt="Carousel slide ${slide.num}"><figcaption>Slide ${slide.num}</figcaption></figure>`).join("\n    ")}
  </div>
</body>
</html>`;

  await fs.writeFile(contactSheetHtmlPath, contactSheetHtml);

  try {
    await page.setViewportSize({ width: 1800, height: 1500 });
    await page.goto(pathToFileURL(contactSheetHtmlPath).href, { waitUntil: "networkidle" });
    await page.screenshot({ path: contactSheetPath, fullPage: true });
  } finally {
    await fs.rm(contactSheetHtmlPath, { force: true });
  }
}

async function writeZipBundle() {
  const zip = new JSZip();
  const zipPath = path.join(__dirname, "huggingfaceR-linkedin-post-kit.zip");
  const filePaths = [
    "README.md",
    "post-copy.md",
    "carousel.html",
    "export-assets.mjs",
    "package.json",
    "package-lock.json",
    "huggingfaceR-hex-sticker.svg",
    "huggingfaceR-linkedin-carousel.pdf",
    "huggingfaceR-linkedin-contact-sheet.png",
    "huggingfaceR-linkedin-cover-1200x627.png",
    "huggingfaceR-linkedin-square-1080.png"
  ];

  const slideFiles = (await fs.readdir(slidesDir))
    .filter((file) => /^huggingfaceR-carousel-\d{2}\.png$/.test(file))
    .sort()
    .map((file) => path.join("slides", file));

  for (const relativePath of [...filePaths, ...slideFiles]) {
    const absolutePath = path.join(__dirname, relativePath);
    zip.file(relativePath.replace(/\\/g, "/"), await fs.readFile(absolutePath));
  }

  const buffer = await zip.generateAsync({
    type: "nodebuffer",
    compression: "DEFLATE",
    compressionOptions: { level: 9 }
  });

  await fs.writeFile(zipPath, buffer);
}

await ensureDirs();
await syncSticker();
const browser = await launchBrowser();
const page = await browser.newPage({ deviceScaleFactor: 1 });

for (let i = 1; i <= 6; i++) {
  const num = String(i).padStart(2, "0");
  await screenshot(
    page,
    `${htmlUrl}?slide=${i}`,
    path.join(slidesDir, `huggingfaceR-carousel-${num}.png`),
    { width: 1080, height: 1350 }
  );
}

await screenshot(
  page,
  `${htmlUrl}?asset=cover`,
  path.join(__dirname, "huggingfaceR-linkedin-cover-1200x627.png"),
  { width: 1200, height: 627 }
);

await screenshot(
  page,
  `${htmlUrl}?asset=square`,
  path.join(__dirname, "huggingfaceR-linkedin-square-1080.png"),
  { width: 1080, height: 1080 }
);

await page.setViewportSize({ width: 1080, height: 1350 });
await page.goto(`${htmlUrl}?print=1`, { waitUntil: "networkidle" });
await page.pdf({
  path: path.join(__dirname, "huggingfaceR-linkedin-carousel.pdf"),
  printBackground: true,
  width: "1080px",
  height: "1350px",
  margin: { top: "0", right: "0", bottom: "0", left: "0" }
});

await exportContactSheet(page);
await browser.close();
await writeZipBundle();

console.log(`Exported LinkedIn assets to ${__dirname}`);
