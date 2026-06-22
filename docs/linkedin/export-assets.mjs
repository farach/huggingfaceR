import { chromium } from "playwright";
import { fileURLToPath } from "node:url";
import path from "node:path";
import fs from "node:fs/promises";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const htmlPath = path.join(__dirname, "carousel.html");
const slidesDir = path.join(__dirname, "slides");
const htmlUrl = `file:///${htmlPath.replace(/\\/g, "/")}`;

async function ensureDirs() {
  await fs.mkdir(slidesDir, { recursive: true });
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

await ensureDirs();
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

await browser.close();

console.log(`Exported LinkedIn assets to ${__dirname}`);
