// Captures only authored/public content. Start the built app first.
import { chromium } from "@playwright/test";
import { mkdir } from "node:fs/promises";
const base = process.env.PREVIEW_URL || "http://127.0.0.1:7860";
await mkdir("output", { recursive: true });
const browser = await chromium.launch();
try {
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1100 },
  });
  await page.goto(base);
  await page.getByText("Classifier online", { exact: true }).waitFor();
  await page.screenshot({
    path: "output/live-demo-desktop.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "Analyze post" }).click();
  await page.getByRole("heading", { name: "Signals in the words" }).waitFor();
  await page.screenshot({
    path: "output/prediction-desktop.png",
    fullPage: true,
  });
  await page.goto(base + "/compare");
  await page.screenshot({
    path: "output/model-comparison.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 393, height: 852 });
  await page.goto(base);
  await page.getByText("Classifier online", { exact: true }).waitFor();
  await page.screenshot({
    path: "output/live-demo-mobile.png",
    fullPage: true,
  });
  console.log("Captured 4 public-only screenshots in output/.");
} finally {
  await browser.close();
}
