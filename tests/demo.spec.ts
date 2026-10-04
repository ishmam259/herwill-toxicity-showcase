import { test, expect } from "@playwright/test";
test("record the showcase walkthrough", async ({ page }, info) => {
  test.skip(
    !process.env.RECORD_DEMO || info.project.name !== "chromium",
    "Run npm run record:demo for the desktop walkthrough.",
  );
  await page.goto("/");
  await page.getByRole("button", { name: "Analyze post" }).click();
  await expect(
    page.getByRole("heading", { name: "Signals in the words" }),
  ).toBeVisible();
  await page.waitForTimeout(1500);
  await page.getByLabel("Example language").selectOption("Bangla");
  await page.locator(".example-button").last().click();
  await page.getByRole("button", { name: "Analyze post" }).click();
  await expect(page.locator(".prediction-caption")).toContainText(
    "Bangla script",
  );
  await page.waitForTimeout(1500);
  for (const name of [
    "Model comparison",
    "Confusion matrices",
    "Calibration & thresholds",
    "Hard cases",
  ]) {
    await page.getByRole("navigation").getByRole("link", { name }).click();
    await page.waitForTimeout(1500);
    if (name === "Calibration & thresholds") {
      await page
        .getByRole("slider", { name: "Subtle log-probability offset" })
        .fill("14");
      await page.waitForTimeout(1000);
    }
  }
});
