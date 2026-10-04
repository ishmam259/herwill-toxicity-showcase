import { test, expect } from "@playwright/test";
test("record the showcase walkthrough", async ({ page }, info) => {
  test.skip(
    !process.env.RECORD_DEMO || info.project.name !== "chromium",
    "Run npm run record:demo for the desktop walkthrough.",
  );
  await page.goto("/");
  await page.locator(".lens-example").first().click();
  await expect(
    page.getByRole("heading", { name: "What each model says" }),
  ).toBeVisible();
  await page.waitForTimeout(1500);
  await page
    .getByRole("group", { name: "Example language" })
    .getByRole("button", { name: "English", exact: true })
    .click();
  await page.locator(".lens-example").nth(2).click();
  await expect(page.locator(".lens-verdict")).toBeVisible();
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
