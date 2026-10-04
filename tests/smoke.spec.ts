import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
async function nav(page: import("@playwright/test").Page, name: string) {
  if (await page.getByRole("button", { name: "Open navigation" }).isVisible())
    await page.getByRole("button", { name: "Open navigation" }).click();
  await page
    .getByRole("navigation")
    .getByRole("link", { name, exact: false })
    .click();
}

test("classifies entered text and shows word signals", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Post to check").fill("You are an idiot. Shut up.");
  await page.getByRole("button", { name: "Check post" }).click();
  await expect(page.locator(".lens-verdict")).toHaveText(
    /Explicitly toxic|Subtly toxic|Not toxic/,
  );
  await expect(
    page.getByRole("heading", { name: "What each model says" }),
  ).toBeVisible();
  await expect(page.locator(".lens-post .lens-w").first()).toBeVisible();
  await page.getByLabel("Post to check").fill("Edited post");
  await expect(page.locator(".lens-verdict")).toHaveCount(0);
});

test("all pages, filters and threshold offsets work", async ({ page }) => {
  await page.goto("/");
  await nav(page, "Model comparison");
  await expect(
    page.getByRole("heading", { name: "Compare the details" }),
  ).toBeVisible();
  await page.getByLabel("Script group").selectOption("L");
  await expect(page.getByRole("row").filter({ hasText: "V1.7" })).toContainText(
    "No coverage",
  );
  await nav(page, "Confusion matrices");
  await page.getByLabel("Model", { exact: true }).selectOption("V1.7");
  await expect(
    page.getByRole("heading", { name: "Matrix as a table" }),
  ).toBeVisible();
  await page.getByLabel("Show row percentages").uncheck();
  await expect(
    page.locator(".matrix-cell").first().locator("small"),
  ).toContainText("of true Explicit");
  await nav(page, "Calibration & thresholds");
  const before = await page.locator(".threshold-result strong").textContent();
  await page
    .getByRole("slider", { name: "Subtle log-probability offset" })
    .fill("20");
  await expect(page.locator(".threshold-result strong")).not.toHaveText(
    before!,
  );
  await page.getByRole("button", { name: "Reset class offsets" }).click();
  await expect(page.locator(".threshold-result strong")).toHaveText(before!);
  await nav(page, "Hard cases");
  await expect(
    page.getByRole("heading", {
      name: "Some research belongs behind a closed door.",
    }),
  ).toBeVisible();
});

test("examples and keyboard shortcut work for Bangla", async ({ page }) => {
  await page.goto("/");
  await page
    .getByRole("group", { name: "Example language" })
    .getByRole("button", { name: "Bangla", exact: true })
    .click();
  await page.locator(".lens-example").last().click();
  const editor = page.getByLabel("Post to check");
  await expect(editor).toHaveValue(/[ঀ-৿]/);
  await expect(page.locator(".lens-verdict-meta")).toContainText("Bangla script");
  await editor.press("Control+Enter");
  await expect(page.locator(".lens-verdict-meta")).toContainText("Bangla script");
});

test("handles API errors without presenting a prediction", async ({ page }) => {
  await page.route("**/api/predict", (route) =>
    route.fulfill({ status: 503, body: "Service unavailable" }),
  );
  await page.goto("/");
  await page.getByLabel("Post to check").fill("Hello there");
  await page.getByRole("button", { name: "Check post" }).click();
  await expect(page.getByRole("alert")).toContainText("503");
  await expect(page.locator(".lens-verdict")).toHaveCount(0);
});

test("editing during a request prevents stale results", async ({ page }) => {
  await page.route("**/api/predict", async (route) => {
    await new Promise((r) => setTimeout(r, 500));
    try {
      await route.continue();
    } catch {}
  });
  await page.goto("/");
  await page.getByLabel("Post to check").fill("Congratulations on the new job!");
  await page.getByRole("button", { name: "Check post" }).click();
  await page.getByLabel("Post to check").fill("A different post");
  await expect(page.getByRole("button", { name: "Check post" })).toBeEnabled();
  await page.waitForTimeout(800);
  await expect(page.locator(".lens-verdict")).toHaveCount(0);
});

for (const path of [
  "/",
  "/compare",
  "/confusion",
  "/calibration",
  "/hard-cases",
])
  test(`accessible and responsive: ${path}`, async ({ page }) => {
    await page.goto(path);
    await expect(page.locator("h1")).toBeVisible();
    if (path === "/hard-cases")
      await expect(page.locator(".private-empty h2")).toContainText(
        "closed door",
      );
    const accessibility = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
      .analyze();
    expect(
      accessibility.violations.map((v) => ({
        rule: v.id,
        nodes: v.nodes.map((n) => ({
          target: n.target,
          reason: n.failureSummary,
        })),
      })),
    ).toEqual([]);
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth > window.innerWidth,
    );
    expect(overflow).toBe(false);
  });
