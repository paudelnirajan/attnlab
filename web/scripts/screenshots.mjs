// Visual smoke check: drives the real app in a real Chromium and writes a
// screenshot per state, plus any console error, page error or failed request.
//
//   npm run shots            -> ./.screenshots
//   npm run shots -- /tmp/x  -> somewhere else
//
// The vitest suite proves behaviour; this is the part it structurally cannot
// cover, because jsdom has no canvas backend and no layout engine. It caught
// two real bugs on first run: the causal mask rendering light in dark mode
// (a theme-attribute ordering hazard), and the linked band painting over
// cells that causal masking makes impossible.
//
// Requires both dev servers: uvicorn on :8000 and vite on :5173. Covers the
// attention lab (01-11) and the tokenizer lab + home page (20-26).

import { chromium } from "playwright";
const OUT = process.argv[2] ?? ".screenshots";
await (await import("node:fs/promises")).mkdir(OUT, { recursive: true });
const problems = [];
let failed = false;

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1380, height: 1000 } });
page.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") problems.push(`[${m.type()}] ${m.text()}`); });
page.on("pageerror", (e) => problems.push(`[pageerror] ${e.message}`));
page.on("requestfailed", (r) => problems.push(`[requestfailed] ${r.url()} ${r.failure()?.errorText}`));

const shot = async (name, opts = {}) => { await page.screenshot({ path: `${OUT}/${name}.png`, ...opts }); console.log("shot:", name); };

await page.goto("http://127.0.0.1:5173/attention", { waitUntil: "networkidle" });
await page.waitForSelector(".chip", { timeout: 20000 });
await page.waitForFunction(() => document.querySelectorAll("canvas").length > 0, null, { timeout: 20000 });
await page.waitForTimeout(600);
await shot("01-overview", { fullPage: true });

// hover a token -> tinting + linked bands across every tile
const chips = page.locator(".chip");
console.log("tokens:", await chips.count(), "| tiles:", await page.locator(".tile").count());
await chips.nth(6).hover();
await page.waitForTimeout(250);
await shot("02-hover-link", { fullPage: true });

// expand a head
await page.locator('[title="Head 5"]').click();
await page.waitForTimeout(400);
await shot("03-head-detail", { fullPage: true });

// crosshair + tooltip on the matrix
const canvas = page.locator(".heat__canvas");
const box = await canvas.boundingBox();
await page.mouse.move(box.x + box.width * 0.25, box.y + box.height * 0.8);
await page.waitForTimeout(250);
await canvas.screenshot({ path: `${OUT}/04-tooltip.png` });
console.log("shot: 04-tooltip");

// flip direction
await page.getByRole("button", { name: "Source → Destination" }).click();
await page.waitForTimeout(300);
await shot("05-direction-src2dest", { fullPage: true });

// modal
await page.getByRole("button", { name: /How to read this/ }).click();
await page.waitForTimeout(400);
await shot("06-modal");
await page.keyboard.press("Escape");
await page.waitForTimeout(300);

// dark mode: the theme button cycles system -> light -> dark
const themeBtn = page.locator(".topbar .btn--icon").last();
await themeBtn.click(); await page.waitForTimeout(150);
await themeBtn.click(); await page.waitForTimeout(600);
await shot("07-dark", { fullPage: true });

// narrow viewport
await page.setViewportSize({ width: 680, height: 1000 });
await page.waitForTimeout(500);
await shot("08-narrow", { fullPage: true });

// phone
await page.setViewportSize({ width: 390, height: 844 });
await page.waitForTimeout(500);
await shot("09-phone", { fullPage: true });
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
console.log("horizontal overflow at 390px:", overflow, "px (expect 0)");

// The two URLs below are deliberately the pre-labs "/?model=…" form: main.tsx
// must move them to /attention, and these shots fail if it doesn't.
//
// A script the tokenizer has no merges for. Byte-level BPE spends 2-3 tokens
// per character here, none of which decodes to a character on its own; this
// checks the strip shows the text rather than a row of U+FFFD.
await page.setViewportSize({ width: 1200, height: 900 });
const devanagari = encodeURIComponent("म हिमालयी राष्ट्र नेपालको हुँ र मलाई यसमा धेरै गर्व छ।");
await page.goto(`http://127.0.0.1:5173/?model=gpt2-small&prompt=${devanagari}&layer=0`, { waitUntil: "networkidle" });
await page.waitForSelector(".cluster--frag", { timeout: 120000 });
await page.waitForTimeout(900);
await page.locator(".card").filter({ hasText: "positions" }).first().screenshot({ path: `${OUT}/10-fragmentation.png` });
console.log("shot: 10-fragmentation");
const clean = await page.evaluate(() => !document.querySelector(".chipstrip")?.textContent?.includes("\uFFFD"));
console.log("no U+FFFD rendered in the token strip:", clean);
if (!clean) { problems.push("token strip rendered a replacement character"); }

// Long prompt with a head open. The axis ticks must not stretch .heat__plot
// taller than the canvas inside it: every overlay (the linked band, the
// crosshair) is positioned as a percentage of that box, so any drift puts them
// on the wrong row. Under ~40 tokens the ticks fit within the canvas height
// and the bug is invisible — which is exactly how it shipped once.
await page.setViewportSize({ width: 1380, height: 1000 });
const longPrompt = encodeURIComponent(Array.from({ length: 60 }, (_, i) => `w${i}`).join(" "));
await page.goto(`http://127.0.0.1:5173/?model=attn-only-2l-demo&prompt=${longPrompt}&layer=0&head=2`, {
  waitUntil: "networkidle",
});
await page.waitForSelector(".heat__band", { timeout: 60000 });
await page.waitForTimeout(800);
const geo = await page.evaluate(() => {
  const r = (sel) => document.querySelector(sel)?.getBoundingClientRect() ?? null;
  const plot = r(".heat__plot"), canvas = r(".heat__canvas"), band = r(".heat__band");
  return {
    plotH: plot.height, canvasH: canvas.height,
    plotTop: plot.top, canvasTop: canvas.top,
    canvasBottom: canvas.bottom, bandBottom: band?.bottom ?? null, bandTop: band?.top ?? null,
  };
});
const drift = Math.abs(geo.plotH - geo.canvasH) + Math.abs(geo.plotTop - geo.canvasTop);
console.log(`plot box ${geo.plotH.toFixed(1)}px vs canvas ${geo.canvasH.toFixed(1)}px — drift ${drift.toFixed(2)}px`);
if (drift > 1) problems.push(`overlay box drifts ${drift.toFixed(1)}px from the canvas; band/crosshair will be misplaced`);
if (geo.bandBottom !== null && (geo.bandBottom > geo.canvasBottom + 1 || geo.bandTop < geo.canvasTop - 1)) {
  problems.push("the linked band is drawn outside the canvas");
}
await shot("11-long-prompt", { fullPage: true });

// --- Tokenizer lab (step 1) and the path overview -------------------------
await page.goto("http://127.0.0.1:5173/", { waitUntil: "networkidle" });
await shot("20-home", { fullPage: true });

const nepali = encodeURIComponent("नेपाल एक सुन्दर देश हो।");
await page.goto(`http://127.0.0.1:5173/tokens?tok=gpt2&text=${nepali}`, { waitUntil: "networkidle" });
await page.waitForSelector(".cluster--frag", { timeout: 60000 });
await page.locator(".chip").nth(1).hover();
await page.waitForTimeout(300);
await shot("21-toklab-inspect", { fullPage: true });
const tokClean = await page.evaluate(() => !document.querySelector(".chipstrip")?.textContent?.includes("\uFFFD"));
if (!tokClean) problems.push("tokenizer lab rendered a replacement character");

for (const [tab, sel, name] of [
  ["Compare", ".compare__row", "22-toklab-compare"],
  ["Languages", ".langtable", "23-toklab-languages"],
  ["BPE step-through", ".symrow", "24-toklab-bpe"],
  ["Vocabulary", ".ranked__row--button", "25-toklab-vocab"],
]) {
  await page.getByRole("tab", { name: tab }).click();
  await page.waitForSelector(sel, { timeout: 60000 });
  await page.waitForTimeout(500);
  await shot(name, { fullPage: true });
}

await page.setViewportSize({ width: 390, height: 900 });
await page.getByRole("tab", { name: "Languages" }).click();
await page.waitForTimeout(500);
const tokOverflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
console.log("tokenizer lab horizontal overflow at 390px:", tokOverflow);
if (tokOverflow > 0) problems.push(`tokenizer lab overflows horizontally by ${tokOverflow}px at 390px`);
await shot("26-toklab-phone", { fullPage: true });

console.log("\nURL:", page.url());
console.log("\n=== console/network problems ===");
if (problems.length) { failed = true; console.log([...new Set(problems)].join("\n")); } else console.log("(none)");
await browser.close();
process.exit(failed ? 1 : 0);
