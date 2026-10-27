const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const html = fs.readFileSync(path.join(__dirname, "../api/static/index.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1]
  .replace("    warmPredictionService();", "");

function browser(fetch) {
  const elements = new Map();
  function element() {
    return {
      style: {}, hidden: false, disabled: false, children: [], value: "",
      classList: { add() {}, remove() {} },
      addEventListener() {}, removeAttribute() {},
      append(...items) { this.children.push(...items); },
      replaceChildren(...items) { this.children = items; },
    };
  }
  const context = vm.createContext({
    fetch, AbortController, setTimeout, clearTimeout, Blob, console,
    document: {
      getElementById(id) {
        if (!elements.has(id)) elements.set(id, element());
        return elements.get(id);
      },
      createElement: element,
    },
    URL: { createObjectURL: () => "blob:test", revokeObjectURL() {} },
  });
  vm.runInContext(script, context);
  return { context, elements, run: code => vm.runInContext(code, context) };
}

const prediction = {
  predicted_category: "serif", confidence: 0.8, accepted: true,
  probabilities: { display: 0.05, handwriting: 0.05, monospace: 0.05, sans_serif: 0.05, serif: 0.8 },
};

test("an invalid upload is not sent twice", async () => {
  let calls = 0;
  const page = browser(async () => {
    calls++;
    return { ok: false, status: 400, json: async () => ({ detail: "blank image" }) };
  });
  await assert.rejects(page.run('requestPrediction(new Blob(["image"]), new AbortController().signal)'), /blank image/);
  assert.equal(calls, 1);
});

test("a transient failure retries the same captured image", async () => {
  const uploads = [];
  const page = browser(async (url, options) => {
    uploads.push(options.body);
    if (uploads.length === 1) {
      page.run('selectedUpload = new Blob(["different image"])');
      return { ok: false, status: 503, json: async () => ({ detail: "warming up" }) };
    }
    return { ok: true, json: async () => prediction };
  });
  await page.run('requestPrediction(new Blob(["original image"]), new AbortController().signal)');
  assert.equal(uploads.length, 2);
  assert.equal(uploads[0], uploads[1]);
});

test("reset or replacement prevents a stale prediction from changing the result", async () => {
  let resolve;
  const page = browser(() => new Promise(done => { resolve = done; }));
  const pending = page.run('selectedUpload = new Blob(["old image"]); predict()');
  page.run('clearSelection(); setMessage("New image selected", "new.png")');
  resolve({ ok: true, json: async () => prediction });
  await pending;
  assert.equal(page.elements.get("message").children[0].textContent, "New image selected");
  assert.equal(page.elements.get("predict-button").disabled, true);
});

test("malformed model responses never display confidence or probabilities", async () => {
  const page = browser(async () => ({ ok: true, json: async () => ({ ...prediction, confidence: NaN }) }));
  await assert.rejects(page.run('requestPrediction(new Blob(["image"]), new AbortController().signal)'), /invalid prediction/);
});
