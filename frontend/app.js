// AgriGuide frontend — vanilla JS, no build step, no framework.

const API_BASE = "/api";

const SAMPLE_QUESTIONS = [
  "How should nitrogen be applied in rice cultivation?",
  "How do I control cutworms without pesticides?",
  "What is the planting distance for mango?",
  "What is cation exchange capacity?",
  "How long should hardwood stem cuttings be?",
  "What are examples of organic manures?",
];

const categoryListEl = document.getElementById("categoryList");
const sampleListEl = document.getElementById("sampleList");
const entriesEl = document.getElementById("entries");
const emptyStateEl = document.getElementById("emptyState");
const formEl = document.getElementById("askForm");
const inputEl = document.getElementById("questionInput");
const askBtn = document.getElementById("askBtn");
const clearBtn = document.getElementById("clearBtn");

let selectedCategory = null; // null = all documents
let entryCount = 0;

// ---------------------------------------------------------------- helpers
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function humanizeCategory(name) {
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

// Turn the model's lightly-formatted text (**bold**, "- " bullets, [1] citations,
// blank-line paragraphs, "Note: ..." asides) into safe, structured HTML.
function formatAnswer(raw) {
  const escaped = escapeHtml(raw);
  const lines = escaped.split("\n");

  const blocks = [];
  let currentList = null;

  function flushList() {
    if (currentList) {
      blocks.push(`<ul>${currentList.join("")}</ul>`);
      currentList = null;
    }
  }

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) {
      flushList();
      continue;
    }
    // Accept "-", "*", and the literal bullet character "•" as markers - the model
    // uses all three depending on nesting/emphasis, and none should ever leak into
    // the rendered text as a raw character.
    const bulletMatch = line.match(/^[-*\u2022]\s+(.*)$/);
    if (bulletMatch) {
      if (!currentList) currentList = [];
      currentList.push(`<li>${inlineFormat(bulletMatch[1])}</li>`);
    } else {
      flushList();
      const cls = /^Note:/i.test(line) ? "note" : "";
      const tag = cls ? "div" : "p";
      blocks.push(`<${tag}${cls ? ' class="note"' : ""}>${inlineFormat(line)}</${tag}>`);
    }
  }
  flushList();
  return blocks.join("");
}

function inlineFormat(text) {
  let out = text.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/\*(.+?)\*/g, "<em>$1</em>"); // single-asterisk asides, e.g. *note*
  out = out.replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (_, nums) =>
    nums
      .split(",")
      .map((n) => `<span class="citation">[${n.trim()}]</span>`)
      .join("")
  );
  return out;
}

function renderSources(sources) {
  if (!sources || sources.length === 0) return "";
  const cards = sources
    .map(
      (s) =>
        `<span class="source-card"><b>[${s.ref}]</b> ${escapeHtml(s.title)} — p.${s.page} (${escapeHtml(s.category)})</span>`
    )
    .join("");
  return `<div class="sources">${cards}</div>`;
}

function autoScroll() {
  window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
}

function updateEmptyState() {
  emptyStateEl.style.display = entriesEl.querySelectorAll(".entry").length ? "none" : "";
}

// ---------------------------------------------------------------- entries
function addPendingEntry(question) {
  entryCount += 1;
  const el = document.createElement("div");
  el.className = "entry pending";
  el.innerHTML = `
    <p class="entry-number">No. ${String(entryCount).padStart(2, "0")}</p>
    <p class="entry-question"></p>
    <div class="entry-answer"><span class="pending-dots">Consulting the documents</span></div>
  `;
  el.querySelector(".entry-question").textContent = question;
  entriesEl.appendChild(el);
  updateEmptyState();
  autoScroll();
  return el;
}

function fillEntrySuccess(el, data) {
  el.classList.remove("pending");
  el.querySelector(".entry-answer").innerHTML = formatAnswer(data.answer) + renderSources(data.sources);
}

function fillEntryError(el, message) {
  el.classList.remove("pending");
  el.classList.add("error");
  el.querySelector(".entry-answer").innerHTML =
    `<p>Something went wrong: ${escapeHtml(message)}</p><p>Try rephrasing the question, or try again.</p>`;
}

// ---------------------------------------------------------------- rail
function renderCategories(categories) {
  const options = [{ value: null, label: "All documents" }].concat(
    categories.map((c) => ({ value: c, label: humanizeCategory(c) }))
  );
  categoryListEl.innerHTML = "";
  options.forEach((opt) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "topic-item";
    btn.textContent = opt.label;
    btn.setAttribute("role", "radio");
    btn.setAttribute("aria-pressed", String(selectedCategory === opt.value));
    btn.addEventListener("click", () => {
      selectedCategory = opt.value;
      [...categoryListEl.children].forEach((b) => b.setAttribute("aria-pressed", "false"));
      btn.setAttribute("aria-pressed", "true");
    });
    categoryListEl.appendChild(btn);
  });
}

function renderSamples() {
  sampleListEl.innerHTML = "";
  SAMPLE_QUESTIONS.forEach((q) => {
    const li = document.createElement("li");
    li.className = "sample-item";
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = q;
    btn.addEventListener("click", () => {
      inputEl.value = q;
      inputEl.dispatchEvent(new Event("input"));
      formEl.requestSubmit();
    });
    li.appendChild(btn);
    sampleListEl.appendChild(li);
  });
}

// ---------------------------------------------------------------- events
formEl.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = inputEl.value.trim();
  if (!question) return;

  askBtn.disabled = true;
  inputEl.disabled = true;
  const entryEl = addPendingEntry(question);

  try {
    const res = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, category: selectedCategory }),
    });
    const data = await res.json().catch(() => null);
    if (!res.ok) {
      throw new Error((data && data.detail) || `Server returned ${res.status}`);
    }
    fillEntrySuccess(entryEl, data);
  } catch (err) {
    fillEntryError(entryEl, err.message || "the request failed");
  } finally {
    askBtn.disabled = false;
    inputEl.disabled = false;
    inputEl.value = "";
    inputEl.style.height = "auto";
    inputEl.focus();
    autoScroll();
  }
});

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    formEl.requestSubmit();
  }
});

inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = `${Math.min(inputEl.scrollHeight, 140)}px`;
});

clearBtn.addEventListener("click", () => {
  entriesEl.querySelectorAll(".entry").forEach((el) => el.remove());
  entryCount = 0;
  updateEmptyState();
});

// ---------------------------------------------------------------- init
async function init() {
  renderSamples();
  try {
    const res = await fetch(`${API_BASE}/categories`);
    const data = await res.json();
    renderCategories(data.categories || []);
  } catch {
    renderCategories([]); // backend still starting up; "All documents" still works
  }
}
init();