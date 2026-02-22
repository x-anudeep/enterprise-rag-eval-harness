const metricsEl = document.querySelector("#overview");
const statusEl = document.querySelector("#statusBadge");
const caseListEl = document.querySelector("#caseList");
const documentTableEl = document.querySelector("#documentTable");
const answerBoxEl = document.querySelector("#answerBox");
const contextListEl = document.querySelector("#contextList");
const queryInputEl = document.querySelector("#queryInput");
const runSearchEl = document.querySelector("#runSearch");

const metricLabels = {
  faithfulness: "Faithfulness",
  answer_relevancy: "Answer relevancy",
  context_precision: "Context precision",
  recall_at_3: "Recall@3",
  safety_pass_rate: "Safety pass rate",
  latency_ms_p50: "Median latency",
  estimated_cost_usd: "Estimated cost",
  input_tokens: "Input tokens",
  output_tokens: "Output tokens",
};

function formatMetric(key, value) {
  if (key.includes("cost")) return `$${Number(value).toFixed(6)}`;
  if (key.includes("latency")) return `${Number(value).toFixed(2)} ms`;
  if (key.includes("tokens")) return Number(value).toLocaleString();
  return Number(value).toFixed(3);
}

function metricCard(key, value) {
  return `
    <article class="metric-card">
      <span>${metricLabels[key] ?? key}</span>
      <strong>${formatMetric(key, value)}</strong>
    </article>
  `;
}

function renderOverview(data) {
  statusEl.textContent = data.passed ? "Passing" : "Failing";
  statusEl.className = `status-badge ${data.passed ? "pass" : "fail"}`;
  const preferred = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "recall_at_3",
    "safety_pass_rate",
    "latency_ms_p50",
    "estimated_cost_usd",
    "input_tokens",
  ];
  metricsEl.innerHTML = preferred.map((key) => metricCard(key, data.metrics[key])).join("");
  caseListEl.innerHTML = data.cases
    .slice(0, 8)
    .map(
      (item) => `
        <article class="case-card">
          <header>
            <strong>${item.question}</strong>
            <span class="pill ${item.guardrails.passed ? "pass" : "fail"}">
              ${item.guardrails.passed ? "safe" : "blocked"}
            </span>
          </header>
          <p>${item.answer}</p>
        </article>
      `,
    )
    .join("");
}

function renderDocuments(documents) {
  documentTableEl.innerHTML = documents
    .map(
      (document) => `
        <article class="document-row">
          <div>
            <strong>${document.title}</strong>
            <p class="eyebrow">${document.domain} · ${document.path}</p>
          </div>
          <span class="pill">${document.chunks} chunk${document.chunks === 1 ? "" : "s"}</span>
        </article>
      `,
    )
    .join("");
}

function renderSearch(data) {
  answerBoxEl.innerHTML = `
    <strong>Answer</strong>
    <p>${data.answer}</p>
    <p class="eyebrow">${data.latency_ms.toFixed(2)} ms · ${
      data.guardrails.passed ? "guardrails passed" : "guardrails blocked"
    }</p>
  `;
  contextListEl.innerHTML = data.contexts
    .map(
      (context) => `
        <article class="context-card">
          <header>
            <strong>${context.title}</strong>
            <span class="pill">score ${context.score.toFixed(3)}</span>
          </header>
          <p>${context.text}</p>
        </article>
      `,
    )
    .join("");
}

async function loadOverview() {
  const [overviewResponse, documentsResponse] = await Promise.all([
    fetch("/api/overview"),
    fetch("/api/documents"),
  ]);
  renderOverview(await overviewResponse.json());
  renderDocuments(await documentsResponse.json());
}

async function runSearch() {
  runSearchEl.disabled = true;
  runSearchEl.textContent = "Running";
  const response = await fetch(`/api/search?q=${encodeURIComponent(queryInputEl.value)}`);
  renderSearch(await response.json());
  runSearchEl.disabled = false;
  runSearchEl.textContent = "Run Query";
}

runSearchEl.addEventListener("click", runSearch);
queryInputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter") runSearch();
});

await loadOverview();
await runSearch();
