const metricsEl = document.querySelector("#overview");
const statusEl = document.querySelector("#statusBadge");
const caseListEl = document.querySelector("#caseList");
const documentTableEl = document.querySelector("#documentTable");
const documentDetailEl = document.querySelector("#documentDetail");
const answerBoxEl = document.querySelector("#answerBox");
const contextListEl = document.querySelector("#contextList");
const queryInputEl = document.querySelector("#queryInput");
const runSearchEl = document.querySelector("#runSearch");
const refreshEvalEl = document.querySelector("#refreshEval");
const workflowListEl = document.querySelector("#workflowList");
const domainFilterEl = document.querySelector("#domainFilter");
const caseFilterEl = document.querySelector("#caseFilter");
const reportPreviewEl = document.querySelector("#reportPreview");

let evaluationData = null;
let documentData = [];

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

const workflowSteps = [
  ["Ingest", "Load HIPAA, FDA, and clinical trial documents from the corpus."],
  ["Chunk", "Split documents into sentence-aware chunks with metadata."],
  ["Retrieve", "Combine BM25 keyword ranking with semantic vector similarity."],
  ["Rerank", "Promote the contexts that best match the question."],
  ["Generate", "Produce a supported answer from retrieved evidence."],
  ["Evaluate", "Score faithfulness, relevancy, precision, recall, safety, latency, and cost."],
];

function formatMetric(key, value) {
  if (key.includes("cost")) return `$${Number(value).toFixed(6)}`;
  if (key.includes("latency")) return `${Number(value).toFixed(2)} ms`;
  if (key.includes("tokens")) return Number(value).toLocaleString();
  return Number(value).toFixed(3);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function metricCard(key, value) {
  return `
    <article class="metric-card">
      <span>${metricLabels[key] ?? key}</span>
      <strong>${formatMetric(key, value)}</strong>
    </article>
  `;
}

function renderWorkflow() {
  workflowListEl.innerHTML = workflowSteps
    .map(
      ([title, detail], index) => `
        <article class="workflow-step">
          <span>${String(index + 1).padStart(2, "0")}</span>
          <div>
            <strong>${title}</strong>
            <p>${detail}</p>
          </div>
        </article>
      `,
    )
    .join("");
}

function renderOverview(data) {
  evaluationData = data;
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
  renderCases();
}

function renderCases() {
  const filter = caseFilterEl.value.trim().toLowerCase();
  const cases = (evaluationData?.cases ?? []).filter((item) =>
    `${item.question} ${item.answer}`.toLowerCase().includes(filter),
  );
  caseListEl.innerHTML = cases
    .slice(0, 12)
    .map(
      (item) => `
        <article class="case-card">
          <header>
            <strong>${escapeHtml(item.question)}</strong>
            <span class="pill ${item.guardrails.passed ? "pass" : "fail"}">
              ${item.guardrails.passed ? "safe" : "blocked"}
            </span>
          </header>
          <p>${escapeHtml(item.answer)}</p>
          <footer>
            <span class="pill">faith ${Number(item.faithfulness).toFixed(2)}</span>
            <span class="pill">ctx ${Number(item.context_precision).toFixed(2)}</span>
            <span class="pill">${Number(item.latency_ms).toFixed(2)} ms</span>
          </footer>
        </article>
      `,
    )
    .join("");
}

function renderDomainFilter(documents) {
  const domains = [...new Set(documents.map((document) => document.domain))].sort();
  domainFilterEl.innerHTML = [
    '<option value="">All domains</option>',
    ...domains.map((domain) => `<option value="${domain}">${domain}</option>`),
  ].join("");
}

function renderDocuments() {
  const selectedDomain = domainFilterEl.value;
  const documents = selectedDomain
    ? documentData.filter((document) => document.domain === selectedDomain)
    : documentData;
  documentTableEl.innerHTML = documents
    .map(
      (document) => `
        <button class="document-row" type="button" data-document-id="${document.id}">
          <div>
            <strong>${escapeHtml(document.title)}</strong>
            <p class="eyebrow">${document.domain} · ${document.path}</p>
          </div>
          <span class="pill">${document.chunks} chunk${document.chunks === 1 ? "" : "s"}</span>
        </button>
      `,
    )
    .join("");
}

async function showDocumentDetail(documentId) {
  const response = await fetch(`/api/documents/${encodeURIComponent(documentId)}`);
  const document = await response.json();
  documentDetailEl.innerHTML = `
    <article class="detail-card">
      <div class="panel-heading">
        <div>
          <p class="eyebrow">${document.domain}</p>
          <h3>${escapeHtml(document.title)}</h3>
        </div>
        <span class="pill">${document.chunks.length} chunks</span>
      </div>
      <p>${escapeHtml(document.text)}</p>
    </article>
  `;
}

function renderSearch(data) {
  answerBoxEl.innerHTML = `
    <strong>Answer</strong>
    <p>${escapeHtml(data.answer)}</p>
    <p class="eyebrow">${data.latency_ms.toFixed(2)} ms · ${
      data.guardrails.passed ? "guardrails passed" : "guardrails blocked"
    }</p>
  `;
  contextListEl.innerHTML = data.contexts
    .map(
      (context) => `
        <article class="context-card">
          <header>
            <strong>${escapeHtml(context.title)}</strong>
            <span class="pill">score ${context.score.toFixed(3)}</span>
          </header>
          <p>${escapeHtml(context.text)}</p>
          <footer>
            <span class="pill">bm25 ${context.bm25_score.toFixed(3)}</span>
            <span class="pill">semantic ${context.semantic_score.toFixed(3)}</span>
          </footer>
        </article>
      `,
    )
    .join("");
}

async function loadOverview() {
  const [overviewResponse, documentsResponse, reportResponse] = await Promise.all([
    fetch("/api/overview"),
    fetch("/api/documents"),
    fetch("/api/reports/summary"),
  ]);
  renderOverview(await overviewResponse.json());
  documentData = await documentsResponse.json();
  renderDomainFilter(documentData);
  renderDocuments();
  reportPreviewEl.textContent = await reportResponse.text();
}

async function refreshEvaluation() {
  refreshEvalEl.disabled = true;
  refreshEvalEl.textContent = "Refreshing";
  const response = await fetch("/api/evaluations?qa_limit=20", { method: "POST" });
  renderOverview(await response.json());
  reportPreviewEl.textContent = await (await fetch("/api/reports/summary")).text();
  refreshEvalEl.disabled = false;
  refreshEvalEl.textContent = "Refresh Eval";
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
refreshEvalEl.addEventListener("click", refreshEvaluation);
domainFilterEl.addEventListener("change", renderDocuments);
caseFilterEl.addEventListener("input", renderCases);
queryInputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter") runSearch();
});
documentTableEl.addEventListener("click", (event) => {
  const row = event.target.closest("[data-document-id]");
  if (row) showDocumentDetail(row.dataset.documentId);
});

renderWorkflow();
await loadOverview();
await runSearch();
