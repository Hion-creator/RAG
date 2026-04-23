const defaults = window.APP_DEFAULTS || {};

const el = {
  chipHost: document.getElementById("chipHost"),
  chipIndex: document.getElementById("chipIndex"),
  questionInput: document.getElementById("questionInput"),
  chatModelInput: document.getElementById("chatModelInput"),
  cloudModelInput: document.getElementById("cloudModelInput"),
  embeddingModelInput: document.getElementById("embeddingModelInput"),
  indexPathInput: document.getElementById("indexPathInput"),
  topKInput: document.getElementById("topKInput"),
  thresholdInput: document.getElementById("thresholdInput"),
  timeoutInput: document.getElementById("timeoutInput"),
  askButton: document.getElementById("askButton"),
  compareButton: document.getElementById("compareButton"),
  statusBar: document.getElementById("statusBar"),
  metricMode: document.getElementById("metricMode"),
  metricRagTime: document.getElementById("metricRagTime"),
  metricCloudTime: document.getElementById("metricCloudTime"),
  metricSources: document.getElementById("metricSources"),
  emptyState: document.getElementById("emptyState"),
  resultGrid: document.getElementById("resultGrid"),
  ragPanel: document.getElementById("ragPanel"),
  cloudPanel: document.getElementById("cloudPanel"),
  ragModelTag: document.getElementById("ragModelTag"),
  cloudModelTag: document.getElementById("cloudModelTag"),
  ragAnswer: document.getElementById("ragAnswer"),
  cloudAnswer: document.getElementById("cloudAnswer"),
  citationList: document.getElementById("citationList"),
};

function formatSeconds(value) {
  const number = Number(value);
  if (Number.isNaN(number)) {
    return "-";
  }
  return `${number.toFixed(2)}s`;
}

function setStatus(message, type = "info") {
  el.statusBar.textContent = message;
  el.statusBar.className = `status ${type}`;
}

function setLoading(isLoading, mode) {
  el.askButton.disabled = isLoading;
  el.compareButton.disabled = isLoading;

  el.askButton.classList.toggle("is-loading", isLoading && mode === "ask");
  el.compareButton.classList.toggle("is-loading", isLoading && mode === "compare");
}

function showResults() {
  el.emptyState.classList.add("hidden");
  el.resultGrid.classList.remove("hidden");
}

function renderCitations(citations) {
  el.citationList.innerHTML = "";

  if (!citations || citations.length === 0) {
    const item = document.createElement("li");
    item.textContent = "No se encontraron fuentes recuperadas.";
    el.citationList.appendChild(item);
    return;
  }

  for (const citation of citations) {
    const item = document.createElement("li");
    item.textContent = citation;
    el.citationList.appendChild(item);
  }
}

function renderRag(payload) {
  const rag = payload.rag;

  showResults();
  el.cloudPanel.classList.add("hidden");

  el.metricMode.textContent = "RAG";
  el.metricRagTime.textContent = formatSeconds(rag.seconds);
  el.metricCloudTime.textContent = "-";
  el.metricSources.textContent = String((rag.citations || []).length);

  el.ragModelTag.textContent = rag.model || "-";
  el.ragAnswer.textContent = rag.answer || "";
  renderCitations(rag.citations || []);

  el.cloudModelTag.textContent = "-";
  el.cloudAnswer.textContent = "";
}

function renderCompare(payload) {
  const rag = payload.rag;
  const cloud = payload.cloud;

  showResults();
  el.cloudPanel.classList.remove("hidden");

  el.metricMode.textContent = "RAG vs Cloud";
  el.metricRagTime.textContent = formatSeconds(rag.seconds);
  el.metricCloudTime.textContent = formatSeconds(cloud.seconds);
  el.metricSources.textContent = String((rag.citations || []).length);

  el.ragModelTag.textContent = rag.model || "-";
  el.cloudModelTag.textContent = cloud.model || "-";

  el.ragAnswer.textContent = rag.answer || "";
  el.cloudAnswer.textContent = cloud.answer || "";

  renderCitations(rag.citations || []);
}

function buildPayload() {
  return {
    question: el.questionInput.value.trim(),
    chat_model: el.chatModelInput.value.trim(),
    cloud_model: el.cloudModelInput.value.trim(),
    embedding_model: el.embeddingModelInput.value.trim(),
    index_path: el.indexPathInput.value.trim(),
    top_k: Number(el.topKInput.value),
    threshold: Number(el.thresholdInput.value),
    timeout: Number(el.timeoutInput.value),
  };
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  let body;
  try {
    body = await response.json();
  } catch (_error) {
    throw new Error("Respuesta invalida del servidor");
  }

  if (!response.ok) {
    throw new Error(body.error || `HTTP ${response.status}`);
  }

  return body;
}

async function runAsk() {
  const payload = buildPayload();
  if (!payload.question) {
    setStatus("Escribe una pregunta antes de consultar.", "error");
    return;
  }

  try {
    setLoading(true, "ask");
    setStatus("Consultando RAG...", "info");
    const result = await postJson("/api/ask", payload);
    renderRag(result);
    setStatus("Respuesta RAG lista.", "success");
  } catch (error) {
    setStatus(error.message || "Error consultando RAG", "error");
  } finally {
    setLoading(false);
  }
}

async function runCompare() {
  const payload = buildPayload();
  if (!payload.question) {
    setStatus("Escribe una pregunta antes de comparar.", "error");
    return;
  }

  try {
    setLoading(true, "compare");
    setStatus("Ejecutando comparativa RAG vs Cloud...", "info");
    const result = await postJson("/api/compare", payload);
    renderCompare(result);
    setStatus("Comparativa lista para demo.", "success");
  } catch (error) {
    setStatus(error.message || "Error ejecutando comparativa", "error");
  } finally {
    setLoading(false);
  }
}

function applyDefaults() {
  el.chatModelInput.value = defaults.chat_model || "qwen3.5:cloud";
  el.cloudModelInput.value = defaults.cloud_model || defaults.chat_model || "qwen3.5:cloud";
  el.embeddingModelInput.value = defaults.embedding_model || "embeddinggemma";
  el.indexPathInput.value = defaults.index_path || "storage/index.json";
  el.topKInput.value = defaults.top_k ?? 4;
  el.thresholdInput.value = defaults.threshold ?? 0.18;
  el.timeoutInput.value = defaults.timeout ?? 240;
  el.questionInput.value = "Cuales son los criterios de escalamiento a humano?";

  el.chipHost.textContent = `Host: ${window.location.origin}`;
  el.chipIndex.textContent = `Indice: ${el.indexPathInput.value}`;
}

function wireEvents() {
  el.askButton.addEventListener("click", runAsk);
  el.compareButton.addEventListener("click", runCompare);
  el.indexPathInput.addEventListener("change", () => {
    el.chipIndex.textContent = `Indice: ${el.indexPathInput.value || "--"}`;
  });

  el.questionInput.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      runCompare();
    }
  });
}

applyDefaults();
wireEvents();
