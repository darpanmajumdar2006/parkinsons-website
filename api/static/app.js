"use strict";

const state = {
  featureNames: [],
  csvRows: [],
  csvFileName: "",
};

const elements = {
  statusDot: document.querySelector("#status-dot"),
  serviceStatus: document.querySelector("#service-status"),
  uploadTab: document.querySelector("#upload-tab"),
  manualTab: document.querySelector("#manual-tab"),
  uploadPanel: document.querySelector("#upload-panel"),
  manualPanel: document.querySelector("#manual-panel"),
  csvFile: document.querySelector("#csv-file"),
  dropzone: document.querySelector("#dropzone"),
  removeFile: document.querySelector("#remove-file"),
  fileLoaded: document.querySelector("#file-loaded"),
  fileName: document.querySelector("#file-name"),
  fileSummary: document.querySelector("#file-summary"),
  rowPicker: document.querySelector("#row-picker"),
  rowSelect: document.querySelector("#row-select"),
  loadRow: document.querySelector("#load-row"),
  featureGroups: document.querySelector("#feature-groups"),
  featureSearch: document.querySelector("#feature-search"),
  manualProgress: document.querySelector("#manual-progress"),
  clearFields: document.querySelector("#clear-fields"),
  analyzeButton: document.querySelector("#analyze-button"),
  formError: document.querySelector("#form-error"),
  resultEmpty: document.querySelector("#result-empty"),
  resultLoading: document.querySelector("#result-loading"),
  resultContent: document.querySelector("#result-content"),
  resultError: document.querySelector("#result-error"),
  predictionClass: document.querySelector("#prediction-class"),
  probabilityList: document.querySelector("#probability-list"),
  resultTimestamp: document.querySelector("#result-timestamp"),
  anotherButton: document.querySelector("#another-button"),
};

function setServiceStatus(status, message) {
  elements.statusDot.className = `status-dot ${status}`;
  elements.serviceStatus.textContent = message;
}

async function loadFeatureSchema() {
  try {
    const [healthResponse, schemaResponse] = await Promise.all([
      fetch("/health"),
      fetch("/schema"),
    ]);
    if (!healthResponse.ok || !schemaResponse.ok) {
      throw new Error("The prediction service is not ready.");
    }

    const schema = await schemaResponse.json();
    if (!Array.isArray(schema.features) || schema.features.length === 0) {
      throw new Error("The model did not return a feature schema.");
    }

    state.featureNames = schema.features;
    renderFeatureGroups(state.featureNames);
    updateManualProgress();
    elements.analyzeButton.disabled = false;
    setServiceStatus("online", "Model API online");
  } catch (error) {
    setServiceStatus("offline", "Model API unavailable");
    elements.featureGroups.innerHTML =
      '<p class="form-error">Could not load the model feature schema. Refresh this page or try again shortly.</p>';
    showFormError(error.message);
  }
}

function categoryForFeature(name) {
  const feature = name.toLowerCase();
  if (feature === "gender") return "Participant feature";
  if (/jitter|shimmer/.test(feature)) return "Jitter & shimmer";
  if (/period|pulse|^ppe$|^dfa$|^rpde$/.test(feature)) return "Period & rhythm";
  if (/harmonic|intensity|noise/.test(feature)) return "Harmonicity & intensity";
  if (/^(f[1-4]|b[1-4])$/.test(feature)) return "Formants & bandwidth";
  if (/^(gq|gne|vfer|imf)/.test(feature)) return "Voice quality";
  if (/tqwt/.test(feature)) return "TQWT features";
  if (/mfcc/.test(feature)) return "MFCC coefficients";
  if (/delta|log_energy|^mean_|^std_/.test(feature)) return "Signal statistics";
  return "Other acoustic features";
}

function renderFeatureGroups(featureNames) {
  const groups = new Map();
  for (const name of featureNames) {
    const category = categoryForFeature(name);
    if (!groups.has(category)) groups.set(category, []);
    groups.get(category).push(name);
  }

  const fragment = document.createDocumentFragment();
  for (const [category, names] of groups) {
    const details = document.createElement("details");
    details.className = "feature-group";
    details.dataset.category = category.toLowerCase();

    const summary = document.createElement("summary");
    summary.textContent = category;
    const count = document.createElement("span");
    count.textContent = `${names.length} features`;
    summary.append(count);

    const fields = document.createElement("div");
    fields.className = "feature-fields";
    for (const name of names) {
      const field = document.createElement("div");
      field.className = "feature-field";
      field.dataset.feature = name.toLowerCase();
      const label = document.createElement("label");
      const inputId = `feature-${name}`;
      label.htmlFor = inputId;
      label.textContent = name;

      const input = document.createElement("input");
      input.className = "feature-input";
      input.id = inputId;
      input.name = name;
      input.type = "number";
      input.step = "any";
      input.autocomplete = "off";
      input.setAttribute("aria-label", name);
      input.addEventListener("input", updateManualProgress);
      field.append(label, input);
      fields.append(field);
    }

    details.append(summary, fields);
    fragment.append(details);
  }

  elements.featureGroups.replaceChildren(fragment);
}

function updateManualProgress() {
  const filled = [...elements.featureGroups.querySelectorAll(".feature-input")]
    .filter((input) => input.value.trim() !== "").length;
  elements.manualProgress.textContent = `${filled} of ${state.featureNames.length} features filled`;
}

function showFormError(message) {
  if (!message) {
    elements.formError.hidden = true;
    elements.formError.textContent = "";
    return;
  }
  elements.formError.textContent = message;
  elements.formError.hidden = false;
}

function showResultError(message) {
  elements.resultEmpty.hidden = true;
  elements.resultLoading.hidden = true;
  elements.resultContent.hidden = true;
  elements.resultError.textContent = message;
  elements.resultError.hidden = false;
}

function clearResultError() {
  elements.resultError.hidden = true;
  elements.resultError.textContent = "";
}

function setInputMode(mode) {
  const uploadSelected = mode === "upload";
  elements.uploadTab.classList.toggle("selected", uploadSelected);
  elements.uploadTab.setAttribute("aria-selected", String(uploadSelected));
  elements.manualTab.classList.toggle("selected", !uploadSelected);
  elements.manualTab.setAttribute("aria-selected", String(!uploadSelected));
  elements.uploadPanel.hidden = !uploadSelected;
  elements.manualPanel.hidden = uploadSelected;
  showFormError("");
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let value = "";
  let quoted = false;

  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (quoted) {
      if (character === '"' && text[index + 1] === '"') {
        value += '"';
        index += 1;
      } else if (character === '"') {
        quoted = false;
      } else {
        value += character;
      }
    } else if (character === '"' && value.length === 0) {
      quoted = true;
    } else if (character === ",") {
      row.push(value.trim());
      value = "";
    } else if (character === "\n" || character === "\r") {
      if (character === "\r" && text[index + 1] === "\n") index += 1;
      row.push(value.trim());
      if (row.some((cell) => cell !== "")) rows.push(row);
      row = [];
      value = "";
    } else {
      value += character;
    }
  }

  row.push(value.trim());
  if (row.some((cell) => cell !== "")) rows.push(row);
  return rows;
}

function normalizeHeader(value) {
  return value.replace(/^\uFEFF/, "").trim();
}

function extractFeatureRows(csvText) {
  const rows = parseCsv(csvText);
  const requiredNames = new Set(state.featureNames);
  const headerIndex = rows.findIndex((row) => {
    const names = new Set(row.map(normalizeHeader));
    return state.featureNames.every((name) => names.has(name));
  });

  if (headerIndex < 0) {
    throw new Error(
      `Could not find a header containing all ${state.featureNames.length} required model features.`
    );
  }

  const headers = rows[headerIndex].map(normalizeHeader);
  const featureIndexes = new Map();
  headers.forEach((header, index) => {
    if (requiredNames.has(header)) featureIndexes.set(header, index);
  });

  const validRows = [];
  let invalidCount = 0;
  for (const cells of rows.slice(headerIndex + 1)) {
    if (cells.length <= Math.max(...featureIndexes.values())) {
      invalidCount += 1;
      continue;
    }

    const features = {};
    let valid = true;
    for (const name of state.featureNames) {
      const cell = cells[featureIndexes.get(name)];
      const number = Number(cell);
      if (cell === "" || !Number.isFinite(number)) {
        valid = false;
        break;
      }
      features[name] = number;
    }

    if (valid) validRows.push(features);
    else invalidCount += 1;
  }

  if (validRows.length === 0) {
    throw new Error("The CSV header is valid, but no complete numeric feature rows were found.");
  }
  return { rows: validRows, invalidCount };
}

function renderCsvRows(rows) {
  elements.rowSelect.replaceChildren();
  rows.forEach((_, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `Row ${index + 1}`;
    elements.rowSelect.append(option);
  });
  elements.rowPicker.hidden = false;
  elements.fileLoaded.hidden = false;
  elements.fileName.textContent = state.csvFileName;
}

async function readCsvFile(file) {
  showFormError("");
  try {
    const contents = await file.text();
    const parsed = extractFeatureRows(contents);
    state.csvRows = parsed.rows;
    state.csvFileName = file.name;
    renderCsvRows(parsed.rows);
    const skipped = parsed.invalidCount
      ? ` · ${parsed.invalidCount} incomplete row${parsed.invalidCount === 1 ? "" : "s"} skipped`
      : "";
    elements.fileSummary.textContent =
      `${parsed.rows.length} complete feature row${parsed.rows.length === 1 ? "" : "s"}${skipped}`;
    elements.rowSelect.value = "0";
    loadSelectedRow();
  } catch (error) {
    resetUploadedFile();
    showFormError(error.message);
  }
}

function loadSelectedRow() {
  const rowIndex = Number(elements.rowSelect.value);
  const features = state.csvRows[rowIndex];
  if (!features) {
    showFormError("Select a valid row from the uploaded CSV.");
    return;
  }

  const inputs = new Map(
    [...elements.featureGroups.querySelectorAll(".feature-input")].map((input) => [
      input.name,
      input,
    ])
  );
  for (const [name, value] of Object.entries(features)) {
    const input = inputs.get(name);
    if (input) input.value = String(value);
  }
  updateManualProgress();
  setInputMode("upload");
  showFormError("");
}

function resetUploadedFile() {
  state.csvRows = [];
  state.csvFileName = "";
  elements.csvFile.value = "";
  elements.fileLoaded.hidden = true;
  elements.rowPicker.hidden = true;
  elements.fileName.textContent = "";
  elements.fileSummary.textContent = "";
}

function collectFeatures() {
  const features = {};
  const emptyFeatures = [];
  for (const input of elements.featureGroups.querySelectorAll(".feature-input")) {
    const rawValue = input.value.trim();
    if (rawValue === "") {
      emptyFeatures.push(input.name);
      continue;
    }
    const value = Number(rawValue);
    if (!Number.isFinite(value)) {
      throw new Error(`${input.name} must be a finite number.`);
    }
    features[input.name] = value;
  }

  if (emptyFeatures.length > 0) {
    throw new Error(
      `${emptyFeatures.length} feature${emptyFeatures.length === 1 ? " is" : "s are"} missing. ` +
      "Upload a complete feature CSV or fill every required feature."
    );
  }
  return features;
}

async function runPrediction() {
  showFormError("");
  clearResultError();
  let features;
  try {
    features = collectFeatures();
  } catch (error) {
    showFormError(error.message);
    setInputMode("manual");
    return;
  }

  elements.resultEmpty.hidden = true;
  elements.resultContent.hidden = true;
  elements.resultLoading.hidden = false;
  elements.analyzeButton.disabled = true;

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features }),
    });
    const body = await response.json();
    if (!response.ok) {
      const detail = typeof body.detail === "string"
        ? body.detail
        : "The model rejected the feature input. Confirm that every required value is numeric.";
      throw new Error(detail);
    }
    renderPrediction(body);
  } catch (error) {
    showResultError(
      error instanceof TypeError
        ? "Could not reach the prediction service. Check the connection and try again."
        : error.message
    );
  } finally {
    elements.resultLoading.hidden = true;
    elements.analyzeButton.disabled = false;
  }
}

function renderPrediction(result) {
  if (!Number.isInteger(result.prediction) || !result.probabilities) {
    throw new Error("The prediction service returned an unexpected response.");
  }
  elements.predictionClass.textContent = `Class ${result.prediction}`;
  elements.probabilityList.replaceChildren();

  const probabilities = Object.entries(result.probabilities)
    .map(([label, probability]) => [label, Number(probability)])
    .filter(([, probability]) => Number.isFinite(probability));
  if (probabilities.length === 0) {
    throw new Error("The prediction service did not return class probabilities.");
  }

  for (const [label, probability] of probabilities) {
    const percentage = Math.max(0, Math.min(100, probability * 100));
    const row = document.createElement("div");
    row.className = "probability-row";
    const name = document.createElement("span");
    name.className = "probability-label";
    name.textContent = `Class ${label}`;
    const track = document.createElement("div");
    track.className = "probability-track";
    track.setAttribute("role", "progressbar");
    track.setAttribute("aria-label", `Probability for class ${label}`);
    track.setAttribute("aria-valuemin", "0");
    track.setAttribute("aria-valuemax", "100");
    track.setAttribute("aria-valuenow", percentage.toFixed(1));
    const fill = document.createElement("div");
    fill.className = "probability-fill";
    fill.style.width = `${percentage}%`;
    track.append(fill);
    const value = document.createElement("span");
    value.className = "probability-value";
    value.textContent = `${percentage.toFixed(1)}%`;
    row.append(name, track, value);
    elements.probabilityList.append(row);
  }

  elements.resultTimestamp.textContent = new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date());
  elements.resultError.hidden = true;
  elements.resultContent.hidden = false;
}

function clearPrediction() {
  elements.resultContent.hidden = true;
  elements.resultError.hidden = true;
  elements.resultEmpty.hidden = false;
  elements.resultTimestamp.textContent = "";
  window.scrollTo({ top: document.querySelector(".workspace").offsetTop, behavior: "smooth" });
}

elements.uploadTab.addEventListener("click", () => setInputMode("upload"));
elements.manualTab.addEventListener("click", () => setInputMode("manual"));
elements.csvFile.addEventListener("change", (event) => {
  const [file] = event.currentTarget.files;
  if (file) readCsvFile(file);
});
elements.dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  elements.dropzone.classList.add("drag-over");
});
elements.dropzone.addEventListener("dragleave", () => {
  elements.dropzone.classList.remove("drag-over");
});
elements.dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  elements.dropzone.classList.remove("drag-over");
  const [file] = event.dataTransfer.files;
  if (file && file.name.toLowerCase().endsWith(".csv")) readCsvFile(file);
  else showFormError("Choose a CSV file containing named numeric feature columns.");
});
elements.removeFile.addEventListener("click", resetUploadedFile);
elements.loadRow.addEventListener("click", loadSelectedRow);
elements.analyzeButton.addEventListener("click", runPrediction);
elements.anotherButton.addEventListener("click", clearPrediction);
elements.clearFields.addEventListener("click", () => {
  for (const input of elements.featureGroups.querySelectorAll(".feature-input")) {
    input.value = "";
  }
  updateManualProgress();
  showFormError("");
  setInputMode("manual");
});
elements.featureSearch.addEventListener("input", (event) => {
  const query = event.currentTarget.value.trim().toLowerCase();
  for (const group of elements.featureGroups.querySelectorAll(".feature-group")) {
    let visibleFields = 0;
    for (const field of group.querySelectorAll(".feature-field")) {
      const visible = field.dataset.feature.includes(query);
      field.hidden = !visible;
      if (visible) visibleFields += 1;
    }
    group.hidden = visibleFields === 0;
    if (query && visibleFields > 0) group.open = true;
  }
});
document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    setInputMode("manual");
    elements.featureSearch.focus();
  }
});

loadFeatureSchema();
