const sampleMlir = `func.func @__cost_expr(%dpas_cost: f64) -> f64 {
  %cst = arith.constant 2.200000e+01 : f64
  %cst_0 = arith.constant 3.100000e+01 : f64
  %cst_1 = arith.constant 1.280000e+02 : f64
  %cst_2 = arith.constant 0.000000e+00 : f64
  %cst_3 = arith.constant 2.700000e+01 : f64
  %0 = arith.addf %dpas_cost, %cst_3 : f64
  %1 = arith.mulf %dpas_cost, %cst_0 : f64
  %2 = arith.addf %0, %1 : f64
  %3 = arith.addf %2, %cst_2 : f64
  %4 = arith.mulf %3, %cst_1 : f64
  %5 = arith.addf %4, %cst_3 : f64
  %6 = arith.addf %5, %cst : f64
  %7 = arith.addf %6, %cst_2 : f64
  return %7 : f64
}`;

const mlirInput = document.getElementById("mlir-input");
const parseButton = document.getElementById("parse-button");
const statusEl = document.getElementById("status");
const equationOutput = document.getElementById("equation-output");
const variablesEl = document.getElementById("variables");
const costOutput = document.getElementById("cost-output");

const DEFAULT_VARIABLE_VALUE = 1;
const DEFAULT_SLIDER_MIN = 0;
const DEFAULT_SLIDER_MAX = 100;
const DEFAULT_SLIDER_STEP = 0.1;

let expression = null;
let evaluator = null;
let values = {};

mlirInput.value = sampleMlir;

parseButton.addEventListener("click", parseMlir);

async function parseMlir() {
  setStatus("Parsing...");
  expression = null;
  evaluator = null;
  costOutput.value = "-";

  const response = await fetch("/api/parse", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mlir: mlirInput.value }),
  });

  const payload = await response.json();
  if (!response.ok) {
    setStatus(payload.detail || "Parse failed.", true);
    return;
  }

  expression = payload.javascriptExpression;
  evaluator = compileEquation(payload.variables, payload.javascriptParameters, expression);
  equationOutput.textContent = payload.equationText;
  renderVariables(payload.variables);
  setStatus(`Parsed ${payload.variables.length} variable(s).`);
  updateCost();
}

function renderVariables(variables) {
  variablesEl.replaceChildren();
  values = {};

  for (const name of variables) {
    values[name] = DEFAULT_VARIABLE_VALUE;

    const row = document.createElement("label");
    row.className = "variable-row";

    const nameEl = document.createElement("span");
    nameEl.className = "variable-name";
    nameEl.textContent = name;

    const input = document.createElement("input");
    input.className = "variable-input";
    input.type = "number";
    input.step = "any";
    input.value = String(DEFAULT_VARIABLE_VALUE);

    const slider = document.createElement("input");
    slider.className = "variable-slider";
    slider.type = "range";
    slider.min = String(DEFAULT_SLIDER_MIN);
    slider.max = String(DEFAULT_SLIDER_MAX);
    slider.step = String(DEFAULT_SLIDER_STEP);
    slider.value = String(DEFAULT_VARIABLE_VALUE);

    slider.addEventListener("input", () => {
      const value = Number(slider.value);
      values[name] = value;
      input.value = formatInputValue(value);
      updateCost();
    });

    input.addEventListener("input", () => {
      const value = Number(input.value || 0);
      values[name] = value;
      expandSliderRange(slider, value);
      slider.value = String(value);
      updateCost();
    });

    row.append(nameEl, slider, input);
    variablesEl.append(row);
  }
}

function updateCost() {
  if (!evaluator) {
    return;
  }

  const cost = evaluator(values);
  costOutput.value = Number.isFinite(cost) ? formatCost(cost) : "Invalid";
}

function compileEquation(variables, javascriptParameters, javascriptExpression) {
  const fn = new Function(
    ...javascriptParameters,
    `"use strict"; return (${javascriptExpression});`,
  );
  return (currentValues) => {
    const orderedValues = variables.map((name) => Number(currentValues[name] || 0));
    return fn(...orderedValues);
  };
}

function formatCost(value) {
  return Number(value.toPrecision(12)).toString();
}

function formatInputValue(value) {
  return Number(value.toPrecision(8)).toString();
}

function expandSliderRange(slider, value) {
  if (value < Number(slider.min)) {
    slider.min = String(Math.floor(value));
  }
  if (value > Number(slider.max)) {
    slider.max = String(Math.ceil(value));
  }
}

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}
