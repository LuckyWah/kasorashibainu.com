(function () {
  const form = document.getElementById("simulation-form");
  const status = document.getElementById("simulation-status");
  const error = document.getElementById("simulation-error");
  const results = document.getElementById("simulation-results");
  const outcome = document.getElementById("simulation-outcome");
  const summary = document.getElementById("simulation-summary");
  const predictionChart = document.getElementById("prediction-chart");
  const strategyChart = document.getElementById("strategy-chart");
  const strategyMetric = document.getElementById("strategy-metric");

  if (!form) {
    return;
  }

  const apiBase = (window.LUCKY_STOCK_API_BASE || "").replace(/\/$/, "");
  const totalCashInput = form.elements.totalCash;
  const aggressivenessInput = form.elements.aggressiveness;
  const startDateInput = form.elements.startDate;
  const endDateInput = form.elements.endDate;
  const SIMULATION_TIMEOUT_MS = 15 * 60 * 1000;
  const strategyMetricTitles = [
    "Portfolio Value ($)",
    "Stock Value ($)",
    "Average Cost ($)",
    "Daily Investment ($)",
    "Remaining Cash ($)",
    "Return on Invested Capital",
  ];

  function normalizeMoney(value) {
    return String(value || "").replace(/,/g, "").trim();
  }

  function formatMoneyInput() {
    const rawValue = normalizeMoney(totalCashInput.value);

    if (!rawValue || !/^\d+(?:\.\d{0,2})?$/.test(rawValue)) {
      return;
    }

    const [whole, decimals] = rawValue.split(".");
    const formattedWhole = Number(whole).toLocaleString("en-US");
    totalCashInput.value = decimals !== undefined
      ? `${formattedWhole}.${decimals}`
      : formattedWhole;
  }

  function validateTotalCash() {
    const value = normalizeMoney(totalCashInput.value);
    const isMoney = /^\d+(?:\.\d{1,2})?$/.test(value);
    const amount = Number(value);

    if (!isMoney || !Number.isFinite(amount) || amount < 100) {
      totalCashInput.setCustomValidity("Enter an amount of at least $100 with at most two decimal places.");
      return false;
    }

    totalCashInput.setCustomValidity("");
    return true;
  }

  function validateAggressiveness() {
    const amount = Number(aggressivenessInput.value);

    if (!Number.isFinite(amount) || amount < 0.5 || amount > 3) {
      aggressivenessInput.setCustomValidity("Choose aggressiveness from 0.5 to 3.");
      return false;
    }

    aggressivenessInput.setCustomValidity("");
    return true;
  }

  function parseLocalDate(value) {
    const [year, month, day] = String(value || "").split("-").map(Number);

    if (!year || !month || !day) {
      return null;
    }

    return new Date(year, month - 1, day);
  }

  function todayLocalDate() {
    const now = new Date();
    return new Date(now.getFullYear(), now.getMonth(), now.getDate());
  }

  function validateDates() {
    const startDate = parseLocalDate(startDateInput.value);
    const endDate = parseLocalDate(endDateInput.value);
    const today = todayLocalDate();

    startDateInput.setCustomValidity("");
    endDateInput.setCustomValidity("");

    if (!startDate || !endDate) {
      return true;
    }

    if (startDate > today) {
      startDateInput.setCustomValidity("Start date must be today or a past date.");
      return false;
    }

    if (endDate > today) {
      endDateInput.setCustomValidity("End date must be today or a past date.");
      return false;
    }

    if (endDate <= startDate) {
      endDateInput.setCustomValidity("End date must be later than start date.");
      return false;
    }

    const days = (endDate - startDate) / 86400000;
    if (days < 28) {
      endDateInput.setCustomValidity("Choose a longer period so the simulation includes at least 20 trading days.");
      return false;
    }

    return true;
  }

  function setLoading(isLoading) {
    const button = form.querySelector("button[type='submit']");
    button.disabled = isLoading;
    status.hidden = !isLoading;
  }

  function showError(message) {
    error.textContent = message;
    error.hidden = false;
  }

  function clearResults() {
    error.hidden = true;
    error.textContent = "";
    results.hidden = true;
    outcome.hidden = true;
    outcome.textContent = "";
    summary.hidden = true;
    summary.innerHTML = "";
    predictionChart.innerHTML = "";
    strategyChart.innerHTML = "";
  }

  function updateStrategyMetric() {
    const metricIndex = Number(strategyMetric.value);
    const traceCount = strategyMetricTitles.length * 2;
    const visible = Array.from({ length: traceCount }, (_, index) => {
      return index === metricIndex * 2 || index === metricIndex * 2 + 1;
    });

    Plotly.restyle(strategyChart, { visible, showlegend: visible });
    Plotly.relayout(strategyChart, {
      "yaxis.title.text": strategyMetricTitles[metricIndex],
    });
  }

  function formatDisplayDate(value) {
    const [year, month, day] = String(value || "").split("-");

    if (!year || !month || !day) {
      return value;
    }

    return `${month}/${day}/${year}`;
  }

  function renderSummary(data) {
    const difference = Number(data.toolFinalValue) - Number(data.dcaFinalValue);
    const differenceText = Math.abs(difference).toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
    const comparisonText = difference >= 0
      ? `$${differenceText} more than DCA`
      : `$${differenceText} less than DCA`;

    outcome.textContent = `By using Lucky Stock for daily investment in ${data.ticker} from ${formatDisplayDate(data.startDate)} to ${formatDisplayDate(data.endDate)}, your portfolio would grow to $${Number(data.toolFinalValue).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}, ${comparisonText}.`;
    outcome.hidden = false;

    const rows = [
      ["Ticker", data.ticker],
      ["Simulation Period", `${formatDisplayDate(data.startDate)} to ${formatDisplayDate(data.endDate)}`],
      ["Trading Days", data.tradingDays],
      ["Lucky Stock Final Value", `$${Number(data.toolFinalValue).toLocaleString()}`],
      ["DCA Final Value", `$${Number(data.dcaFinalValue).toLocaleString()}`],
    ];

    summary.innerHTML = rows
      .map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`)
      .join("");
    results.hidden = false;
    summary.hidden = false;
  }

  totalCashInput.addEventListener("input", validateTotalCash);
  aggressivenessInput.addEventListener("input", validateAggressiveness);
  totalCashInput.addEventListener("blur", () => {
    formatMoneyInput();
    validateTotalCash();
  });
  startDateInput.addEventListener("input", validateDates);
  endDateInput.addEventListener("input", validateDates);
  strategyMetric.addEventListener("change", updateStrategyMetric);

  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (!validateTotalCash() || !validateAggressiveness() || !validateDates() || !form.reportValidity()) {
      return;
    }

    clearResults();
    setLoading(true);

    const formData = new FormData(form);
    const payload = {
      ticker: String(formData.get("ticker") || "").trim().toUpperCase(),
      startDate: String(formData.get("startDate") || ""),
      endDate: String(formData.get("endDate") || ""),
      totalCash: Number(normalizeMoney(formData.get("totalCash"))),
      aggressiveness: Number(formData.get("aggressiveness")),
    };

    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => {
      controller.abort();
    }, SIMULATION_TIMEOUT_MS);

    try {
      const response = await fetch(`${apiBase}/api/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      const contentType = response.headers.get("content-type") || "";
      const data = contentType.includes("application/json")
        ? await response.json()
        : null;

      if (!response.ok) {
        throw new Error(data?.detail || "Simulation API request failed.");
      }

      if (!data) {
        throw new Error("Simulation API did not return JSON. Check that /api/simulate is proxied to the backend server.");
      }

      renderSummary(data.summary);
      Plotly.newPlot(predictionChart, data.predictionChart.data, data.predictionChart.layout, {
        responsive: true,
        displaylogo: false,
      });
      Plotly.newPlot(strategyChart, data.strategyChart.data, data.strategyChart.layout, {
        responsive: true,
        displaylogo: false,
      }).then(() => {
        strategyMetric.value = "0";
        updateStrategyMetric();
      });
    } catch (err) {
      if (err.name === "AbortError") {
        showError("Simulation timed out after 15 minutes. Try a shorter period or restart the backend with a longer proxy/server timeout.");
      } else {
        showError(err.message || "Simulation failed.");
      }
    } finally {
      window.clearTimeout(timeoutId);
      setLoading(false);
    }
  });
})();
