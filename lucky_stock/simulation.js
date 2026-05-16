(function () {
  const form = document.getElementById("simulation-form");
  const status = document.getElementById("simulation-status");
  const error = document.getElementById("simulation-error");
  const results = document.getElementById("simulation-results");
  const summary = document.getElementById("simulation-summary");
  const predictionChart = document.getElementById("prediction-chart");
  const strategyChart = document.getElementById("strategy-chart");

  if (!form) {
    return;
  }

  const apiBase = (window.LUCKY_STOCK_API_BASE || "").replace(/\/$/, "");
  const totalCashInput = form.elements.totalCash;

  function validateTotalCash() {
    const value = String(totalCashInput.value || "").trim();
    const isMoney = /^\d+(?:\.\d{1,2})?$/.test(value);
    const amount = Number(value);

    if (!isMoney || !Number.isFinite(amount) || amount <= 100) {
      totalCashInput.setCustomValidity("Enter an amount greater than $100 with at most two decimal places.");
      return false;
    }

    totalCashInput.setCustomValidity("");
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
    summary.hidden = true;
    summary.innerHTML = "";
    predictionChart.innerHTML = "";
    strategyChart.innerHTML = "";
  }

  function renderSummary(data) {
    const rows = [
      ["Ticker", data.ticker],
      ["Simulation Period", `${data.startDate} to ${data.endDate}`],
      ["Trading Days", data.tradingDays],
      ["Total Cash", `$${Number(data.totalCash).toLocaleString()}`],
      ["Lucky Stock Final Value", `$${Number(data.toolFinalValue).toLocaleString()}`],
      ["DCA Final Value", `$${Number(data.dcaFinalValue).toLocaleString()}`],
      ["CPU Workers", data.cpuWorkers],
    ];

    summary.innerHTML = rows
      .map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`)
      .join("");
    results.hidden = false;
    summary.hidden = false;
  }

  totalCashInput.addEventListener("input", validateTotalCash);

  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (!validateTotalCash() || !form.reportValidity()) {
      return;
    }

    clearResults();
    setLoading(true);

    const formData = new FormData(form);
    const payload = {
      ticker: String(formData.get("ticker") || "").trim().toUpperCase(),
      startDate: String(formData.get("startDate") || ""),
      endDate: String(formData.get("endDate") || "") || null,
      totalCash: Number(String(formData.get("totalCash") || "").trim()),
    };

    try {
      const response = await fetch(`${apiBase}/api/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Simulation failed.");
      }

      renderSummary(data.summary);
      Plotly.newPlot(predictionChart, data.predictionChart.data, data.predictionChart.layout, {
        responsive: true,
        displaylogo: false,
      });
      Plotly.newPlot(strategyChart, data.strategyChart.data, data.strategyChart.layout, {
        responsive: true,
        displaylogo: false,
      });
    } catch (err) {
      showError(err.message || "Simulation failed.");
    } finally {
      setLoading(false);
    }
  });
})();
