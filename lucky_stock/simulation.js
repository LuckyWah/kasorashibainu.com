(function () {
  const apiBase = (window.LUCKY_STOCK_API_BASE || "").replace(/\/$/, "");
  const SIMULATION_TIMEOUT_MS = 15 * 60 * 1000;
  const MAX_SIMULATION_DAYS = 150;

  const simulationConfigs = {
    buy: {
      formId: "buy-simulation-form",
      endpoint: "/api/simulate-buy",
      statusId: "buy-simulation-status",
      errorId: "buy-simulation-error",
      resultsId: "buy-simulation-results",
      outcomeId: "buy-simulation-outcome",
      summaryId: "buy-simulation-summary",
      strategyChartId: "buy-strategy-chart",
      metricId: "buy-strategy-metric",
      metricTitles: [
        "Portfolio Value ($)",
        "Stock Value ($)",
        "Average Cost ($)",
        "Daily Investment ($)",
        "Remaining Cash ($)",
        "Return on Invested Capital",
      ],
    },
    sell: {
      formId: "sell-simulation-form",
      endpoint: "/api/simulate-sell",
      statusId: "sell-simulation-status",
      errorId: "sell-simulation-error",
      resultsId: "sell-simulation-results",
      outcomeId: "sell-simulation-outcome",
      summaryId: "sell-simulation-summary",
      strategyChartId: "sell-strategy-chart",
      metricId: "sell-strategy-metric",
      metricTitles: [
        "Sale Total ($)",
        "Unsold Share Value ($)",
        "Average Sold Price ($)",
        "Daily Shares Sold",
        "Remaining Shares",
        "Total Value ($)",
      ],
    },
  };

  function normalizeNumber(value) {
    return String(value || "").replace(/,/g, "").trim();
  }

  function formatMoneyInput(input) {
    const rawValue = normalizeNumber(input.value);

    if (!rawValue || !/^\d+(?:\.\d{0,2})?$/.test(rawValue)) {
      return;
    }

    const [whole, decimals] = rawValue.split(".");
    const formattedWhole = Number(whole).toLocaleString("en-US");
    input.value = decimals !== undefined
      ? `${formattedWhole}.${decimals}`
      : formattedWhole;
  }

  function validateTotalCash(input) {
    const value = normalizeNumber(input.value);
    const isMoney = /^\d+(?:\.\d{1,2})?$/.test(value);
    const amount = Number(value);

    if (!isMoney || !Number.isFinite(amount) || amount < 100) {
      input.setCustomValidity("Enter an amount of at least $100 with at most two decimal places.");
      return false;
    }

    input.setCustomValidity("");
    return true;
  }

  function validateInitialShares(input) {
    const value = normalizeNumber(input.value);
    const isShares = /^\d+(?:\.\d{1,6})?$/.test(value);
    const shares = Number(value);

    if (!isShares || !Number.isFinite(shares) || shares <= 0) {
      input.setCustomValidity("Enter a share amount greater than 0 with at most six decimal places.");
      return false;
    }

    input.setCustomValidity("");
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

  function validateDates(startDateInput, endDateInput) {
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

    if (days > MAX_SIMULATION_DAYS) {
      endDateInput.setCustomValidity("Choose a simulation period of about 100 trading days or less.");
      return false;
    }

    return true;
  }

  function formatSimulationError(message) {
    const text = String(message || "");

    if (/not enough usable training rows/i.test(text)) {
      return "This stock is new and does not have sufficient historical data. Please pick another one.";
    }

    if (/no yahoo finance data found|no close price found/i.test(text)) {
      return "We could not find market data for that ticker. Please check the symbol or pick another stock.";
    }

    if (/not enough usable data|dataset has no usable feature rows/i.test(text)) {
      return "This stock does not have enough clean historical market data for a simulation. Please pick another one.";
    }

    if (/start date is after the available market data/i.test(text)) {
      return "The start date is later than the available market data for this stock. Please choose an earlier start date.";
    }

    if (/no market rows found/i.test(text)) {
      return "No trading data was found for that date range. Please choose a different period.";
    }

    if (/simulation period must include at least 20 trading days/i.test(text)) {
      return "Please choose a longer date range. The simulation needs at least 20 trading days.";
    }

    if (/simulation period must include (?:60|100) trading days or fewer|simulation period must be 1 year or less/i.test(text)) {
      return "Please choose a simulation period of 100 trading days or fewer.";
    }

    if (/initial_shares must be greater than 0/i.test(text)) {
      return "Enter a share amount greater than 0.";
    }

    if (/simulation api request failed/i.test(text)) {
      return "The simulation service could not complete the request. Please try again in a moment.";
    }

    if (/simulation api did not return json|proxied to the backend server/i.test(text)) {
      return "The simulation service is temporarily unavailable. Please try again later.";
    }

    return text;
  }

  function formatDisplayDate(value) {
    const [year, month, day] = String(value || "").split("-");

    if (!year || !month || !day) {
      return value;
    }

    return `${month}/${day}/${year}`;
  }

  function formatCurrency(value) {
    return `$${Number(value).toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })}`;
  }

  function formatNumber(value, maximumFractionDigits = 6) {
    return Number(value).toLocaleString(undefined, {
      maximumFractionDigits,
    });
  }

  function formatPercent(value) {
    return `${(Number(value) * 100).toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })}%`;
  }

  function setLoading(form, status, isLoading) {
    const button = form.querySelector("button[type='submit']");
    button.disabled = isLoading;
    status.hidden = !isLoading;
  }

  function showError(error, message) {
    error.textContent = message;
    error.hidden = false;
  }

  function clearResults(elements) {
    elements.error.hidden = true;
    elements.error.textContent = "";
    elements.results.hidden = true;
    elements.outcome.hidden = true;
    elements.outcome.textContent = "";
    elements.summary.hidden = true;
    elements.summary.innerHTML = "";
    if (elements.predictionChart) {
      elements.predictionChart.innerHTML = "";
    }
    elements.strategyChart.innerHTML = "";
  }

  function updateStrategyMetric(strategyChart, metricSelect, metricTitles) {
    if (!strategyChart.data || strategyChart.data.length === 0) {
      return;
    }

    const metricIndex = Number(metricSelect.value);
    const traceCount = metricTitles.length * 2;
    const visible = Array.from({ length: traceCount }, (_, index) => {
      return index === metricIndex * 2 || index === metricIndex * 2 + 1;
    });

    Plotly.restyle(strategyChart, { visible, showlegend: visible });
    Plotly.relayout(strategyChart, {
      "yaxis.title.text": metricTitles[metricIndex],
    });
  }

  function renderBuySummary(data, elements) {
    const difference = Number(data.toolFinalValue) - Number(data.dcaFinalValue);
    const comparisonText = difference >= 0
      ? `${formatCurrency(Math.abs(difference))} more than DCA`
      : `${formatCurrency(Math.abs(difference))} less than DCA`;

    elements.outcome.textContent = `By using Lucky Stock to invest ${formatCurrency(data.toolInvested)} in ${data.ticker} from ${formatDisplayDate(data.startDate)} to ${formatDisplayDate(data.endDate)}, your portfolio would grow to ${formatCurrency(data.toolFinalValue)}, which is ${comparisonText}.`;
    elements.outcome.hidden = false;

    const rows = [
      ["Ticker", data.ticker],
      ["Simulation Period", `${formatDisplayDate(data.startDate)} to ${formatDisplayDate(data.endDate)}`],
      ["Trading Days", data.tradingDays],
      ["Lucky Stock Final Value", formatCurrency(data.toolFinalValue)],
      ["DCA Final Value", formatCurrency(data.dcaFinalValue)],
    ];

    elements.summary.innerHTML = rows
      .map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`)
      .join("");
    elements.results.hidden = false;
    elements.summary.hidden = false;
  }

  function renderSellSummary(data, elements) {
    const difference = Number(data.toolVsLinearSell);
    const comparisonText = difference >= 0
      ? `${formatCurrency(Math.abs(difference))} more than average sell`
      : `${formatCurrency(Math.abs(difference))} less than average sell`;

    elements.outcome.textContent = `By using Lucky Stock to sell ${formatNumber(data.toolSharesSold)} shares of ${data.ticker} from ${formatDisplayDate(data.startDate)} to ${formatDisplayDate(data.endDate)}, your realized cash would be ${formatCurrency(data.toolFinalValue)}, which is ${comparisonText}.`;
    elements.outcome.hidden = false;

    const rows = [
      ["Ticker", data.ticker],
      ["Simulation Period", `${formatDisplayDate(data.startDate)} to ${formatDisplayDate(data.endDate)}`],
      ["Trading Days", data.tradingDays],
      ["Lucky Stock Realized Cash", formatCurrency(data.toolFinalValue)],
      ["Average Sell Realized Cash", formatCurrency(data.linearSellFinalValue)],
    ];

    elements.summary.innerHTML = rows
      .map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`)
      .join("");
    elements.results.hidden = false;
    elements.summary.hidden = false;
  }

  function buildPayload(kind, formData) {
    const payload = {
      ticker: String(formData.get("ticker") || "").trim().toUpperCase(),
      startDate: String(formData.get("startDate") || ""),
      endDate: String(formData.get("endDate") || ""),
    };

    if (kind === "sell") {
      payload.initialShares = Number(normalizeNumber(formData.get("initialShares")));
    } else {
      payload.totalCash = Number(normalizeNumber(formData.get("totalCash")));
    }

    return payload;
  }

  function attachSimulation(config, kind) {
    const form = document.getElementById(config.formId);

    if (!form) {
      return;
    }

    const elements = {
      status: document.getElementById(config.statusId),
      error: document.getElementById(config.errorId),
      results: document.getElementById(config.resultsId),
      outcome: document.getElementById(config.outcomeId),
      summary: document.getElementById(config.summaryId),
      predictionChart: document.getElementById(config.predictionChartId),
      strategyChart: document.getElementById(config.strategyChartId),
      metricSelect: document.getElementById(config.metricId),
    };
    const startDateInput = form.elements.startDate;
    const endDateInput = form.elements.endDate;
    const totalCashInput = form.elements.totalCash;
    const initialSharesInput = form.elements.initialShares;

    if (totalCashInput) {
      totalCashInput.addEventListener("input", () => validateTotalCash(totalCashInput));
      totalCashInput.addEventListener("blur", () => {
        formatMoneyInput(totalCashInput);
        validateTotalCash(totalCashInput);
      });
    }

    if (initialSharesInput) {
      initialSharesInput.addEventListener("input", () => validateInitialShares(initialSharesInput));
    }

    startDateInput.addEventListener("input", () => validateDates(startDateInput, endDateInput));
    endDateInput.addEventListener("input", () => validateDates(startDateInput, endDateInput));
    elements.metricSelect.addEventListener("change", () => {
      updateStrategyMetric(elements.strategyChart, elements.metricSelect, config.metricTitles);
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();

      const isAmountValid = kind === "sell"
        ? validateInitialShares(initialSharesInput)
        : validateTotalCash(totalCashInput);

      if (!isAmountValid || !validateDates(startDateInput, endDateInput) || !form.reportValidity()) {
        return;
      }

      clearResults(elements);
      setLoading(form, elements.status, true);

      const payload = buildPayload(kind, new FormData(form));
      const controller = new AbortController();
      const timeoutId = window.setTimeout(() => {
        controller.abort();
      }, SIMULATION_TIMEOUT_MS);

      try {
        const response = await fetch(`${apiBase}${config.endpoint}`, {
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
          throw new Error(`Simulation API did not return JSON. Check that ${config.endpoint} is proxied to the backend server.`);
        }

        if (kind === "sell") {
          renderSellSummary(data.summary, elements);
        } else {
          renderBuySummary(data.summary, elements);
        }

        if (elements.predictionChart && data.predictionChart) {
          Plotly.newPlot(elements.predictionChart, data.predictionChart.data, data.predictionChart.layout, {
            responsive: true,
            displaylogo: false,
          });
        }
        Plotly.newPlot(elements.strategyChart, data.strategyChart.data, data.strategyChart.layout, {
          responsive: true,
          displaylogo: false,
        }).then(() => {
          elements.metricSelect.value = "0";
          updateStrategyMetric(elements.strategyChart, elements.metricSelect, config.metricTitles);
        });
      } catch (err) {
        if (err.name === "AbortError") {
          showError(elements.error, "The simulation took too long to finish. Please try a shorter date range or try again later.");
        } else {
          showError(elements.error, formatSimulationError(err.message || "Simulation failed."));
        }
      } finally {
        window.clearTimeout(timeoutId);
        setLoading(form, elements.status, false);
      }
    });
  }

  Object.entries(simulationConfigs).forEach(([kind, config]) => {
    attachSimulation(config, kind);
  });
})();
