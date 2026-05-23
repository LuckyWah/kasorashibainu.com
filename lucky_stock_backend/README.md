# Lucky Stock Simulation Backend

FastAPI backend for the Lucky Stock website simulation section.

## Setup

```bash
cd lucky_stock_backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

For one simulation request at a time using all CPU cores:

```bash
LUCKY_STOCK_ALLOWED_ORIGINS="https://kasorashibainu.com,https://www.kasorashibainu.com" \
uvicorn api:app --host 0.0.0.0 --port 8051 --workers 1
```

The simulation uses all available CPU cores through:

- `RandomForestRegressor(n_jobs=-1)`
- `scipy.optimize.differential_evolution(workers=-1)`
- BLAS thread environment variables set to the server CPU count

Use `--workers 1` because each simulation already consumes the machine. Multiple Uvicorn workers can make simultaneous requests fight for CPU and memory.

## Long Simulation Timeouts

The frontend waits up to 15 minutes for `/api/simulate-buy` and `/api/simulate-sell`. Uvicorn does not add a normal request-duration timeout, but a reverse proxy in front of it often does.

For Nginx, set the API proxy timeout to at least 15 minutes:

```nginx
location /api/ {
    proxy_pass http://127.0.0.1:8051;
    proxy_read_timeout 900s;
    proxy_send_timeout 900s;
    proxy_connect_timeout 900s;
}
```

If the site is behind a platform with a fixed request timeout, that platform may still cut off long simulations before the backend finishes.

## Frontend API URL

If the API is behind the same domain as the website, proxy `/api/*` to this server and no frontend change is needed.

If the API is on another domain, add this before `simulation.js` in `lucky_stock/index.html`:

```html
<script>
  window.LUCKY_STOCK_API_BASE = "https://your-backend-domain.com";
</script>
```

## Endpoint

`POST /api/simulate-buy`

```json
{
  "ticker": "IBM",
  "startDate": "2024-01-01",
  "endDate": "",
  "totalCash": 1000
}
```

`POST /api/simulate-sell`

```json
{
  "ticker": "IBM",
  "startDate": "2024-01-01",
  "endDate": "",
  "initialShares": 10
}
```
