import os

cpu_count = os.cpu_count() or 1
os.environ.setdefault("OMP_NUM_THREADS", str(cpu_count))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cpu_count))
os.environ.setdefault("MKL_NUM_THREADS", str(cpu_count))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(cpu_count))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from simulation_engine import run_simulation


DATA_DIR = os.getenv("LUCKY_STOCK_DATA_DIR", "datasets")
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("LUCKY_STOCK_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

app = FastAPI(title="Lucky Stock Simulation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class SimulationRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=12)
    startDate: str
    endDate: str = Field(..., min_length=1)
    totalCash: float = Field(..., ge=100, le=100000000)
    aggressiveness: float = Field(1.0, ge=0.0, le=3.0)


@app.get("/api/health")
def health():
    return {"status": "ok", "cpus": cpu_count}


@app.post("/api/simulate")
def simulate(payload: SimulationRequest):
    try:
        return run_simulation(
            ticker=payload.ticker,
            start_date=payload.startDate,
            end_date=payload.endDate,
            total_cash=payload.totalCash,
            data_dir=DATA_DIR,
            aggressiveness=payload.aggressiveness,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
