from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.routers import experiments, explanations, health, ingestion, matches, predictions
from src.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(
    title="Brasileirão Predictor AI API",
    description="API de predição de resultados da Série A com fusão de dados estatísticos e contexto esportivo estruturado por LLM.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable CORS for local web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(matches.router)
app.include_router(predictions.router)
app.include_router(explanations.router)
app.include_router(experiments.router)
app.include_router(ingestion.router)

@app.get("/")
def root():
    return {
        "service": "Brasileirão Predictor AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/healthz"
    }
