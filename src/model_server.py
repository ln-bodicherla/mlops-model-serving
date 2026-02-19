"""
FastAPI model serving application.

Provides prediction endpoints with A/B test routing, health checks,
Prometheus metrics, request logging, and background model refresh.
"""

import asyncio
import hashlib
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


PREDICTION_COUNTER = Counter(
    "model_predictions_total",
    "Total prediction requests",
    ["model_version", "status"],
)
PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Prediction latency in seconds",
    ["model_version"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
ERROR_COUNTER = Counter(
    "model_errors_total",
    "Total prediction errors",
    ["error_type"],
)
REQUEST_COUNTER = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)


class PredictRequest(BaseModel):
    """Single prediction request."""
    features: Dict[str, float] = Field(
        ..., description="Feature name-value pairs for prediction"
    )
    request_id: Optional[str] = Field(
        default=None, description="Optional request tracking ID"
    )


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""
    instances: List[Dict[str, float]] = Field(
        ..., min_length=1, max_length=1000,
        description="List of feature dictionaries for batch prediction",
    )
    request_id: Optional[str] = Field(default=None)


class ABPredictRequest(BaseModel):
    """A/B test prediction request."""
    features: Dict[str, float] = Field(
        ..., description="Feature name-value pairs"
    )
    user_id: str = Field(
        ..., description="User ID for consistent routing"
    )
    experiment_id: Optional[str] = Field(
        default=None, description="A/B test experiment ID"
    )


class PredictResponse(BaseModel):
    """Prediction response."""
    prediction: Union[float, int, str] = Field(description="Model prediction")
    probability: Optional[List[float]] = Field(
        default=None, description="Class probabilities if available"
    )
    model_version: str = Field(description="Model version used")
    request_id: Optional[str] = Field(default=None)
    latency_ms: float = Field(description="Processing time in milliseconds")


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[Union[float, int, str]] = Field(description="Predictions")
    probabilities: Optional[List[List[float]]] = Field(default=None)
    model_version: str = Field(description="Model version used")
    count: int = Field(description="Number of predictions")
    request_id: Optional[str] = Field(default=None)
    latency_ms: float = Field(description="Processing time in milliseconds")


class ABPredictResponse(BaseModel):
    """A/B test prediction response."""
    prediction: Union[float, int, str]
    probability: Optional[List[float]] = None
    variant: str = Field(description="A/B test variant (champion/challenger)")
    model_version: str
    experiment_id: Optional[str] = None
    latency_ms: float


class ModelInfoResponse(BaseModel):
    """Model metadata response."""
    model_name: str
    model_version: str
    model_type: str
    feature_names: List[str]
    num_features: int
    loaded_at: str
    status: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
    total_predictions: int


class ModelManager:
    """Manages model loading and versioning."""

    def __init__(self):
        self.champion_model = None
        self.challenger_model = None
        self.champion_version = "none"
        self.challenger_version = "none"
        self.feature_names: List[str] = []
        self.model_type = "unknown"
        self.loaded_at = ""
        self.total_predictions = 0
        self.start_time = time.time()

    def load_model(self, model_path: str, version: str = "v1") -> None:
        """Load a model from disk."""
        if os.path.exists(model_path):
            self.champion_model = joblib.load(model_path)
            self.champion_version = version
            self.model_type = type(self.champion_model).__name__
            self.loaded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            if hasattr(self.champion_model, "feature_names_in_"):
                self.feature_names = list(self.champion_model.feature_names_in_)
            elif hasattr(self.champion_model, "n_features_in_"):
                self.feature_names = [
                    f"feature_{i}" for i in range(self.champion_model.n_features_in_)
                ]

            logger.info(f"Loaded champion model v{version} from {model_path}")
        else:
            logger.warning(f"Model path not found: {model_path}")

    def load_challenger(self, model_path: str, version: str = "v2") -> None:
        """Load a challenger model for A/B testing."""
        if os.path.exists(model_path):
            self.challenger_model = joblib.load(model_path)
            self.challenger_version = version
            logger.info(f"Loaded challenger model v{version} from {model_path}")

    def predict(
        self, features: np.ndarray, variant: str = "champion"
    ) -> Dict[str, Any]:
        """Run prediction with the specified model variant."""
        model = self.champion_model if variant == "champion" else self.challenger_model
        version = self.champion_version if variant == "champion" else self.challenger_version

        if model is None:
            raise ValueError(f"No {variant} model loaded")

        prediction = model.predict(features)
        result = {"prediction": prediction[0], "model_version": version}

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)
            result["probability"] = probabilities[0].tolist()

        self.total_predictions += 1
        return result

    def predict_batch(
        self, features: np.ndarray, variant: str = "champion"
    ) -> Dict[str, Any]:
        """Run batch prediction."""
        model = self.champion_model if variant == "champion" else self.challenger_model
        version = self.champion_version if variant == "champion" else self.challenger_version

        if model is None:
            raise ValueError(f"No {variant} model loaded")

        predictions = model.predict(features)
        result = {
            "predictions": predictions.tolist(),
            "model_version": version,
            "count": len(predictions),
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)
            result["probabilities"] = probabilities.tolist()

        self.total_predictions += len(predictions)
        return result


model_manager = ModelManager()


def _load_config() -> Dict:
    """Load server configuration from YAML file."""
    config_path = os.environ.get("CONFIG_PATH", "configs/config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


async def _model_refresh_task(interval: int = 300):
    """Background task to poll for new model versions."""
    while True:
        await asyncio.sleep(interval)
        try:
            model_path = os.environ.get("MODEL_PATH", "models/champion.joblib")
            if os.path.exists(model_path):
                mtime = os.path.getmtime(model_path)
                if not hasattr(_model_refresh_task, "_last_mtime"):
                    _model_refresh_task._last_mtime = mtime
                elif mtime > _model_refresh_task._last_mtime:
                    model_manager.load_model(model_path)
                    _model_refresh_task._last_mtime = mtime
                    logger.info("Model refreshed from disk")
        except Exception as e:
            logger.error(f"Model refresh failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    config = _load_config()
    server_config = config.get("server", {})

    model_path = os.environ.get("MODEL_PATH", "models/champion.joblib")
    model_manager.load_model(model_path)

    challenger_path = os.environ.get("CHALLENGER_MODEL_PATH", "models/challenger.joblib")
    model_manager.load_challenger(challenger_path)

    refresh_interval = server_config.get("model_refresh_interval", 300)
    refresh_task = asyncio.create_task(_model_refresh_task(refresh_interval))

    logger.info("Model server started")
    yield

    refresh_task.cancel()
    logger.info("Model server shutting down")


app = FastAPI(
    title="ML Model Serving API",
    description="Production model serving with A/B testing and monitoring",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log requests, track latency, and count errors."""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    REQUEST_COUNTER.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code,
    ).inc()

    logger.debug(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} duration={duration:.3f}s"
    )
    return response


def _prepare_features(feature_dict: Dict[str, float]) -> np.ndarray:
    """Convert feature dictionary to numpy array in correct order."""
    if model_manager.feature_names:
        values = [feature_dict.get(name, 0.0) for name in model_manager.feature_names]
    else:
        values = list(feature_dict.values())
    return np.array([values])


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Generate a single prediction from input features."""
    if model_manager.champion_model is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    start_time = time.time()

    try:
        features = _prepare_features(request.features)
        result = model_manager.predict(features, variant="champion")

        latency = (time.time() - start_time) * 1000
        PREDICTION_LATENCY.labels(model_version=result["model_version"]).observe(latency / 1000)
        PREDICTION_COUNTER.labels(
            model_version=result["model_version"], status="success"
        ).inc()

        return PredictResponse(
            prediction=result["prediction"],
            probability=result.get("probability"),
            model_version=result["model_version"],
            request_id=request.request_id,
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        ERROR_COUNTER.labels(error_type=type(e).__name__).inc()
        PREDICTION_COUNTER.labels(
            model_version=model_manager.champion_version, status="error"
        ).inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Generate predictions for a batch of instances."""
    if model_manager.champion_model is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    start_time = time.time()

    try:
        all_features = []
        for instance in request.instances:
            features = _prepare_features(instance)
            all_features.append(features[0])

        feature_array = np.array(all_features)
        result = model_manager.predict_batch(feature_array, variant="champion")

        latency = (time.time() - start_time) * 1000
        PREDICTION_LATENCY.labels(model_version=result["model_version"]).observe(latency / 1000)
        PREDICTION_COUNTER.labels(
            model_version=result["model_version"], status="success"
        ).inc(len(request.instances))

        return BatchPredictResponse(
            predictions=result["predictions"],
            probabilities=result.get("probabilities"),
            model_version=result["model_version"],
            count=result["count"],
            request_id=request.request_id,
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        ERROR_COUNTER.labels(error_type=type(e).__name__).inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/ab", response_model=ABPredictResponse)
async def predict_ab(request: ABPredictRequest) -> ABPredictResponse:
    """A/B test routed prediction using consistent hashing on user ID."""
    if model_manager.champion_model is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    config = _load_config()
    ab_config = config.get("ab_testing", {}).get("default_split", {})
    champion_weight = ab_config.get("champion", 0.9)

    user_hash = int(hashlib.md5(request.user_id.encode()).hexdigest(), 16)
    threshold = int(champion_weight * (2**128))
    variant = "champion" if user_hash < threshold else "challenger"

    if variant == "challenger" and model_manager.challenger_model is None:
        variant = "champion"

    start_time = time.time()

    try:
        features = _prepare_features(request.features)
        result = model_manager.predict(features, variant=variant)
        latency = (time.time() - start_time) * 1000

        PREDICTION_COUNTER.labels(
            model_version=result["model_version"], status="success"
        ).inc()

        return ABPredictResponse(
            prediction=result["prediction"],
            probability=result.get("probability"),
            variant=variant,
            model_version=result["model_version"],
            experiment_id=request.experiment_id,
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        ERROR_COUNTER.labels(error_type=type(e).__name__).inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """Return current model version and metadata."""
    return ModelInfoResponse(
        model_name=os.environ.get("MODEL_NAME", "production_model"),
        model_version=model_manager.champion_version,
        model_type=model_manager.model_type,
        feature_names=model_manager.feature_names,
        num_features=len(model_manager.feature_names),
        loaded_at=model_manager.loaded_at or "not_loaded",
        status="ready" if model_manager.champion_model else "no_model",
    )


@app.get("/model/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check with model validation."""
    uptime = time.time() - model_manager.start_time
    model_loaded = model_manager.champion_model is not None

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_version=model_manager.champion_version,
        uptime_seconds=round(uptime, 1),
        total_predictions=model_manager.total_predictions,
    )


@app.get("/metrics")
async def prometheus_metrics():
    """Expose Prometheus metrics for scraping."""
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
