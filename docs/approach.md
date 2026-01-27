# Agile Development Plan: stock-agent-ops

This document outlines an iterative, agile development plan for building the `stock-agent-ops` project from scratch. The plan is divided into seven "Runs," each delivering a functional increment of the final system, building upon the previous one.

---

### Run 1: The Foundation - Data Ingestion & Storage

**Goal:** Establish the core data pipeline. Create a reliable, automated process for fetching raw stock data, computing features, and storing them in a structured feature store.

**Key Tasks:**
1.  **Project Scaffolding:**
    - Initialize the project with `pyproject.toml` for dependency management (using `uv` or `poetry`).
    - Create the initial directory structure (`src`, `feature_store`, `data`).
2.  **Data Ingestion (`src/data/ingestion.py`):**
    - Implement a function to fetch OHLCV data from `yfinance`.
    - Add basic feature engineering (e.g., RSI, MACD).
    - Handle errors and data validation gracefully.
3.  **Feature Store Setup (`feature_store/`):**
    - Define the Feast feature store configuration in `feature_store.yaml`. Start with a local provider (File offline store, Redis online store).
    - Define the feature schema (`Entity`, `FeatureView`) in `features.py`.
4.  **Pipeline Integration:**
    - Modify the ingestion script to save the computed features to a Parquet file in `/data`.
    - Add logic to run `feast apply` and `feast materialize-incremental` to populate the online Redis store.
5.  **Containerization:**
    - Create a `docker-compose.yml` file with a Redis service.

**Deliverable:** A runnable script that populates a local Feast feature store (offline and online) with processed stock data.

---

### Run 2: Model Training MVP

**Goal:** Develop and train a baseline predictive model using the data from the feature store.

**Key Tasks:**
1.  **Data Preparation (`src/data/preparation.py`):**
    - Create a script to load feature data and transform it into sequences suitable for training a time-series model (e.g., sliding windows of `(context_length, prediction_length)`).
2.  **Model Definition (`src/model/definition.py`):**
    - Define a baseline model architecture, such as a simple LSTM or GRU, using a framework like PyTorch or TensorFlow.
3.  **Training & Evaluation Logic (`src/model/`):**
    - Implement the training loop in `training.py`.
    - Implement a basic evaluation script in `evaluation.py` (e.g., calculating MSE, MAE).
    - Implement model serialization (saving/loading) in `saving.py`.
4.  **Training Pipeline (`src/pipelines/training_pipeline.py`):**
    - Create an orchestrator script that runs the full data preparation, model training, evaluation, and saving process.

**Deliverable:** A training pipeline that takes feature data and produces a trained and serialized model file.

---

### Run 3: Backend API for Inference

**Goal:** Serve predictions from the trained model via a REST API.

**Key Tasks:**
1.  **Inference Pipeline (`src/pipelines/inference_pipeline.py`):**
    - Create a pipeline that:
        - Loads the serialized model.
        - Fetches the latest features for a given ticker from the Feast **online store (Redis)**.
        - Prepares the features into the shape expected by the model.
        - Returns a prediction.
2.  **FastAPI Backend (`backend/`):**
    - Set up a FastAPI application in `main.py`.
    - Create a prediction endpoint (e.g., `/predict/{ticker}`) that executes the inference pipeline.
    - Define API data schemas (`schemas.py`) for requests and responses.
3.  **Containerization:**
    - Create a `Dockerfile` for the backend service.
    - Update `docker-compose.yml` to include the backend service and manage dependencies (e.g., ensure it can connect to Redis).

**Deliverable:** A containerized API that can serve a crypto prediction for a given ticker.

---

### Run 4: Basic Frontend Interface

**Goal:** Create a simple user interface to interact with the system.

**Key Tasks:**
1.  **UI Development (`frontend/app.py`):**
    - Build a simple web application using Streamlit or Dash.
    - The UI should include:
        - An input box for the stock ticker.
        - A button to trigger prediction.
        - A display area for the model's output.
        - (Optional) A basic chart to visualize historical data and the prediction.
2.  **API Integration:**
    - The frontend app will make HTTP requests to the backend API created in Run 3.
3.  **Containerization:**
    - Create a `Dockerfile` for the frontend service.
    - Add the frontend service to `docker-compose.yml` and configure networking between the frontend and backend.

**Deliverable:** A fully containerized stack (`frontend`, `backend`, `redis`) that allows a user to get a stock prediction through a web interface.

---

### Run 5: The Agentic Layer

**Goal:** Augment the system with an intelligent agent capable of using tools to answer more complex financial questions.

**Key Tasks:**
1.  **Agent Architecture (`src/agents/`):**
    - Design an agent using a framework like LangGraph.
    - Define the agent's state graph in `graph.py` and its processing nodes in `nodes.py`.
2.  **Tool Definition (`src/agents/tools.py`):**
    - Implement a set of tools the agent can use:
        - `get_stock_prediction`: A tool that calls the existing inference pipeline.
        - `get_company_news`: A tool that fetches recent news for a ticker from an external API.
        - `get_historical_data`: A tool that fetches historical data from the feature store.
3.  **Semantic Caching (`src/memory/semantic_cache.py`):**
    - Implement a cache (e.g., using Qdrant) to store the results of expensive agent queries, reducing latency and cost for repeated questions.
4.  **API Integration:**
    - Add a new endpoint to the FastAPI backend (e.g., `/agent/invoke`) that takes a natural language query and routes it to the agent.

**Deliverable:** An API endpoint that leverages an LLM agent to provide nuanced answers to financial questions by combining model predictions with other data sources.

---

### Run 6: Monitoring and Observability

**Goal:** Instrument the system to monitor its health, performance, and data quality.

**Key Tasks:**
1.  **Metrics Instrumentation:**
    - Add Prometheus client libraries to the FastAPI backend to expose key metrics (e.g., request latency, error rates, prediction counts).
2.  **Monitoring Stack Setup (`prometheus/`, `grafana/`):**
    - Configure Prometheus to scrape metrics from the backend.
    - Set up Grafana and create a basic dashboard to visualize the metrics from Prometheus.
3.  **ML Monitoring (`src/monitoring/`):**
    - Implement a data drift detection script (`drift.py`) to check if incoming data distributions have shifted significantly from the training data.
    - Create a separate `monitoring_app` that runs these checks and exposes the results as metrics for Prometheus.
4.  **Compose Integration:**
    - Add the Prometheus, Grafana, and `monitoring_app` services to the `docker-compose.yml` file.

**Deliverable:** A complete, locally runnable system with a Grafana dashboard for monitoring application and model health.

---

### Run 7: Production Hardening and Kubernetes Deployment

**Goal:** Prepare the application for a production-grade deployment on Kubernetes.

**Key Tasks:**
1.  **Kubernetes Manifests (`k8s/`):**
    - Create YAML manifests for all services: FastAPI, Frontend, Redis, Qdrant, Prometheus, Grafana.
    - Define Deployments, Services, ConfigMaps, and Persistent Volumes.
2.  **Dockerfile Optimization:**
    - Refine all Dockerfiles for production using best practices (e.g., multi-stage builds, non-root users, smaller base images).
3.  **Configuration Management (`src/config.py`):**
    - Refactor the configuration to be fully driven by environment variables, removing hardcoded values.
4.  **Deployment Scripts (`run_k8s.sh`):**
    - Create helper scripts to simplify applying the Kubernetes manifests.
5.  **Documentation (`README.md`, `doc/`):**
    - Write comprehensive documentation covering architecture, setup, and deployment procedures.

**Deliverable:** A fully documented, production-ready project that can be deployed to a Kubernetes cluster with a single script.

---

### Run 8: Advanced Indicators & Comprehensive Frontend

**Goal:** Enhance the system with more sophisticated trading indicators and develop a comprehensive, advanced frontend for in-depth crypto analysis.

**Key Tasks:**
1.  **Advanced Indicators (`src/features/advanced_indicators.py`):**
    - Implement placeholders for sophisticated indicators like:
        - Ichimoku Cloud
        - Bollinger Bands %B
        - On-Chain Metrics (e.g., NVT Ratio, MVRV-Z Score) - *placeholder implementation*
    - Integrate these new features into the existing data ingestion and feature store pipeline.
2.  **Advanced Frontend (`frontend-v2/`):**
    - Initialize a new, more robust frontend project (e.g., using React or Angular).
    - Design and build a multi-panel dashboard interface.
    - **Dashboard Components:**
        - Advanced charting library (e.g., TradingView Lightweight Charts) to visualize price action and the new indicators.
        - A dedicated panel for agent interaction (from Run 5).
        - A data grid to display historical data and feature values.
        - A news feed component.
    - **Backend Integration:**
        - The frontend will consume data from all relevant API endpoints (prediction, agent, historical data).
3.  **Containerization & Orchestration:**
    - Create a `Dockerfile` for the new advanced frontend.
    - Update `docker-compose.yml` to include the `frontend-v2` service, replacing the basic Streamlit/Dash frontend.

**Deliverable:** A sophisticated, containerized crypto dashboard that provides users with advanced indicators, agent-driven insights, and comprehensive data visualization, replacing the basic UI from Run 4.
