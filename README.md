# Crypto Prism Ops

Crypto Prism Ops is an advanced institutional-grade market intelligence platform designed for cryptocurrency analysis and forecasting. It leverages multi-agent research, technical price forecasting, and real-time sentiment analysis to provide strategic insights.

## Features

- Autonomous multi-agent research powered by LangGraph.
- LSTM-based technical price forecasting.
- Real-time news sentiment analysis.
- Feature store integration with Feast.
- Scalable vector storage with Qdrant.
- Experiment tracking and model management with MLflow.

## Getting Started

### Prerequisites

- Docker
- Docker Compose
- Ollama

### Running the Project

The simplest way to run the entire application is using Docker Compose. This will build and start all necessary services including the FastAPI backend, Streamlit frontend, Redis, and Qdrant.

1. Clone the repository to your local machine.
2. Navigate to the project root directory.
3. Run the following command:

```bash
docker-compose up -d --build
```
4. You also need Ollama server running locally

```bash
ollama serve
```

Once the containers are running, you can access the following services:

- **Frontend (Streamlit):** http://localhost:8501
- **Backend (FastAPI):** http://localhost:8000
- **Redis Insights:** http://localhost:8001

## Future Support

Kubernetes support for orchestration and cloud-native deployment is coming soon.
