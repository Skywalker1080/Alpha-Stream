from Backend.main import app
import uvicorn

if __name__ == "__main__":
    # Redirect execution to the new backend structure
    uvicorn.run("Backend.main:app", host="0.0.0.0", port=8000)