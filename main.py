"""Root entrypoint for cloud PaaS deployments (Railway, Render, Fly.io)."""
import os
import uvicorn
from backend.app.main import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=port)
