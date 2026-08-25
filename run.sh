#!/bin/bash
# FastAPI Face Recognition Server
# Run with: ./run.sh

cd "$(dirname "$0")"

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run FastAPI server
echo "🚀 Starting EmAtSy FastAPI Server..."
uvicorn fast_api.main:app --host 0.0.0.0 --port 8000 --reload
