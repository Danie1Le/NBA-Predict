#!/usr/bin/env python3
"""
Clean, modular FastAPI backend for NBA Game Predictor
"""

import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import our modular components
from data_loader import DataLoader
from prediction_service import PredictionService
from routes import create_routes


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    
    # Initialize FastAPI app
    app = FastAPI(
        title="NBA Game Predictor API",
        description="Predict NBA game outcomes using machine learning",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify actual origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize services
    data_loader = DataLoader()
    prediction_service = PredictionService(data_loader)
    
    # Create and include routes
    router = create_routes(data_loader, prediction_service)
    app.include_router(router)
    
    # Startup event
    @app.on_event("startup")
    async def startup_event():
        """Load data and models on startup"""
        success = await data_loader.load_all_data()
        if not success:
            print("⚠️ Some data failed to load, but server will continue")
    
    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Render default port is 8000
    
    print(f"🚀 Starting NBA Game Predictor API on port {port}")
    print(f"📁 Current working directory: {os.getcwd()}")
    print(f"📄 Script location: {__file__}")
    
    # Run the app - disable reload in production
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload in production
        log_level="info"
    )
