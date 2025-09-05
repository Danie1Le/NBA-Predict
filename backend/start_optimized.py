#!/usr/bin/env python3
"""
Optimized startup script for NBA Game Predictor API
Ensures models are pre-loaded for fast predictions
"""

import os
import sys
import uvicorn
from pathlib import Path

def start_api():
    """Start the API with optimized settings"""
    
    # Set environment variables for production
    os.environ.setdefault("CORS_ORIGINS", 
        "http://localhost:3000,http://127.0.0.1:3000,https://nba-predict.vercel.app,https://nba-predict-7hz6.onrender.com")
    
    # Get port from environment
    port = int(os.getenv("PORT", 8000))
    
    print("🚀 Starting NBA Game Predictor API (Optimized)")
    print("="*60)
    print(f"📁 Current working directory: {os.getcwd()}")
    print(f"🌐 Port: {port}")
    print(f"🔧 Environment: {'Production' if os.getenv('PORT') else 'Development'}")
    
    # Check if cache files exist
    cache_dir = Path("model_cache")
    cache_file = cache_dir / "cached_models.pkl"
    
    if cache_file.exists():
        print("✅ Model cache found - API will start with instant predictions!")
    else:
        print("⚠️ No model cache found - models will be trained on first request")
        print("💡 Run 'python prebuild_models.py' to pre-build models")
    
    print("🎯 Starting API server...")
    print("="*60)
    
    # Start the server with optimized settings
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload in production
        log_level="info",
        access_log=False,  # Disable access logs for better performance
        workers=1  # Single worker for better memory usage
    )

if __name__ == "__main__":
    start_api()
