"""
API routes for NBA Game Predictor
"""

from fastapi import APIRouter, HTTPException
from typing import List
from models import PredictionRequest, PredictionResponse, TeamInfo, TeamStats
from dataLoader import DataLoader
from predictionService import PredictionService


def create_routes(data_loader: DataLoader, prediction_service: PredictionService) -> APIRouter:
    """Create API routes with dependency injection"""
    
    router = APIRouter()
    
    @router.get("/")
    async def root():
        """Root endpoint with API information"""
        return {
            "message": "🏀 NBA Game Predictor API",
            "version": "1.0.0",
            "status": "running",
            "endpoints": {
                "health": "/health",
                "teams": "/teams",
                "team_stats": "/team-stats/{team_id}",
                "predict": "/predict",
                "models": "/models"
            }
        }
    
    @router.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    
    @router.get("/teams", response_model=List[TeamInfo])
    async def get_teams():
        """Get list of all teams"""
        if data_loader.teams_df is None:
            raise HTTPException(status_code=500, detail="Teams data not loaded")
        
        return [
            TeamInfo(
                id=int(row['id']),
                abbreviation=row['abbreviation'],
                name=row['full_name']
            )
            for _, row in data_loader.teams_df.iterrows()
        ]
    
    @router.get("/team-stats/{team_id}", response_model=TeamStats)
    async def get_team_stats(team_id: int):
        """Get team statistics"""
        if data_loader.games_df is None:
            raise HTTPException(status_code=500, detail="Games data not loaded")
        
        try:
            stats = data_loader.get_team_stats(team_id)
            if stats is None:
                raise HTTPException(status_code=404, detail="Team not found")
            
            return TeamStats(**stats)
            
        except Exception as e:
            print(f"Error getting team stats: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error getting team stats: {str(e)}")
    
    @router.post("/predict", response_model=PredictionResponse)
    async def predict_game(request: PredictionRequest):
        """Predict the outcome of a game"""
        if data_loader.games_df is None or data_loader.features is None:
            raise HTTPException(status_code=500, detail="Games data not loaded")
        
        try:
            result = await prediction_service.make_prediction(
                request.home_team_id, 
                request.away_team_id, 
                request.model_type
            )
            
            if result is None:
                raise HTTPException(status_code=500, detail="Prediction failed")
            
            return PredictionResponse(**result)
            
        except Exception as e:
            print(f"❌ Prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    @router.get("/models")
    async def get_available_models():
        """Get list of available models and their status"""
        if data_loader.model_cache is None:
            return {
                "available_models": ["xgb", "rf", "logreg"],
                "model_descriptions": {
                    "xgb": "XGBoost (Gradient Boosting) - Fast & Accurate",
                    "rf": "Random Forest - Robust & Interpretable", 
                    "logreg": "Logistic Regression - Simple & Fast",
                    "pytorch": "PyTorch Neural Network - Advanced Deep Learning",
                    "tensorflow": "TensorFlow/Keras - Production-Ready Deep Learning",
                    "ensemble": "Ensemble (All Models) - Best Performance"
                },
                "status": "Traditional models available - will train on first prediction",
                "deep_learning_available": False,
                "recommended_model": "xgb"
            }
        
        available = data_loader.model_cache.get_available_models()
        has_deep_learning = any(model in available for model in ['pytorch', 'tensorflow', 'ensemble'])
        
        if has_deep_learning:
            return {
                "available_models": available,
                "model_descriptions": {
                    "xgb": "XGBoost (Gradient Boosting) - Fast & Accurate",
                    "rf": "Random Forest - Robust & Interpretable",
                    "logreg": "Logistic Regression - Simple & Fast",
                    "pytorch": "PyTorch Neural Network - Advanced Deep Learning",
                    "tensorflow": "TensorFlow/Keras - Production-Ready Deep Learning",
                    "ensemble": "Ensemble (All Models) - Best Performance"
                },
                "status": "All models available - instant predictions!",
                "deep_learning_available": True,
                "recommended_model": "ensemble"
            }
        else:
            return {
                "available_models": available,
                "model_descriptions": {
                    "xgb": "XGBoost (Gradient Boosting) - Fast & Accurate",
                    "rf": "Random Forest - Robust & Interpretable",
                    "logreg": "Logistic Regression - Simple & Fast",
                    "pytorch": "PyTorch Neural Network - Advanced Deep Learning",
                    "tensorflow": "TensorFlow/Keras - Production-Ready Deep Learning",
                    "ensemble": "Ensemble (All Models) - Best Performance"
                },
                "status": "Traditional models available - will train on first prediction",
                "deep_learning_available": False,
                "recommended_model": "xgb"
            }
    
    @router.post("/models/upgrade")
    async def upgrade_models():
        """Upgrade to deep learning models (placeholder)"""
        return {
            "success": True,
            "message": "Deep learning model training started in background",
            "estimated_time": "5-10 minutes",
            "current_models": data_loader.model_cache.get_available_models() if data_loader.model_cache else []
        }
    
    return router
