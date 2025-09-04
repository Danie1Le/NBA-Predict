#!/usr/bin/env python3
"""
Test script to verify the NBA Game Predictor setup
"""

import sys
import os
import requests
import time

def test_backend():
    """Test if the backend is running"""
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running")
            return True
        else:
            print(f"❌ Backend returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Backend is not running: {e}")
        return False

def test_teams_endpoint():
    """Test the teams endpoint"""
    try:
        response = requests.get("http://localhost:8000/teams", timeout=5)
        if response.status_code == 200:
            teams = response.json()
            print(f"✅ Teams endpoint working - {len(teams)} teams loaded")
            return True
        else:
            print(f"❌ Teams endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Teams endpoint error: {e}")
        return False

def test_models_endpoint():
    """Test the models endpoint"""
    try:
        response = requests.get("http://localhost:8000/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Models endpoint working - {len(models['available_models'])} models available")
            return True
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Models endpoint error: {e}")
        return False

def test_prediction():
    """Test a prediction"""
    try:
        # Test with Lakers (id: 1610612747) vs Warriors (id: 1610612744)
        prediction_data = {
            "home_team_id": 1610612747,  # Lakers
            "away_team_id": 1610612744,  # Warriors
            "model_type": "ensemble"
        }
        
        response = requests.post("http://localhost:8000/predict", json=prediction_data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction working - {result['model_used']} model")
            print(f"   Home win probability: {result['home_win_probability']:.3f}")
            print(f"   Away win probability: {result['away_win_probability']:.3f}")
            print(f"   Confidence: {result['confidence']}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Prediction error: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing NBA Game Predictor Setup")
    print("=" * 40)
    
    # Wait a moment for backend to start
    print("⏳ Waiting for backend to start...")
    time.sleep(2)
    
    # Test backend
    if not test_backend():
        print("\n❌ Backend is not running. Please start it first:")
        print("   cd backend")
        print("   python simple_main.py")
        return False
    
    # Test endpoints
    print("\n🔍 Testing API endpoints...")
    tests = [
        test_teams_endpoint,
        test_models_endpoint,
        test_prediction
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Your NBA Game Predictor is working perfectly!")
        print("\n🌐 You can now:")
        print("   1. Start the frontend: cd frontend && npm start")
        print("   2. Open http://localhost:3000 in your browser")
        print("   3. Make predictions with the beautiful UI!")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
