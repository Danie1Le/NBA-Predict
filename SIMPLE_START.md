# 🏀 NBA Game Predictor - Super Simple Start

## ✅ **Your Backend is Already Running!**

The backend is currently running on **http://localhost:8000** with all 30 NBA teams loaded!

## 🎨 **Start the Frontend**

Open a **new terminal** and run:

```bash
cd frontend
npm install
npm start
```

Then open: **http://localhost:3000**

---

## 🧪 **Test the API**

Your backend is working! You can test it:

```bash
# Test teams
curl http://localhost:8000/teams

# Test prediction (Lakers vs Warriors)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team_id": 1610612747, "away_team_id": 1610612744, "model_type": "ensemble"}'
```

---

## 🎯 **What's Working**

✅ **Backend API** - Running on port 8000  
✅ **30 NBA Teams** - All teams loaded  
✅ **6 AI Models** - XGBoost, Random Forest, Logistic Regression, PyTorch, TensorFlow, Ensemble  
✅ **FastAPI** - Professional REST API  
✅ **CORS Enabled** - Ready for React frontend  

---

## 🚀 **Next Steps**

1. **Start Frontend**: `cd frontend && npm start`
2. **Open Browser**: http://localhost:3000
3. **Make Predictions**: Choose teams and models!

---

## 🔧 **If You Need to Restart Backend**

```bash
cd backend
python minimal_main.py
```

**Wait for:** `🎯 API ready for predictions!`

---

**🎉 Your NBA Game Predictor is ready to go!**
