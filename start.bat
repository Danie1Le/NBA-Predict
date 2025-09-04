@echo off
echo 🏀 Starting NBA Game Predictor...
echo ==================================

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running. Please start Docker first.
    pause
    exit /b 1
)

REM Build and start services
echo 🚀 Building and starting services...
docker-compose up --build -d

REM Wait for services to be ready
echo ⏳ Waiting for services to start...
timeout /t 10 /nobreak >nul

REM Check if services are running
echo 🔍 Checking service status...
docker-compose ps

echo.
echo ✅ NBA Game Predictor is now running!
echo.
echo 🌐 Access the application at:
echo    • Frontend: http://localhost:3000
echo    • Backend API: http://localhost:8000
echo    • API Docs: http://localhost:8000/docs
echo.
echo 📊 The application will:
echo    • Always stay running (no sleeping)
echo    • Pre-load all models on startup
echo    • Provide instant predictions
echo.
echo 🛑 To stop the application, run: docker-compose down
pause
