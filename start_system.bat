@echo off
echo 🚀 Starting AI Travel Agent System
echo ====================================

echo.
echo 📡 Starting Django Backend Server...
cd travel_ai_backend
start "Django Backend" cmd /k "python manage.py runserver 8000"

echo.
echo ⏳ Waiting for backend to start...
timeout /t 5 /nobreak > nul

echo.
echo 🌐 Starting React Frontend...
cd ..\travel_ai_frontend
start "React Frontend" cmd /k "npm start"

echo.
echo ✅ System Starting!
echo.
echo 📋 Access Points:
echo   🔧 Backend API: http://localhost:8000/api/
echo   🌐 Frontend UI: http://localhost:3000
echo   📊 Admin Panel: http://localhost:8000/admin/
echo.
echo 🎯 The AI Travel Agent will be available at http://localhost:3000
echo.
pause
