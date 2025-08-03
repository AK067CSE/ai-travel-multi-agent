# Troubleshooting Guide

## Common Issues and Solutions

### 1. "Failed to fetch" Error

**Symptoms:**
- Frontend shows "Failed to fetch" error
- Chat interface doesn't respond
- Network errors in browser console

**Solutions:**

#### Check Backend Server
1. Make sure Django server is running:
   ```bash
   cd simplified_travel_ai/backend
   python manage.py runserver
   ```

2. Verify server is accessible:
   - Open: http://127.0.0.1:8000/api/status/
   - Should show JSON response with system status

#### Check Frontend Server
1. Make sure React server is running:
   ```bash
   cd simplified_travel_ai/frontend
   npm start
   ```

2. Verify frontend is accessible:
   - Open: http://localhost:3000
   - Should show the travel AI interface

#### Test API Connection
1. Open: http://localhost:3000/test.html
2. Click "Test API Connection" button
3. Should show successful response

### 2. HTTP 400 Bad Request

**Symptoms:**
- API returns 400 status code
- "Invalid request data" error

**Solutions:**
1. Check request format in browser developer tools
2. Ensure message field is not empty
3. Verify Content-Type header is set to application/json

### 3. CORS Errors

**Symptoms:**
- "Access to fetch blocked by CORS policy" error
- Cross-origin request errors

**Solutions:**
1. Restart Django server after CORS settings changes
2. Check CORS settings in `backend/settings.py`
3. Ensure frontend is running on port 3000

### 4. Port Already in Use

**Symptoms:**
- "Port 8000 is already in use" error
- "Port 3000 is already in use" error

**Solutions:**

#### For Backend (Port 8000):
```bash
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F

# Or use different port
python manage.py runserver 8001
```

#### For Frontend (Port 3000):
```bash
# Find process using port 3000
netstat -ano | findstr :3000

# Kill the process
taskkill /PID <PID> /F

# Or set different port
set PORT=3001 && npm start
```

### 5. Database Errors

**Symptoms:**
- "no such table" errors
- Database connection issues

**Solutions:**
1. Run migrations:
   ```bash
   cd simplified_travel_ai/backend
   python manage.py makemigrations
   python manage.py migrate
   ```

2. If issues persist, delete db.sqlite3 and run migrations again

### 6. Missing Dependencies

**Symptoms:**
- Import errors
- Module not found errors

**Solutions:**

#### Backend:
```bash
cd simplified_travel_ai/backend
pip install -r requirements.txt
```

#### Frontend:
```bash
cd simplified_travel_ai/frontend
npm install
```

### 7. Environment Variables

**Symptoms:**
- OpenAI integration not working
- Configuration errors

**Solutions:**
1. Copy `.env.example` to `.env` in backend folder
2. Add your API keys to the `.env` file
3. Restart Django server

## Quick Diagnostic Commands

### Check if servers are running:
```bash
# Check backend (should show Django server)
netstat -an | findstr :8000

# Check frontend (should show React server)
netstat -an | findstr :3000
```

### Test API directly:
```powershell
$body = @{message="Test"} | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/chat/" -Method POST -Body $body -ContentType "application/json"
```

### Check system status:
```bash
# Open in browser
http://127.0.0.1:8000/api/status/
```

## Getting Help

If you're still experiencing issues:

1. Check the browser developer console for detailed error messages
2. Check Django server logs in the terminal
3. Try the test page: http://localhost:3000/test.html
4. Restart both servers and try again

## System Requirements

- Python 3.8+
- Node.js 14+
- Modern web browser with JavaScript enabled
- Available ports: 3000 (frontend) and 8000 (backend)
