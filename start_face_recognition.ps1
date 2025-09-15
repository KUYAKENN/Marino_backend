#!/usr/bin/env pwsh
# Face Recognition System Startup Script
# This script helps you run the face recognition system components

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "FACE RECOGNITION SYSTEM STARTUP" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python and try again" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if required files exist
if (-not (Test-Path "src\flask_app.py")) {
    Write-Host "❌ ERROR: src\flask_app.py not found" -ForegroundColor Red
    Write-Host "Please ensure all face recognition files are present" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

if (-not (Test-Path "QUICK_SETUP.sql")) {
    Write-Host "❌ ERROR: QUICK_SETUP.sql not found" -ForegroundColor Red
    Write-Host "Please ensure the database setup file is present" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Show menu
Write-Host "📋 Available Options:" -ForegroundColor Yellow
Write-Host "1. 🚀 Start Flask Server" -ForegroundColor White
Write-Host "2. 🔍 Scan Face Database" -ForegroundColor White
Write-Host "3. 🧪 Test Complete System" -ForegroundColor White
Write-Host "4. 📖 View Instructions" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Select option (1-4)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "STARTING FLASK SERVER" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "🚀 Server will run on http://localhost:5000" -ForegroundColor Cyan
        Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
        Write-Host ""
        
        try {
            python src\flask_app.py
        } catch {
            Write-Host "❌ Failed to start server" -ForegroundColor Red
        }
    }
    
    "2" {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "RUNNING FACE SCANNER" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "🔍 This will scan existing face images from your database" -ForegroundColor Cyan
        Write-Host ""
        
        try {
            python scan_faces.py
        } catch {
            Write-Host "❌ Failed to run face scanner" -ForegroundColor Red
        }
    }
    
    "3" {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "TESTING COMPLETE SYSTEM" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "🧪 This will test all face recognition endpoints" -ForegroundColor Cyan
        Write-Host ""
        
        if (Test-Path "test_complete_system.py") {
            try {
                python test_complete_system.py
            } catch {
                Write-Host "❌ Failed to run system test" -ForegroundColor Red
            }
        } else {
            Write-Host "⚠️  test_complete_system.py not found" -ForegroundColor Yellow
            Write-Host "Running basic face scanner instead..." -ForegroundColor Cyan
            try {
                python scan_faces.py
            } catch {
                Write-Host "❌ Failed to run face scanner" -ForegroundColor Red
            }
        }
    }
    
    "4" {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "📖 SETUP INSTRUCTIONS" -ForegroundColor Cyan
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "📋 Before using the face recognition system:" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "1. 🗄️  Database Setup:" -ForegroundColor White
        Write-Host "   - Open your Supabase dashboard" -ForegroundColor Gray
        Write-Host "   - Go to SQL Editor" -ForegroundColor Gray
        Write-Host "   - Copy and paste contents of QUICK_SETUP.sql" -ForegroundColor Gray
        Write-Host "   - Run the SQL commands" -ForegroundColor Gray
        Write-Host ""
        Write-Host "2. 🚀 Start Server:" -ForegroundColor White
        Write-Host "   - Run option 1 to start Flask server" -ForegroundColor Gray
        Write-Host "   - Server runs on http://localhost:5000" -ForegroundColor Gray
        Write-Host ""
        Write-Host "3. 🔍 Scan Existing Faces:" -ForegroundColor White
        Write-Host "   - Run option 2 to scan database images" -ForegroundColor Gray
        Write-Host "   - This processes existing user face images" -ForegroundColor Gray
        Write-Host ""
        Write-Host "4. 🧪 Test System:" -ForegroundColor White
        Write-Host "   - Run option 3 to test all endpoints" -ForegroundColor Gray
        Write-Host "   - Verifies everything is working" -ForegroundColor Gray
        Write-Host ""
        Write-Host "📍 API Endpoints:" -ForegroundColor Yellow
        Write-Host "   POST /recognize - Recognize faces in images" -ForegroundColor Gray
        Write-Host "   POST /enroll - Enroll new faces" -ForegroundColor Gray
        Write-Host "   GET  /faces - List all enrolled faces" -ForegroundColor Gray
        Write-Host "   GET  /face-status - Check enrollment status" -ForegroundColor Gray
        Write-Host "   POST /sync-faces-from-db - Sync existing images" -ForegroundColor Gray
        Write-Host ""
        Read-Host "Press Enter to return to menu"
        
        # Restart script
        & $MyInvocation.MyCommand.Path
        return
    }
    
    default {
        Write-Host "❌ Invalid choice. Please select 1, 2, 3, or 4" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "OPERATION COMPLETE" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Read-Host "Press Enter to exit"
