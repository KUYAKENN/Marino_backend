# Face Recognition System

A comprehensive face recognition system using ArcFace/InsightFace with Flask API and Supabase database integration.

## 🚀 Quick Start

1. **Setup Database**: Run `QUICK_SETUP.sql` in your Supabase dashboard
2. **Start Server**: Run `python main.py`
3. **Test System**: Run `python test_complete_system.py`

## 📁 Project Structure

```
backend-python/
├── main.py                           # Main application entry point
├── QUICK_SETUP.sql                   # Database setup (run this first!)
├── requirements.txt                  # Python dependencies
├── test_complete_system.py           # Complete system testing
├── start_face_recognition.bat        # Windows batch launcher
├── start_face_recognition.ps1        # PowerShell launcher
├── .env                             # Environment variables
└── src/
    ├── flask_app.py                 # Flask API endpoints
    ├── arcface_service.py           # Face recognition engine
    ├── supabase_service.py          # Database operations
    └── __init__.py                  # Package initialization
```

## 🛠 Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Setup environment variables in `.env`:
```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_service_role_key
```

3. Run the database setup in Supabase:
   - Copy contents of `QUICK_SETUP.sql`
   - Paste in Supabase SQL Editor
   - Execute

## 🎯 API Endpoints

- `GET /health` - Health check
- `POST /recognize` - Recognize faces in images
- `POST /enroll` - Enroll new faces
- `POST /extract-landmarks` - Extract facial landmarks
- `GET /faces` - List enrolled faces
- `GET /face-status` - Check enrollment status
- `POST /sync-faces-from-db` - Sync existing face images

## 🧪 Testing

Run the complete system test:
```bash
python test_complete_system.py
```

## 🚀 Usage

1. **Start the server**:
```bash
python main.py
```

2. **Access the API**:
   - Server runs on `http://localhost:5000`
   - Welcome page: `http://localhost:5000/welcome`

3. **Face Recognition**:
   - Send POST request to `/recognize` with base64 image
   - System returns matched user or "no match"

## 🔧 Features

- ✅ ArcFace face recognition with 512-dimensional embeddings
- ✅ Supabase database integration
- ✅ Face landmark extraction
- ✅ Comprehensive logging
- ✅ Auto-sync from existing face images
- ✅ Real-time face enrollment
- ✅ High accuracy recognition (configurable threshold)

## 📊 Database Schema

The system creates these tables:
- `face_embeddings` - Stores face embeddings and metadata
- `face_landmarks` - Stores facial landmark data
- `face_recognition_log` - Logs all recognition attempts
- `face_enrollment_log` - Logs enrollment activities

## 🎉 Ready to Use!

Your face recognition system is now clean, organized, and ready for production use!
- **RESTful API**: Complete set of endpoints for face management

## Start Service

### Step 1 - Create environment

Install requirements:

```bash
pip install -r requirements.txt
```

### Step 2 - Set up environment variables

Create a `.env` file with your Supabase credentials:

```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

### Step 3 - Start service locally

1. Run service with main method:

```bash
python main.py
```

2. The service will be available at http://localhost:5000

## API Endpoints

### Core Recognition
- `POST /recognize` - Recognize face from image and mark attendance
- `GET /health` - Health check endpoint

### Face Enrollment & Management
- `POST /enroll` - Enroll a new face for recognition
- `GET /faces` - List all enrolled faces
- `DELETE /faces/{user_id}` - Remove a specific user's face

### Facial Analysis
- `POST /extract-landmarks` - Extract facial landmarks and information from image

### Attendance Management
- `GET /attendance` - Get today's attendance records
- `GET /attendance/today` - Get detailed today's attendance
- `GET /attendance/stats` - Get attendance statistics
- `GET /attendance/check/{user_id}` - Check specific user's attendance

### System Management
- `POST /initialize` - Initialize face database from Supabase users
- `POST /refresh` - Refresh face database
- `GET /stats` - Get system statistics
- `POST /threshold` - Update recognition threshold

## Usage Examples

### Enroll a new face:
```bash
curl -X POST http://localhost:5000/enroll \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "image": "data:image/jpeg;base64,/9j/4AAQ...",
    "user_data": {
      "firstName": "John",
      "lastName": "Doe",
      "email": "john@example.com"
    }
  }'
```

### Extract facial landmarks:
```bash
curl -X POST http://localhost:5000/extract-landmarks \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,/9j/4AAQ..."
  }'
```

### Recognize face:
```bash
curl -X POST http://localhost:5000/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,/9j/4AAQ..."
  }'
```

## Testing

Run the test script to verify all endpoints:

```bash
python test_face_endpoints.py
```

## Technical Details

- **Face Recognition**: Uses InsightFace ArcFace model for high-accuracy face recognition
- **Facial Landmarks**: Extracts 5-point facial landmarks (eyes, nose, mouth corners)
- **Database**: Stores face embeddings locally in pickle format, user data in Supabase
- **CORS**: Configured for cross-origin requests
- **Image Processing**: Supports base64 encoded images
