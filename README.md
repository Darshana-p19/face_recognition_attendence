# Face Recognition Attendance System

A desktop application for marking attendance using face recognition technology. Built with Python, OpenCV, and Tkinter.

## Features

- **Face Registration**: Register new users with their face images
- **Real-time Face Recognition**: Detect and recognize faces from camera feed
- **Automatic Attendance Marking**: Mark attendance automatically when a registered face is recognized
- **Attendance Records**: View all attendance records in a table format
- **Adjustable Recognition Threshold**: Fine-tune the face matching sensitivity
- **Data Persistence**: All user data and attendance records stored in Excel files

## Requirements

- Python 3.7 or higher
- Webcam/Camera
- Windows/Linux/MacOS

## Installation

1. **Clone or download this repository**

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv

3. Activate the virtual environment:

Windows:

bash
venv\Scripts\activate
Linux/Mac:

bash
source venv/bin/activate
Install required packages:

bash
pip install -r requirements.txt
Usage
Run the application:

bash
python attendance_system.py
Start the camera: Click "Start Camera" button

Register a new user:

Enter Name and Roll Number

Ensure only one face is visible in the camera

Click "Register User"

Mark attendance:

When a registered face appears in the camera, it will be automatically recognized

Green rectangle indicates recognized face with similarity score

Attendance is automatically marked in attendance.xlsx

View attendance: Click "View Attendance" to see all records

Adjust threshold: Use the slider to control recognition sensitivity (higher = stricter matching)

File Structure
text
├── attendance_system.py      # Main application file
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── users.xlsx                # Registered users database (auto-generated)
├── attendance.xlsx           # Attendance records (auto-generated)
├── face_features.pkl         # Extracted face features (auto-generated)
└── faces/                    # Captured face images (auto-generated)
    ├── roll1_name1.jpg
    └── roll2_name2.jpg
How It Works
Face Detection: Uses Haar Cascade Classifier to detect faces in real-time

Feature Extraction: Converts faces to Local Binary Patterns (LBP) histograms

Face Matching: Compares features using histogram intersection

Attendance Tracking: Prevents duplicate entries for the same day

Configuration
You can modify these constants in the code:

SIMILARITY_THRESHOLD: Recognition sensitivity (default: 70%)

CAMERA_WIDTH/HEIGHT: Camera resolution

FACE_WIDTH/HEIGHT: Size of saved face images

Troubleshooting
Camera not working?

Check if camera is connected and not used by another application

Try changing camera index in code (from 0 to 1 if you have multiple cameras)

Recognition not working well?

Ensure good lighting conditions

Adjust threshold slider lower for easier recognition

Re-register face with better lighting/angle

Python not found?

Make sure Python is installed and added to PATH

Use full Python path: C:\Python39\python.exe attendance_system.py

License
This project is for educational purposes.

Author
Created as a Python project for practice.

text

