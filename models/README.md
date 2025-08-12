# Face Detection Models

This directory contains pre-trained models for face detection.

## Required Models

For optimal face detection performance, please download the following OpenCV DNN models:

### 1. OpenCV Face Detection Model
- **Model file**: `opencv_face_detector_uint8.pb`
- **Config file**: `opencv_face_detector.pbtxt`
- **Source**: OpenCV DNN Face Detection models
- **Download**: You can find these models in the OpenCV samples or download from:
  - https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector

### 2. Model Usage
- The Face Detection service will automatically look for these models in this directory
- If models are not found, the system will fall back to Haar Cascade classifiers
- DNN models provide better accuracy and performance

### 3. Installation Instructions
1. Download the model files from the OpenCV repository
2. Place `opencv_face_detector_uint8.pb` and `opencv_face_detector.pbtxt` in this directory
3. Restart the application to load the DNN models

### 4. Alternative Models
- The system also supports using custom trained models
- Ensure the model format is compatible with OpenCV's DNN module
- Update the model paths in `infrastructure/ml_models/face_detector.py` if using custom models

## Notes
- Models are not included in the repository due to file size constraints
- The application will work without these models but with reduced accuracy
- For production use, it's recommended to download and use the DNN models
