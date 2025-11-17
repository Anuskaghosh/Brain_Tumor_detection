# Brain Tumor Detection Web Application

A modern web application for detecting brain tumors in medical images using deep learning.

## Features

- 🎨 Beautiful, modern UI with gradient design
- 📤 Drag and drop image upload
- 🧠 Real-time brain tumor detection
- 📊 Detailed prediction results with confidence scores
- 📱 Responsive design

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

### 3. Use the Application

1. Open your browser and navigate to `http://localhost:5000`
2. Click on the upload area or drag and drop a brain scan image
3. Click "Analyze Image" to get the prediction
4. View the results showing whether a tumor was detected and the confidence level

## Model Information

- Model file: `brain_tumor_model.h5`
- The model expects images to be resized to 150x150 pixels
- Supports both binary classification (Tumor/No Tumor) and multi-class classification

## Technical Details

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Deep Learning**: TensorFlow/Keras
- **Image Processing**: PIL/Pillow

## Notes

- Make sure your model file (`brain_tumor_model.h5`) is in the same directory as `app.py`
- The application automatically handles image preprocessing (resizing, normalization)
- Supported image formats: JPG, PNG, JPEG

