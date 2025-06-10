# 🚘 Semantic Segmentation with U-Net

Welcome to the **Streamlit App**, which enables you to perform semantic segmentation on traffic scences using a U-Net CNN.

The model architecture is based on the original U-Net paper, with adjustments to support:
- **Custom input sizes**
- **Three-dimensional (color) images**
- **Padding** to ensure the predicted masks match the original image dimensions

We trained our model on the **BDD100K** dataset using the university's supercomputer **Hábrók**.
This app supports **inference on images, ZIP files**, and **entire videos**.

---

## 🧭 App Structure

The app is organized into three main pages:

### 1. 📁 Upload Files
- Upload image, ZIP, or video files.
- Files are stored in Streamlit's session state for use throughout the session.

### 2. ⚙️ Run Predictions
- Select a previously uploaded file.
- The file is sent to the backend API for inference.
- Download results as a ZIP, or access them from session state.

### 3. 🖼️ View Results
- Select a processed file to view its results.
- Display original and segmented images side-by-side.
- Supports browsing multiple files.
- Compare original and segmented videos directly in the app.

---

## 🚀 How to Use

1. Start the API either locally or with Docker
2. Start the Streamlit app
3. **Upload your files** on the *Upload Files* page.
4. Navigate to the *Run Predictions* page, select a file, and start inference.
5. Once processing is complete, visit the *View Results* page to inspect your segmented outputs.

---
