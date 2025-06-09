
Welcome to the **Streamlit Prediction App** — a user-friendly interface to upload images, ZIP archives, or videos, send them to a backend prediction API, and visualize the segmented results side-by-side with the originals.

---

## App Structure

The app is organized into three main pages:

### 1. Upload Files

- Upload image, ZIP, or video files.
- Files are stored in Streamlit's session state to persist during the session.

### 2. Run Predictions

- Select a previously uploaded file.
- Send the file to the backend API (`/predict/image/` or `/predict/video/`).
- You can download the predictions, and it stores them in ZIP files mapped by filename in the session state.

### 3. View Results

- Select a processed file to view results.
- Unzips the prediction ZIP in-memory.
- Displays original and segmented images side-by-side.
- Allows browsing multiple processed files easily.

---

## How to Use

1. **Upload files** on the "Upload Files" page.
2. Go to **"Run Predictions"** page, select a file, and click the button to send it to the backend.
3. After processing completes, switch to **"View Results"** page to inspect segmented outputs.

---