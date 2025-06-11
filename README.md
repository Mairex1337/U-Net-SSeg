# 🚗 U-Net Semantic Segmentation Model

![Segmentation Example](images/seg_example.png)

We trained a U-Net model for semantic segmentation using the [BDD100k](https://arxiv.org/abs/1805.04687) dataset. This dataset contains images from road scenes with **19 semantic classes**, captured under diverse conditions (e.g., weather, time, location).

> ✅ Our current model achieves **0.8797 accuracy** on the test set.
> 🎲 A random guess would yield just **1/19 ≈ 0.0526 accuracy**.

---

## 🛠️ Usage Instructions

### ✅ Pre-requisites

1. **Clone the Repository**

   ```bash
   git clone -b submission https://github.com/Mairex1337/U-Net-SSeg.git
   ```

2. **Create and Activate Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Navigate to Project Root**

   ```bash
   cd U-Net-SSeg
   ```

4. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

5. **Download the Model Checkpoint**

   ```bash
   python -m scripts.download_checkpoint
   ```

---

## 🧪 API Endpoints and Input Formats

The FastAPI backend supports the following endpoints for segmentation tasks:

### 1. `/predict/returns-json/` – **Base64 Image Prediction (JSON)**

Run segmentation on one or multiple **base64-encoded images**.


#### 🔹 Request Format

```json
{
  "image_names": ["image1.jpg", "image2.jpg"],
  "images": ["<base64-encoded-image1>", "<base64-encoded-image2>"]
}
```

You can use the provided `api_images.json`, or convert your own images with:

```bash
python -m scripts.img_json \
  --path-to-images /path/to/images \
  --output-path /path/to/save/json \
  --file-name output.json
```

#### 🔹 Response

* JSON with a list of prediction masks (also base64 encoded).
* Additionally writes `output.json` to the repo root (can be converted to `.png` images).

#### 🔄 Converting Output JSON to Images

To visualize the predictions:

```bash
python -m scripts.json_img \
  --path-to-json /path/to/output.json \
  --output-path /path/to/save/images
```

---

### 2. `/predict-image/returns-zip/` – **Image Upload Prediction**

Run segmentation on uploaded **image files** (`.jpg`, `.jpeg`) or a **`.zip` archive** of images.



#### 🔹 Request Format

  * `.jpg`, `.png`, or `.zip` file of images.

#### 🔹 Response

* Returns a `.zip` archive containing:
  * Original image/s.
  * Predicted mask/s.
  * Predicted colorized mask/s.

#### 🔹 Example (cURL)

```bash
curl -X POST http://127.0.0.1:8000/predict-image/ \
  -H "accept: application/zip" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_images.zip"
```

---

### 3. `/predict-video/returns-zip/` – **Video Upload Prediction**

Run segmentation on a **video file** (`.mp4`, `.avi`, `.mov`).


#### 🔹 Request Format

* a single `.mp4`, `.avi`, or `.mov` file.

#### 🔹 Response

* Returns a `.zip` archive containing:

  * Extracted video frames.
  * Predicted mask frames.
  * Predicted colorized mask frames.
  * Original video.
  * Predicted video.

#### 🔹 Example (cURL)

```bash
curl -X POST http://127.0.0.1:8000/predict-video/ \
  -H "accept: application/zip" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_video.mp4"
```

---

You can access all endpoints and test them interactively in your browser via:

🌐 **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)


---

## 📺 Running Streamlit Frontend

### 1. Ensure the API is Running

The Streamlit app needs the FastAPI server running.

### 2. Configure Environment Variables

Create a `.env` file in the root directory and define:

```env
API_BASE_URL=http://localhost:8000
SESSION_STATE_FILE=outputs/session.json
```

Use `.env.sample` as a reference.

### 3. Launch Streamlit

```bash
streamlit run streamlit/app.py
```

Then open: [http://localhost:8501](http://localhost:8501)

---

## 🐳 Run the API and Streamlit App with Docker

### 📥 Install Docker

👉 [https://docs.docker.com/desktop](https://docs.docker.com/desktop)

### 🧠 Memory Requirement

Ensure Docker has at least **7–8 GB RAM** allocated:
**Docker Desktop → Settings → Resources → Memory**

---

### ▶️ Start Both Apps

```bash
docker-compose up -d
```

### 🌍 Access the Apps

* **Streamlit frontend**: [http://127.0.0.1:8501](http://127.0.0.1:8501)
* **FastAPI docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 🛑 Stop the Services

```bash
docker-compose down
```


---

## 🗺️ Class Mapping & Legend

![Class Legend](images/color_legend.png)

```yaml
0: road
1: sidewalk
2: building
3: wall
4: fence
5: pole
6: traffic light
7: traffic sign
8: vegetation
9: terrain
10: sky
11: person
12: rider
13: car
14: truck
15: bus
16: train
17: motorcycle
18: bicycle
```

---


