# 🚗 U-Net Semantic Segmentation Model

![Segmentation Example](images/seg_example.png)

We trained a U-Net model for semantic segmentation using the [BDD100k](https://arxiv.org/abs/1805.04687) dataset. This dataset contains images from road scenes with **19 semantic classes**, captured under diverse conditions (e.g., weather, time, location). 

![Class Distribution](images/class_distribution.png)


Due to labelling inconsistencies and extreme class imabalance we combined and excluded classes such that we ended up using 14. We merged `car + truck + bus`, `wall + fence`, and `person + rider`. We removed the `train` class due to being extremely rare and being commonly mislabeled.


### 📊 Evaluation Metrics by Loss Function

As class imbalance is the key challenge for the semantic segmentation task, we decided to compare 4 different losses that account for that in different manners.

| Loss Function | Pixel Accuracy | Mean Accuracy | Mean IoU | Mean Dice (F1) |
|---------------|----------------|----------------|----------|----------------|
| Weighted CEL  | 0.8926         | 0.6653         | 0.5270   | 0.6554         |
|OHEM CEL       | 0.8965         | 0.5833         | 0.5046   | 0.6216         |
|Dice loss      |          |          |    |         |
|Dice loss + CEL      |          |          |    |          |
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

### Launch FastAPI

```bash
python -m api.main
```

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

## 🏁 Training a Fresh Model from Scratch

To train the a model from scratch on your own BDD100k dataset:

### 1. Get the Dataset

```bash
python3 -m scripts.download_dataset
```

### 2. Adjust hyperparameters (optional)

In the `cfg.yaml` file, you can adjust e.g. `LR`, `Batch size`, `Epochs`, etc. under `hyperparameters/{model_name}`.

### 3. Train the model

```bash
python3 -m scripts.train.py --model {model_name}
```
Insert 'unet' or 'baseline' for 'model_name'.

__NOTE__:
- Training will require at least 8GB of RAM. 
- Make sure the `venv` with dependencies is activated.
- Training even one epoch of the U-Net model without an NVIDIA GPU is likely computationally unfeasible.

### 4. Viewing results

All checkpoints and logs will be stored in the _run directory_ which can be found here: `outputs/{model_name}/{run_id}`

### 5. Evaluate the trained model

If you would like to evaluate the trained model, you can do so via the appropriate `run_id` and the `model_name`:

```bash
python3 -m scripts.eval --model {model_name} --run-id {run_id}
```

Evaluation will automatically use the _best_ checkpoint.
After evaluation, you will find metric results as well as predictions from the model in the _run directory_.

---

## 🗺️ Class Mapping & Legend

![Class Legend](images/color_legend.png)

```yaml
  0: road
  1: sidewalk
  2: building
  3: wall/fence
  4: pole
  5: traffic light
  6: traffic sign
  7: vegetation
  8: terrain
  9: sky
  10: person/rider
  11: car/truck/bus
  12: motorcycle
  13: bicycle
```

---


