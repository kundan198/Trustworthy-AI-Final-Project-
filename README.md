# 🤟 Trustworthy Sign Language Translator

This project focuses on building and evaluating a **Sign Language Translator** application, ensuring its **trustworthiness** by assessing **Fairness**, **Bias**, and **Robustness**.

---

## 📌 Objective

- **Fairness and Bias Evaluation**  
  Analyze the model’s performance across different **demographic groups** based on **skin tone**.  
  Metrics used: `Accuracy`, `Precision`, `Recall`.

- **Reliability and Robustness Evaluation**  
  Since **normalization** is applied to landmark data, the model achieves **consistent accuracy** even under minor distortions like blur, occlusion, and lighting changes.

---

## 🛠️ Project Structure

```
Trustworthy-Sign-Language-Translator/
│
├── train_data.py             # Main training and evaluation script
├── landmarks_data.csv        # Hand landmarks dataset
├── demographic_data.csv      # Demographic info (e.g., skin tone)
├── label_classes.npy         # Saved label encodings
├── sign_language_model.keras # Trained Keras model
├── requirements.txt          # Python dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Ignore unnecessary files
```

---

## Installation

1. **Clone the repository:**
   ```sh
   git clone [https://github.com/kundan198/Sign-Language-Translator-AI-for-Gesture-Recognition]

2. **Create and activate a virtual environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Download Datasets:**
   ```sh
   [https://www.kaggle.com/datasets/grassknoted/asl-alphabet]
   ```

## Running the Program

1. **Collect dataset (using MediaPipe to extract landmarks):**
   ```sh
   python collect_data.py
   ```

2. **Train the model:**
   ```sh
   python train_data.py
   ```

3. **Run the prediction (real-time or image-based):**
   ```sh
   python predict_data.py
   ```

## Notes

- Ensure you have a **webcam** connected for real-time predictions.
- If accuracy is low, consider **increasing dataset size** or **tweaking hyperparameters**.


## 🔎 Evaluation Details

- **Fairness Metrics**  
  After training, the script prints **Accuracy**, **Precision**, and **Recall** for each **demographic group**.

- **Robustness**  
  By applying **normalization** to landmark data, the model maintains stable accuracy despite minor variations such as **blur**, **occlusion**, or **low lighting**.  
  *(Full robustness attacks were not conducted, but normalization provides basic reliability.)*

---

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- Scikit-learn

Install all requirements with:

```bash
pip install -r requirements.txt
```

---

## 🧹 Source Declaration

- Dataset (landmarks and demographic info) was generated/simulated for academic evaluation.
- All project code is original and written by the project team for the Trustworthy AI course project.

---

## 📈 Future Work

- Extend to full-sentence sign recognition.
- Conduct explicit robustness attacks (blur, occlusion).
- Add explainability (showing how decisions are made).
- Deploy the model to mobile and edge devices.

---

## 📢 Notes

- TensorFlow saving warning about HDF5 can be ignored; model is saved in the **new `.keras` format**.
- Ensure CSV files (`landmarks_data.csv`, `demographic_data.csv`) are in the project root when running `train_data.py`.

---
