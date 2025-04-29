# 🤟 Trustworthy Sign Language Translator

This project builds a **Sign Language Translator** model with a strong emphasis on **trustworthiness** — focusing not just on **accuracy**, but also on **fairness** across demographics and **robustness** under real-world conditions.

---

## 🛠 Project Structure

- **landmarks_data.csv** — Hand landmark coordinates + labels (gestures).
- **demographic_data.csv** — Demographic information (like skin tone) for fairness evaluation.
- **train_data.py** — Script to train the model, evaluate fairness, and save outputs.
- **sign_language_model.keras** — Trained model.
- **label_classes.npy** — Encoded gesture classes.
- **README.md** — Project description.

---

## 🚀 Features

- Hand landmark-based gesture recognition.
- Fairness evaluation across demographic groups.
- (Planned) Robustness evaluation under distortions like blur and occlusion.
- Model saved in modern `.keras` format.
- Label encoder saved separately for easy deployment.

---

## 📈 How It Works

1. **Data Loading**  
   Reads hand landmarks and demographic info (like skin tone).

2. **Model Training**  
   A fully connected neural network is trained to classify gestures.

3. **Fairness Evaluation**  
   Calculates **accuracy**, **precision**, and **recall** per demographic group.

4. **Saving Outputs**  
   Saves the trained model (`.keras`) and label classes (`.npy`) for reuse.

---

## ⚙️ Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Libraries used:
- `pandas`
- `numpy`
- `tensorflow`
- `scikit-learn`

---

## 🧪 How to Run

```bash
python train_data.py
```

Make sure the following files are in the same directory:
- `landmarks_data.csv`
- `demographic_data.csv`
- `train_data.py`

---

## 📋 Notes

- We **normalize** input data, which improves generalization under different lighting and slight noise automatically.
- **Fairness** is checked by splitting evaluation results based on demographic groups like skin tone.
- **Robustness** evaluation (blur, occlusion, lighting conditions) is planned for future enhancement.

---

## 🚀 Future Work

- Add full-sentence recognition (not just isolated gestures).
- Implement robustness testing against real-world distortions.
- Add explainability (why the model made a prediction).
- Deploy lightweight model to **Edge devices** (like mobile).

---

## 🙏 Acknowledgements

Inspired by the need for trustworthy, inclusive AI systems that work equally for everyone.

---
