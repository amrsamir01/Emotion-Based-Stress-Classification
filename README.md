
# 📌 Emotion Classification using CNN on KDEF Dataset

## 🧠 Project Summary
This project implements a Convolutional Neural Network (CNN) to classify human facial emotions using the **KDEF (Karolinska Directed Emotional Faces)** dataset. The model performs binary classification to distinguish between two emotional classes.

---

## 🚀 Key Features
- CNN architecture with **3 convolutional blocks** and **fully connected layers**
- Advanced **data augmentation** techniques for improved generalization
- Use of **class weighting** to address class imbalance
- Integrated **early stopping** and **learning rate scheduling**
- Evaluated using **accuracy, precision, recall,** and **confusion matrix**

---

## 📂 Dataset
- **Source**: KDEF Dataset (Karolinska Institute)
- **Classes**: 2 emotions (e.g., *happy* vs *neutral*, depending on your dataset configuration)
- **Images**: 
  - ~4409 training images
  - ~1891 test images
- **Image Size**: 256x256 pixels
- **Split**:
  - 80% Training
  - 20% Validation
  - Separate Test Set

---

## 🏗️ Model Architecture

```
Input → Data Augmentation → Rescaling →
[Conv2D → Conv2D → MaxPooling → BatchNorm] x3 →
Flatten → Dense(512) → Dropout → BatchNorm →
Dense(256) → Dropout → BatchNorm →
Dense(1, Sigmoid)
```

- Total Parameters: ~135M
- Optimizer: AdamW (with weight decay)
- Loss Function: Binary Cross-Entropy

---

## 📊 Training Overview
- **Epochs**: Up to 100 (with EarlyStopping)
- **Best Validation Accuracy**: ~89.4%
- **Performance Metrics Tracked**:
  - Training and Validation Accuracy
  - Precision, Recall
  - Loss curves

---

## 🔧 Key Libraries
- `TensorFlow` & `Keras`
- `Scikit-learn` (metrics)
- `Matplotlib` & `Seaborn` (visualization)
- `NumPy`

---

## 📈 Results & Evaluation
- High recall across most epochs (up to 0.98)
- Balanced precision and recall across validation set
- Confusion matrix and classification report used for detailed analysis

---

## 📌 How to Run
```bash
git clone https://github.com/yourusername/emotion-cnn-kdef.git
cd emotion-cnn-kdef

# Set up environment
pip install -r requirements.txt

# Run the notebook or script
python train_model.py
```

---

## 📊 Visual Output (Optional for GitHub)
- Include:
  - Confusion matrix image
  - Accuracy/loss graphs
  - Sample predictions

---

## ✅ Future Improvements
- Expand to multi-class emotion classification (e.g., happy, sad, angry, etc.)
- Experiment with transfer learning (e.g., ResNet, MobileNet)
- Deploy as a web app using Streamlit

---

## 📚 References
- [KDEF Dataset Info](https://www.kdef.se/)
- [TensorFlow Docs](https://www.tensorflow.org/)
