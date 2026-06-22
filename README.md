# 💅 Automated Nail Pathology Classification Using Deep Learning

An interactive medical computer vision system designed to classify various localized health anomalies and systemic indicators directly from raw clinical images of fingernails. 

## 🎯 Project Overview & Clinical Problem
[cite_start]Computer-aided diagnostics heavily prioritize broad cutaneous skin lessons, leaving onychopathology (nail diagnostics) as a highly specialized niche[cite: 3]. [cite_start]Visible changes in nail morphology, texture, and coloration frequently serve as vital diagnostic indicators for localized fungal issues or severe underlying systemic conditions (such as digital clubbing linked to cardiovascular diseases)[cite: 8, 9]. 

This project solves this diagnostic challenge by utilizing transfer learning on top of a deep, pre-trained convolutional neural network to automatically screen nail images across 6 diagnostic target classes with **93.41% peak validation accuracy**.

### Target Diagnostic Classes
* [cite_start]**Acral Lentiginous Melanoma** (High-risk critical class) [cite: 10]
* [cite_start]**Healthy Nail** [cite: 12]
* [cite_start]**Onychogryphosis** [cite: 12]
* [cite_start]**Blue Finger** [cite: 12]
* [cite_start]**Clubbing** [cite: 12]
* [cite_start]**Pitting** [cite: 12]

---

## 📊 Dataset Sourcing
[cite_start]The model was trained on **Nikhil Gurav's Specialized Nail Disease Detection Dataset** sourced from Kaggle[cite: 6]. The dataset features 3,744 clinical training images displaying high noise, variable backgrounds, and structural class imbalances. 

[cite_start]To resolve the class variations (such as 767 Clubbing images vs. only 323 Healthy Nail images), an **Inverse Class Weighting** strategy was programmatically applied to the Cross-Entropy loss formulation during training loops[cite: 32, 63].

---

## ⚙️ Model Training Strategy
[cite_start]The core pipeline uses **MobileNetV3-Large** as its foundational backbone [cite: 26, 56][cite_start], optimizing for fast edge computation and potential mobile deployment[cite: 1, 127]. [cite_start]Training was executed natively in a cloud GPU environment using a two-phase routine[cite: 29, 34]:

* **Phase 1 (Head Optimization):** Pre-trained backbone parameters were locked, warming up a customized fully-connected classification head with a Dropout layer (0.4) for 15 epochs utilizing the **AdamW** optimizer at a `1e-3` learning rate.
* **Phase 2 (Full Fine-Tuning):** All layer parameters were unlocked. Differential learning rates were applied (`1e-5` for core features, `1e-4` for classification heads) for an additional 10 epochs to carefully align deep texturing filters without disrupting structural ImageNet mappings.

---

## 🚀 How to Run the App Locally / Codespaces

Follow these commands to deploy the interactive Streamlit web application dashboard:

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/Nail-Disease-Detection.git](https://github.com/YOUR_USERNAME/Nail-Disease-Detection.git)
cd Nail-Disease-Detection