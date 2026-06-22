# 💅 Automated Nail Pathology Classification Using Deep Learning

An interactive medical computer vision system designed to classify various localized health anomalies and systemic indicators directly from raw clinical images of fingernails. 

## 🎯 Project Overview & Clinical Problem
Computer-aided diagnostics heavily prioritize broad cutaneous skin lesions, leaving onychopathology (nail diagnostics) as a highly specialized niche. Visible changes in nail morphology, texture, and coloration frequently serve as vital diagnostic indicators for localized fungal issues or severe underlying systemic conditions (such as digital clubbing linked to cardiovascular diseases and lung complications). 

This project solves this diagnostic challenge by utilizing transfer learning on top of a deep, pre-trained convolutional neural network to automatically screen nail images across 6 diagnostic target classes with **93.41% peak validation accuracy**.

### Target Diagnostic Classes
* **Acral Lentiginous Melanoma** (High-risk, critical oncology class)
* **Healthy Nail** (Baseline healthy reference)
* **Onychogryphosis** (Abnormal nail thickening and curving)
* **Blue Finger** (Cyanosis indicating potential circulatory issues)
* **Clubbing** (Nail curvature linked to systemic cardiovascular diseases)
* **Pitting** (Small depressions commonly linked to psoriasis or eczema)

---

## 📊 Dataset Sourcing
The model was trained on **Nikhil Gurav's Specialized Nail Disease Detection Dataset** sourced from Kaggle. The dataset features 3,744 clinical training images displaying high noise, variable lighting backgrounds, and structural class imbalances. 

To resolve the class variations (such as 767 Clubbing images vs. only 323 Healthy Nail images), an **Inverse Class Weighting** strategy was programmatically applied to the Cross-Entropy loss formulation during training loops to prevent model bias.

---

## ⚙️ Model Training Strategy
The core pipeline uses **MobileNetV3-Large** as its foundational backbone, optimizing for fast edge computation and potential mobile app deployment. Training was executed natively in a cloud GPU environment using a robust two-phase training routine:

* **Phase 1 (Head Optimization):** Pre-trained backbone parameters were locked, warming up a customized fully-connected classification head with a Dropout layer (0.4) for 15 epochs utilizing the **AdamW** optimizer at a `1e-3` learning rate.
* **Phase 2 (Full Fine-Tuning):** All layer parameters were unlocked. Differential learning rates were applied (`1e-5` for core features, `1e-4` for classification heads) for an additional 10 epochs to carefully align deep texturing filters without disrupting foundational ImageNet weights.

---

## 🚀 How to Run the App Locally / Codespaces

Follow these commands to deploy the interactive Streamlit web application dashboard:

### 1. Clone the Repository
```bash
git clone [https://github.com/MuhammadOmama/Nail-Disease-Detection.git](https://github.com/MuhammadOmama/Nail-Disease-Detection.git)
cd Nail-Disease-Detection