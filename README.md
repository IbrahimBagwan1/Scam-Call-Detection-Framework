# 🚨 Scam Call Detection Framework (Privacy-Preserving Edge AI)

🔗 **Android Application Repository:**  
👉 https://github.com/IbrahimBagwan1/Fraudulent-Call-Detection-and-Prevention-System-for-Mobile-Devices

> This repository contains the **model training, experimentation, and research framework** for the Scam Call Detection system.  
> The **Android (Kotlin) application** that deploys these models on-device is maintained in the repository linked above.

---

## 📖 Context & Project Background

### 🚨 The Problem: The Rise of *Vishing*
Voice Phishing (Vishing) has evolved from simple robocalls into highly sophisticated, socially engineered attacks. According to the **2024 FBI Internet Crime Report**, global financial losses from phone-based scams exceeded **$16.6 Billion**, representing a **295% increase since 2020**.

Existing solutions suffer from a critical **Privacy–Latency Trade-off**:

- **Blacklist-based apps (e.g., caller ID services):**  
  Ineffective against spoofed or newly generated numbers.
- **Cloud-based AI detection:**  
  High accuracy, but requires streaming live call audio to external servers, violating user privacy and introducing significant network latency (>500 ms).
- **On-device heuristic methods:**  
  Often rely on shallow acoustic features (pitch, jitter) and fail against calm, professional scammers or AI-generated voices.

---

## 💡 Proposed Solution: Domain-Adaptive Privacy-Preserving Framework

This project introduces a **Domain-Adaptive Deep Learning Framework** that detects scam calls **entirely offline** on Android devices.

The central hypothesis is **Modality Invariance**:

> *The semantic intent of fraud (financial urgency, authority pressure, OTP extraction) remains consistent across both text (SMS) and voice (call transcripts).*

By leveraging this insight, we transfer knowledge learned from high-resource SMS datasets to the low-resource scam call domain.

---

## 🛠️ Key Technical Innovations

### 1️⃣ Domain Adaptation (Text → Voice)
- Large-scale **SMS spam datasets** are used as the **source domain** to learn fraud semantics.
- The model is fine-tuned on a curated **Composite Scam Transcript Dataset** as the **target domain**, enabling robust scam detection despite limited labeled call data.

---

### 2️⃣ Knowledge Distillation for Edge Deployment
- **Teacher Model:** `RoBERTa`  
  High accuracy but computationally expensive.
- **Student Model:** `DistilRoBERTa`  
  A compact model distilled from the teacher, achieving:
  - **~40% reduction in size**
  - **~60% faster inference**
  - **~97% of teacher performance**

---

### 3️⃣ Privacy-First Offline Architecture
- **Speech Recognition:**  
  Offline transcription using **Vosk ASR**, ensuring no audio data leaves the device.
- **Inference Engine:**  
  Quantized **INT8 TFLite** model for real-time execution.
- **Latency:**  
  ~**140 ms** inference time on standard Android hardware.
- **Zero Data Exfiltration:**  
  No cloud calls, no server dependency, no user data leakage.

---

## 📊 Dataset

To support this research, we curated the **Composite Scam Transcript Dataset (N = 46,982)** by harmonizing multiple sources:

- **TeleAntiFraud-28k:** Real-world forensic scam call transcripts.
- **Korean Phishing Logs:** Back-translated datasets capturing international scam narratives.
- **Synthetic Augmentation:** GPT-generated scenarios covering emerging fraud patterns such as:
  - Family emergency scams
  - Deepfake voice impersonation
  - Banking and OTP manipulation attacks

🔗 **Dataset Access (Kaggle):**  
https://www.kaggle.com/datasets/a186dc6d3f8d6d5169933f4153fefe9a8916fb77a33f62e31dd1a1e35d565422

---

## 📂 Repository Scope

This repository includes:
- Model training scripts
- Domain adaptation experiments
- Knowledge distillation pipeline
- Evaluation and explainability modules

❗ **Note:**  
Datasets, trained weights, and inference artifacts are intentionally excluded from version control and are provided via external hosting platforms.

---

## 📱 Android Deployment

The on-device scam detection application (Kotlin, Android) that integrates:
- Offline ASR
- Quantized TFLite inference
- Real-time call monitoring

is available here:  
🔗 https://github.com/IbrahimBagwan1/Fraudulent-Call-Detection-and-Prevention-System-for-Mobile-Devices

---

## 📌 Keywords
Scam Call Detection · Vishing · Edge AI · Privacy-Preserving ML · Domain Adaptation · Knowledge Distillation · Offline AI · Android ML


---

# 🚀 How to Use This Repository (Execution Guide)

This section provides a **clear, ordered workflow** to run the repository and obtain results from scratch.

The pipeline follows a **four-stage execution flow**:

1. SMS-based pre-training  
2. Scam call fine-tuning  
3. Model explainability analysis  
4. Model quantization for Android deployment  

Each step depends on the successful completion of the previous one.

---

## 🔄 Execution Flow Overview

```
Dataset Preparation
  ↓
SMS Pre-training
  ↓
Scam Call Fine-tuning
  ↓
Explainability Analysis
  ↓
Model Quantization (TFLite)
```

---

## 🧩 Step-by-Step Execution Instructions

### ✅ Step 0: Verify Dataset Placement

Before running any script, ensure the dataset directory is structured as follows:

```
dataset/
├── sms_dataset.csv
├── call_dataset.csv
```

- `sms_dataset.csv` → Used for SMS-based pre-training
- `call_dataset.csv` → Used for scam call fine-tuning

Do not rename these files unless you update the script paths.

---

### 🧠 Step 1: SMS Pre-training (Source Domain)

This step trains the base language model to learn general fraud semantics from SMS data.

```bash
python 1_pretrain_sms.py
```

**What this step does:**
- Loads SMS dataset
- Trains a transformer model on scam-related text
- Learns linguistic patterns such as urgency, authority, and coercion

**Expected outcome:**
- A pre-trained fraud-aware language model stored locally
- Training loss and accuracy printed in the console

⚠️ *This step may take time on CPU-only systems.*

---

### 🎯 Step 2: Scam Call Fine-tuning (Target Domain)

This step adapts the pre-trained model to voice-based scam transcripts.

```bash
python 2_finetune_calls.py
```

**What this step does:**
- Loads the SMS-pretrained model
- Fine-tunes it on call transcript data
- Specializes the model for scam call detection

**Expected outcome:**
- Fine-tuned scam call detection model
- Evaluation metrics: Accuracy, Precision, Recall, F1-score

📌 *This step is critical for transferring knowledge from text to voice.*

---

### 🔍 Step 3: Model Explainability (Interpretability)

This step provides model transparency using SHAP-based explainability.

```bash
python 3_explain_model.py
```

**What this step does:**
- Computes token-level importance scores
- Identifies linguistic patterns strongly associated with scams
- Validates that the model focuses on semantic fraud cues

**Expected outcome:**
- Explainability metrics
- Visual or numerical indicators of scam-related tokens

**Useful for:**
- Academic validation
- Model trustworthiness
- Presentation and reporting

---

### ⚡ Step 4: Model Quantization for Android Deployment

This step converts the trained model into an INT8 quantized TFLite model suitable for mobile devices.

```bash
python quantize_model.py
```

**What this step does:**
- Converts the trained transformer model to TensorFlow Lite
- Applies post-training quantization
- Optimizes the model for low-latency on-device inference

**Expected outcome:**
- A quantized `.tflite` model
- Ready for integration into the Android application

---

## 📱 Using the Model in the Android App

Once quantization is complete:

1. Copy the generated `.tflite` model (already added in the repo)
2. Place it inside the Android app's `assets/` directory
3. Load it using TensorFlow Lite Interpreter
4. Perform real-time inference on transcribed call text

🔗 Android implementation details: [Fraudulent-Call-Detection-and-Prevention-System-for-Mobile-Devices](https://github.com/IbrahimBagwan1/Fraudulent-Call-Detection-and-Prevention-System-for-Mobile-Devices)

---

## 🧪 Expected Outputs Summary

| Step | Output |
|------|--------|
| SMS Pre-training | Fraud-aware base model |
| Call Fine-tuning | Scam call detection model |
| Explainability | Token importance & semantic insights |
| Quantization | INT8 TFLite model |
```
