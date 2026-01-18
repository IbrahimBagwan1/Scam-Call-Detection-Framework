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
