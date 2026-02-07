# 🗣️ Multi-Modal Voice Pathology Detection using Hybrid Deep Learning & Acoustic Physics

> **State-of-the-Art Classification of Laryngeal Pathologies (96% Accuracy)**
> *Combining ResNet-34 Visual Features with Source-Filter Theory (Jitter/Shimmer/HNR).*

## 📌 Project Overview

This research proposes a novel **Hybrid Intelligence Framework** for automatically detecting voice pathologies from raw audio recordings. Unlike standard Deep Learning models that treat audio spectrograms as simple images, our approach fuses **Deep Visual Representations (ResNet-34)** with **Clinical Physics Biomarkers (Jitter, Shimmer, HNR)**.

This "Doctor-in-the-Loop" architecture achieves **diagnostic-grade accuracy (95.9%)** while solving the critical "Black Box" problem of AI by validating predictions against established acoustic physics.

### 🎯 Key Achievements

| Model Architecture | Accuracy | Key Finding |
| --- | --- | --- |
| **Basic CNN (Baseline)** | 88.0% | Failed to detect structural defects (Recall: 0.18). |
| **Dual-Stream (Physics)** | 89.3% | **+227% Improvement** in Cyst detection. |
| **ResNet-34 (Deep CNN)** | 95.9% | State-of-the-Art texture recognition. |
| **Hybrid ResNet + Physics** | **96.0%** | Best of both worlds: High Accuracy + Explainability. |

---

## 🏗️ Methodology

Our framework utilizes a **Dual-Stream Architecture** that mimics how a clinician diagnoses voice disorders:

### **Stream A: The "Eye" (Visual Texture)**

* **Input:** Mel-Spectrograms (converted to 224x224 RGB images).
* **Backbone:** **ResNet-34** (Pre-trained on ImageNet).
* **Function:** Captures complex time-frequency patterns (e.g., breathiness in Laryngitis, tremors in Parkinson's).

### **Stream B: The "Brain" (Acoustic Physics)**

* **Input:** 10 Clinical Parameters extracted using `parselmouth` (Praat):
* *Frequency Perturbation:* Jitter (Local, RAP).
* *Amplitude Perturbation:* Shimmer (Local, APQ3).
* *Noise Measures:* HNR (Harmonics-to-Noise Ratio).
* *Formants:* F1, F2, F3, F4 (Vocal Tract Resonance).


* **Backbone:** Multi-Layer Perceptron (MLP).
* **Function:** Detects physical structural anomalies (e.g., Cysts, Polyps) that may not be visually obvious in a spectrogram.

### **The Fusion Layer**

* Features from Stream A (512-dim) and Stream B (128-dim) are concatenated.
* A final **Cross-Attention Classifier** makes the diagnosis based on both visual and physical evidence.

---

## 📊 Dataset & Preprocessing

* **Source:** Saarbruecken Voice Database (SVD).
* **Classes (6):**
1. **Healthy (Vox Senilis/Control)**
2. **Parkinson's Disease**
3. **Laryngitis**
4. **Dysarthia**
5. **Vocal Cysts (Structural)**
6. **Spasmodic Dysphonia**


* **Preprocessing:**
* Silence Removal & Normalization.
* Segmentation (3-second chunks).
* Data Augmentation (Time Stretch, Pitch Shift, Additive Noise).



---


### **License**

MIT License - Free for academic and research use.
