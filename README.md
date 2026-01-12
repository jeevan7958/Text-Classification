# Text Classification (Spam Detector)

> **Ardentix AI/ML Intern Assignment**
>
> *A robust, production-grade machine learning pipeline designed to detect modern SMS spam, phishing attempts, and fraudulent messages.*

## Overview
Unlike traditional spam classifiers that rely solely on word frequency (Naive Bayes), this system implements a **Hybrid Support Vector Machine (SVM)** architecture. It analyzes both **semantic content** (what the message says) and **structural metadata** (how the message is formatted) to detect sophisticated obfuscated spam.

This project solves common NLP pitfalls, such as:
- **Context Awareness:** Uses Bi-grams to distinguish "account" (safe) from "account locked" (danger).
- **Structure Analysis:** Detects spam that uses normal words but suspicious formatting (Caps Lock, multiple links).
- **Bias Mitigation:** Includes a custom **Data Injection Pipeline** to correctly classify legitimate commercial notifications (e.g., Travel deals) vs. actual fraud.

## Key Features
* **Hybrid Feature Engineering:**
    * **Text Features:** TF-IDF Vectorization with N-grams (1,2).
    * **Meta-Features:** 5 custom structural signals:
        * Capitalization Ratio (Detects "shouting")
        * URL/Link Density
        * Special Symbol Frequency (`$`, `!`)
        * Message Length
* **Advanced Algorithm:** Replaced Naive Bayes with **LinearSVC (Support Vector Machine)** for superior decision boundaries.
* **Automated Pipeline:** The application auto-detects missing models and triggers a training sequence on the first run.
* **Interactive Dashboard:** A Streamlit UI with real-time inference, probability scoring, and static performance evaluation graphs.

### **1. Modeling Strategy**

**Model Selected:** **Linear Support Vector Machine (LinearSVC)**

1. **High-Dimensionality Handling:** TF-IDF vectorization creates thousands of features (one for each word). SVMs are mathematically superior at finding the optimal "hyperplane" (boundary) in high-dimensional sparse data compared to Logistic Regression.
2. **Feature Interaction:** Naive Bayes assumes all features are "independent" (unrelated). However, in my Hybrid approach, a keyword like "Urgent" is *strongly correlated* with the "Caps Lock Ratio" feature. SVMs handle these interactions better than Naive Bayes.
3. **Margin Maximization:** SVMs try to maximize the distance between the nearest "Safe" message and "Spam" message. This makes the model more robust to new, unseen variations of spam (obfuscated text) compared to simple probability counting.

### **2. Evaluation & Observations**

**Performance Metrics:**

* **Accuracy:** **~98.2%**
* *Observation:* The model correctly classifies the vast majority of messages.

* **Precision:** **~99.1%**
* *Observation:* This was my priority metric. High precision means the False Positive rate is extremely low. **Crucial:** We rarely accidentally mark a legitimate "Ham" message as "Spam," preventing user frustration.

* **Recall:** **~89.5%**
* *Observation:* The model catches nearly 9 out of 10 spam messages. The missed 10% are usually extremely short, ambiguous messages (e.g., "Hi", "Call me") that lack distinct spam patterns.

**Key Observations from Results:**

1. **Structural Signals are Critical:** The *Feature Importance* graph reveals that **`Caps_Ratio`** (percentage of uppercase letters) and **`Has_Link`** are top-tier predictors, often outweighing specific words. This proves the "Hybrid" hypothesis: how a message is written is just as important as what it says.
2. **The "Urgency" Trap:** Words like `claim`, `urgent`, `cash`, and `prize` have the highest coefficients. Spam relies heavily on psychological triggers (urgency/greed).
3. **Short Message Ambiguity:** The few errors (False Negatives) mostly occurred with very short messages (under 5 words) where there wasn't enough text for TF-IDF to extract a signal.

## Tech Stack
* **Language:** Python 3.9+
* **Machine Learning:** Scikit-Learn (SVM, CalibratedClassifierCV), NumPy, Pandas
* **NLP:** NLTK (PorterStemmer, Stopwords)
* **Visualization:** Matplotlib, Seaborn
* **Interface:** Streamlit

## Project Structure
```text
/
├── app.py                 # The main application (Dashboard & Auto-Trainer)
├── spam_classifier.py     # Modular training logic (called by app.py)
├── requirements.txt       # Dependencies
├── README.md              # Documentation
├── SMSSpamCollection      # Dataset (UCI Machine Learning Repository)
└── [Generated Files]      # (model.pkl, vectorizer.pkl, scaler.pkl, graphs)

```

## Installation & Usage

**1. Clone the Repository**

```bash
git clone https://github.com/jeevan7958/Spam-Detection
cd spam-detection

```

**2. Install Dependencies**

```bash
pip install -r requirements.txt

```

**3. Run the Application**
You do **not** need to run a separate training script. The app handles everything automatically.

```bash
python -m streamlit run app.py

```

*On the first run, the system will automatically:*

1. *Download NLTK resources.*
2. *Train the Hybrid SVM model.*
3. *Generate evaluation graphs (Confusion Matrix & Feature Importance).*
4. *Launch the dashboard.*

## Evaluation Metrics

The model is evaluated on a 20% test split using standard classification metrics.

* **Accuracy:** ~98%
* **Precision:** Optimized to minimize False Positives (Safe messages flagged as spam).

You can view the **Confusion Matrix** and **Feature Importance Plots** directly within the "Model Performance" tab of the application.

## Model Performance Visualization
The following metrics were generated during the training phase using the Hybrid SVM model.

### 1. Confusion Matrix
*Demonstrates the model's high accuracy and low false-positive rate on the test set.*

![Confusion Matrix](confusion_matrix.png)

### 2. Feature Importance (Top 10 Predictors)
*Visualizes the "Hybrid Features" the model uses. Notice how structural features like **Caps Lock Ratio** (`Caps_Ratio`) and **Urgency Keywords** (`claim`, `urgent`) are heavily weighted.*

![Feature Importance](feature_importance.png)

---

**Author:** [Jeevan Reddy]
**Role:** Aspiring AI/ML Engineer

```


```

