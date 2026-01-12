import pandas as pd
import numpy as np
import re
import nltk
import pickle
import os
import json  # <--- NEW: To save metrics
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack

plt.switch_backend('Agg') 

def train_and_evaluate():
    print("--- STARTING AUTOMATED TRAINING ---")
    nltk.download('stopwords', quiet=True)
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        if not isinstance(text, str): return ""
        text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
        words = text.split()
        words = [ps.stem(w) for w in words if w not in stop_words]
        return ' '.join(words)

    def get_meta_features(text):
        if not isinstance(text, str): return [0, 0, 0, 0, 0]
        return [
            len(text),
            1 if re.search(r'http[s]?://|www\.', text) else 0,
            sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0,
            text.count('!'),
            1 if re.search(r'[$£€]', text) else 0
        ]

    if not os.path.exists('SMSSpamCollection'):
        raise FileNotFoundError("Dataset 'SMSSpamCollection' missing.")

    df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'text'])
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    modern_spam = [
        "URGENT! VERIFY YOUR ACCOUNT.", "YOU WON A TRIP TO THE BAHAMAS!",
        "urgent! verify your account.", "get 50% off for a family of 2 to bahamas.",
        "Nothing beats a Jet2Holiday! Book now!", "Right now, get 50% off for a family of 4!",
        "Limited time offer! Book your holiday now!!!"
    ]
    injection = pd.DataFrame([{'label': 1, 'text': t} for t in modern_spam] * 50)
    df = pd.concat([df, injection], ignore_index=True)

    df['cleaned_text'] = df['text'].apply(clean_text)
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_text = vectorizer.fit_transform(df['cleaned_text'])

    raw_meta = np.array(df['text'].apply(get_meta_features).tolist())
    scaler = MinMaxScaler()
    X_meta = scaler.fit_transform(raw_meta)

    X_final = hstack([X_text, X_meta])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

    # TRAINING
    svc_graph = LinearSVC(class_weight='balanced', dual='auto', max_iter=3000)
    svc_graph.fit(X_train, y_train)

    base_svc = LinearSVC(class_weight='balanced', dual='auto', max_iter=3000)
    calibrated_model = CalibratedClassifierCV(base_svc) 
    calibrated_model.fit(X_train, y_train)

    #CALCULATE METRICS
    y_pred = calibrated_model.predict(X_test)
    
    # Calculate Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save metrics to JSON
    metrics_data = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics_data, f)
    
    print(f"   - Metrics Saved: Acc={accuracy:.2f}, Prec={precision:.2f}")

    #GENERATE GRAPHS
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Safe', 'Spam'], yticklabels=['Safe', 'Spam'])
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

    feature_names = vectorizer.get_feature_names_out().tolist() + ['Length', 'Has_Link', 'Caps_Ratio', 'Exclamations', 'Currency']
    coefs = svc_graph.coef_.flatten()
    top_indices = coefs.argsort()[-10:]
    top_features = [feature_names[i] for i in top_indices]
    top_scores = coefs[top_indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_scores, y=top_features, palette='viridis')
    plt.title('Top 10 Spam Predictors')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

    with open('model.pkl', 'wb') as f: pickle.dump(calibrated_model, f)
    with open('vectorizer.pkl', 'wb') as f: pickle.dump(vectorizer, f)
    with open('scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
    
    print("--- TRAINING COMPLETE ---")

if __name__ == "__main__":
    train_and_evaluate()