import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib   # <=== tambahan
import streamlit as st

# === Load Data ===
df = pd.read_csv("heart.csv")
print(df.head(10))

x = df.drop("target", axis=1)
y = df['target']

# === Preprocessing ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

print("\nDistribusi Data Awal:")
print(y.value_counts())

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

print("\nDistribusi Data Setelah SMOTE:")
print(pd.Series(y_resampled).value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

print(f"\nUkuran data -> Train: {len(X_train)}, Test: {len(X_test)}")

# === Train Model ===
clf = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4,
    random_state=42
)
clf.fit(X_train, y_train)

# === Evaluasi ===
y_pred = clf.predict(X_test)
acc = round(accuracy_score(y_test, y_pred)*100, 2)

print("Akurasi Model: ", acc, "%")
print("Klasifikasi: ")
print(classification_report(y_test, y_pred, target_names=["Tidak Sakit", "Sakit"]))

# === Visualisasi Pohon ===
plt.figure(figsize=(20,10))
plot_tree(
    clf, 
    filled=True, 
    feature_names=x.columns,
    class_names=["Tidak Sakit", "Sakit"], 
    fontsize=10
)
plt.title("Decision Tree dari Data Heart Disease")
plt.show()

# === Simpan Model dan Scaler ===
joblib.dump(clf, "heart_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model dan Scaler berhasil disimpan sebagai 'heart_model.pkl' & 'scaler.pkl'")
