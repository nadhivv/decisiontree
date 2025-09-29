import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

# === Load Dataset ===
df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Standarisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balancing
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Train Model
clf = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# === Save model & scaler ===
joblib.dump(clf, "heart_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model dan Scaler berhasil disimpan!")
