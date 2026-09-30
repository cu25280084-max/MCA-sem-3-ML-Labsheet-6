# Student Performance Prediction System
# Simple classification project

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = pd.read_csv("Project_1_Student_Performance/student_performance.csv")
print(data.head())
print(data.info())

# Remove duplicate rows
data = data.drop_duplicates()

# Create target if marks are available
if "Performance" not in data.columns:
    if "Previous_Semester_Result" in data.columns:
        data["Performance"] = data["Previous_Semester_Result"].apply(
            lambda x: "Good" if x >= 60 else "Average"
        )
    else:
        raise ValueError("Add a Performance or Previous_Semester_Result column.")

target = "Performance"

features = [
    "Attendance",
    "Internal_Marks",
    "Assignments",
    "Study_Hours",
    "Previous_Semester_Result"
]

X = data[features]
y = data[target]

# Preprocessing
numeric_features = X.columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model 1
logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

# Model 2
random_forest_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])

logistic_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)

# Predictions
logistic_pred = logistic_model.predict(X_test)
forest_pred = random_forest_model.predict(X_test)

print("Logistic Regression Accuracy:",
      accuracy_score(y_test, logistic_pred))
print("Random Forest Accuracy:",
      accuracy_score(y_test, forest_pred))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, forest_pred))

# Confusion matrix
cm = confusion_matrix(y_test, forest_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Student Performance - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save model
joblib.dump(random_forest_model, "student_performance_model.pkl")
print("Model saved.")
