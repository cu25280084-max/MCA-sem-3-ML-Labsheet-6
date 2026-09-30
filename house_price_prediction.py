# House Price Prediction System

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

file_path = os.path.join(
    os.path.dirname(__file__),
    "house_prices.csv"
)

data = pd.read_csv(file_path)

print("First 5 rows:")
print(data.head())

print("\nDataset Information:")
print(data.info())

print("\nStatistical Summary:")
print(data.describe())


# --------------------------------------------------
# 2. Data Preprocessing
# --------------------------------------------------

# Remove duplicate rows
data = data.drop_duplicates()

print("\nMissing Values:")
print(data.isnull().sum())


# Target variable
target = "Price"

# Input features
features = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Parking",
    "Age",
    "Location"
]

X = data[features]
y = data[target]


# Numerical and categorical columns
numeric_features = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Parking",
    "Age"
]

categorical_features = [
    "Location"
]


# Preprocessing for numerical data
numeric_processor = SimpleImputer(
    strategy="median"
)

# Preprocessing for categorical data
categorical_processor = Pipeline([
    ("imputer", SimpleImputer(
        strategy="most_frequent"
    )),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore"
    ))
])


# Combine preprocessing
preprocessor = ColumnTransformer([
    ("numeric", numeric_processor, numeric_features),
    ("categorical", categorical_processor, categorical_features)
])


# --------------------------------------------------
# 3. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# --------------------------------------------------
# 4. Linear Regression
# --------------------------------------------------

linear_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

linear_model.fit(X_train, y_train)

linear_prediction = linear_model.predict(X_test)


# --------------------------------------------------
# 5. Random Forest Regression
# --------------------------------------------------

random_forest_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

random_forest_model.fit(X_train, y_train)

forest_prediction = random_forest_model.predict(X_test)


# --------------------------------------------------
# 6. Model Evaluation
# --------------------------------------------------

def evaluate_model(model_name, actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    mse = mean_squared_error(
        actual,
        predicted
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        actual,
        predicted
    )

    print("\n" + model_name)
    print("-------------------------")
    print("MAE :", mae)
    print("MSE :", mse)
    print("RMSE:", rmse)
    print("R2  :", r2)


evaluate_model(
    "Linear Regression",
    y_test,
    linear_prediction
)

evaluate_model(
    "Random Forest",
    y_test,
    forest_prediction
)


# --------------------------------------------------
# 7. Actual vs Predicted Prices
# --------------------------------------------------

plt.figure(figsize=(7, 5))

plt.scatter(
    y_test,
    forest_prediction
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")

plt.show()


# --------------------------------------------------
# 8. House Price Distribution
# --------------------------------------------------

plt.figure(figsize=(7, 5))

sns.histplot(
    data["Price"],
    kde=True
)

plt.xlabel("House Price")
plt.ylabel("Number of Houses")
plt.title("House Price Distribution")

plt.show()


# --------------------------------------------------
# 9. Save Trained Model
# --------------------------------------------------

model_path = os.path.join(
    os.path.dirname(__file__),
    "house_price_model.pkl"
)

joblib.dump(
    random_forest_model,
    model_path
)

print("\nRandom Forest model saved successfully.")
