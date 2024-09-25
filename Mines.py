# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (Sonar data from Kaggle)
# Replace 'sonar.csv' with the path to your actual dataset
data = pd.read_csv('sonar.csv')

# Check for missing values
print("Missing values per column:\n", data.isnull().sum())

# Handle missing values (if any)
# For example, you can drop missing values or fill them with a strategy like mean/mode
data = data.dropna()  # Alternatively, you can use data.fillna() with a specific strategy

# Separate features and target variable
X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # Last column which contains labels 'M' or 'N'

# Convert target variable into numeric (M -> 1, N -> 0)
y = y.map({'M': 1, 'N': 0})

# Split dataset into training (70%) and testing (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Data Preprocessing: Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal Object', 'Mine'], yticklabels=['Normal Object', 'Mine'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Test the model with a new sonar signal (as input)
def classify_sonar_signal(signal):
    signal_scaled = scaler.transform([signal])
    prediction = model.predict(signal_scaled)
    return 'Mine' if prediction[0] == 1 else 'Normal Object'

# Example: testing with a new sonar signal (replace with actual data)
new_signal = X_test.iloc[0, :]  # Using a sample from the test set
result = classify_sonar_signal(new_signal)
print(f"The system predicts this signal as: {result}")
