import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

# 1. Load the data
df = pd.read_csv('diabetes.csv')

# 2. Look at the first 5 rows to ensure it loaded correctly
print("Data Preview:")
print(df.head())

# 3. Define our features (X) and our target (y)
# X = the input columns, y = the output (Outcome)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 4. Split the data (80% to train, 20% to test)
# This mimics a "Final Exam" for the machine
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Initialize and Train the Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 6. Test the Model
predictions = model.predict(X_test)

# 7. Print the results
accuracy = accuracy_score(y_test, predictions)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# 8. Quick Prediction on a sample (Example patient data)
sample_patient = [[1, 1, 1, 1, 0, 1, 1, 1]] // model was trained on 8 columns of data
result = model.predict(sample_patient)

if result[0] == 1:
    print("\nResult: The model predicts this patient has diabetes.")
else:
    print("\nResult: The model predicts this person doesn't have diabetes.")

# -------------------------------
# 8. EXTRA EVALUATION (by Maisha)
# -------------------------------

# Confusion Matrix shows correct and wrong predictions
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Classification Report shows precision, recall, f1-score
print("\nClassification Report:")
print(classification_report(y_test, predictions))
#Contribution by Maisha - added evaluation metrices
