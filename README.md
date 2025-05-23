import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('/kaggle/input/student/StudentsPerformance.csv')

# Select features and target
X = df[['reading score', 'writing score']]
y = df['math score']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print(f"R² accuracy on test set: {accuracy:.4f}")

# Predict a single input example
single_input = [[80, 75]]
predicted_math_score = model.predict(single_input)
print(f"Predicted math score for reading=80 and writing=75: {predicted_math_score[0]:.2f}")

# Visualization
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Math Scores')
plt.ylabel('Predicted Math Scores')
plt.title('Actual vs Predicted Math Scores')
plt.grid(True)
plt.show()
