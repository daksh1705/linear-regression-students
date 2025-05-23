# Multiple Linear Regression on StudentsPerformance Dataset

This project demonstrates a simple **Multiple Linear Regression** model to predict students' **Math scores** based on their **Reading** and **Writing scores** using the `StudentsPerformance.csv` dataset.

---

## Dataset

The dataset contains students' scores in three subjects:
- Math Score (Target)
- Reading Score (Feature)
- Writing Score (Feature)

You can find the dataset [here](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams).

---

## Project Overview

- Load the dataset using `pandas`.
- Use **Reading** and **Writing scores** as input features to predict the **Math score**.
- Split the dataset into training and testing sets (80% training, 20% testing).
- Train a Multiple Linear Regression model using `scikit-learn`.
- Evaluate the model using the R² (coefficient of determination) score.
- Visualize the actual vs predicted math scores using a scatter plot.
- Predict math score for a single input example.

---

## How to Run

1. Clone the repo or download the notebook/script.
2. Ensure you have the required libraries installed:

```bash
pip install pandas scikit-learn matplotlib
