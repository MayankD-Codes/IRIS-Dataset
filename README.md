🌸 Iris Dataset Classification

This repository explores the famous Iris dataset, one of the most well-known datasets in machine learning. It includes the dataset files (iris.data, iris.names, bezdekIris.data) and demonstrates how to analyze and classify different species of Iris flowers using machine learning techniques.

📊 Dataset: Iris

Total samples: 150

Classes: 3 species of Iris

Iris-setosa

Iris-versicolor

Iris-virginica

Features (all measured in cm):

Sepal length

Sepal width

Petal length

Petal width

The dataset is already split into these 150 labeled samples (no separate train/test split provided).

🚀 Workflow

Load the Dataset

import pandas as pd

df = pd.read_csv("iris.data", header=None,
                 names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
print(df.head())


Data Visualization

Histograms of each feature

Pair plots (Sepal vs Petal)

Example:


Model Building

Algorithms commonly used:

Logistic Regression

Decision Tree

Random Forest

Support Vector Machine (SVM)

Example classification with Scikit-learn:

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))


Accuracy usually reaches 95–100% 🎯

Evaluation

Confusion Matrix

Accuracy, Precision, Recall, F1-score

Example output:


🛠️ Requirements

Python 3.8+

Pandas

Scikit-learn

Matplotlib / Seaborn

Install dependencies:

pip install pandas scikit-learn matplotlib seaborn

📂 Files in this Repo

iris.data → Main dataset (CSV format).

iris.names → Metadata about dataset features and classes.

bezdekIris.data → Alternative version of the dataset.

Index → Supporting file.

🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss improvements.

📜 License

This project is licensed under the MIT License.
