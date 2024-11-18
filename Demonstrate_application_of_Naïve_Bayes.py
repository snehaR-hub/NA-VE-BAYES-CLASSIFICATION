'''Demonstrating the Application of Naive Bayes Using Python Naive Bayes classifiers are a family of probabilistic classifiers based on applying Bayes' Theorem with strong (naive) independence assumptions between the features. In this demonstration, we will use the scikit-learn library to apply Naive Bayes to a classification task using the Iris dataset, a well-known dataset in machine learning.'''

'''1. Import Required Libraries
We will need the following libraries:

scikit-learn for machine learning algorithms and datasets.
pandas for data manipulation .
matplotlib for visualization .'''

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt

'''2. Load the Dataset
We will use the Iris dataset, which is available in scikit-learn. The Iris dataset consists of 150 samples of iris flowers, with four features: sepal length, sepal width, petal length, and petal width. The target variable is the species of the flower, with three possible classes: Setosa, Versicolor, and Virginica.'''

# Load the Iris dataset
data = load_iris()

# Create a pandas DataFrame for easy visualization and analysis
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="Species")

# Display the first few rows of the dataset
print(X.head())
print("\nTarget labels (y):\n", y.head())

'''3. Split the Dataset into Training and Testing Sets
To evaluate the model, we need to split the data into training and testing sets. This helps us assess how well the model generalizes to unseen data.'''

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shape of the training and testing sets
print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")

'''4. Train the Naive Bayes Model
We will use Gaussian Naive Bayes (GaussianNB) from scikit-learn, which is suitable for continuous features that follow a normal distribution . '''

# Initialize the Gaussian Naive Bayes model
nb_model = GaussianNB()

# Train the model using the training set
nb_model.fit(X_train, y_train)

# Print model parameters (means and variances for each class)
print("\nModel Parameters:")
print("Class priors:", nb_model.class_prior_)
print("Class means:\n", nb_model.theta_)
print("Class variances:\n", nb_model.sigma_)

'''5. Make Predictions
Once the model is trained, we can make predictions on the testing set.'''

# Make predictions on the testing set
y_pred = nb_model.predict(X_test)

# Display the first few predictions
print("\nPredictions on the test set:", y_pred[:10])

