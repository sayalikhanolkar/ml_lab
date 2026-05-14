import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=200,
    n_features=3,
    n_redundant=0
)

df = pd.DataFrame(X, columns=['Feature1', 'Feature2', 'Feature3'])

df['Class'] = y

X = df[['Feature1', 'Feature2', 'Feature3']]
y = df['Class']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

print("Predictions:", dt_pred)

print("\nAccuracy:")
print(accuracy_score(y_test, dt_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, dt_pred))

print("\nClassification Report:")
print(classification_report(y_test, dt_pred))

plt.figure(figsize=(8,6))

plot_tree(
    dt,
    feature_names=['Feature1', 'Feature2', 'Feature3'],
    class_names=['0', '1'],
    filled=True
)

plt.show()