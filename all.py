import pandas as pd

data = {
    'Feature1': [2, 1, 3, 5, 6, 7, 4, 2, 8, 3],
    'Feature2': [3, 5, 2, 6, 7, 8, 5, 1, 9, 4],
    'Feature3': [1, 2, 1, 4, 5, 6, 3, 2, 7, 2],
    'Class':    [0, 0, 0, 1, 1, 1, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df[['Feature1', 'Feature2', 'Feature3']]
y = df['Class']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("Decision Tree Predictions:", dt_pred)
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))

from sklearn.svm import SVC

svm = SVC()

svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)

print("\nSVM Predictions:", svm_pred)
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

print("\nKNN Predictions:", knn_pred)
print("KNN Accuracy:", accuracy_score(y_test, knn_pred))

