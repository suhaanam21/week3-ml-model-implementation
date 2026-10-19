"""
Week 3 Task: Model Implementation and Code Documentation
A simple Decision Tree Classifier using scikit-learn's Iris dataset.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def main():
    # Load the sample dataset included with scikit-learn.
    iris = load_iris()
    X = iris.data  # Sepal and petal measurements.
    y = iris.target  # Flower species labels.

    # Keep a separate test set to evaluate unseen examples.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Limit tree depth to help control model complexity.
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=4,
        random_state=42,
    )

    # Train the classifier on the training data.
    model.fit(X_train, y_train)

    # Predict species labels for the test examples.
    y_pred = model.predict(X_test)

    # Display common evaluation metrics.
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=iris.target_names,
            zero_division=0,
        )
    )
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, y_pred))

    # Demonstrate prediction for one held-out example.
    sample = X_test[0].reshape(1, -1)
    predicted_label = model.predict(sample)[0]
    print("\nExample predicted species:", iris.target_names[predicted_label])


if __name__ == "__main__":
    main()
