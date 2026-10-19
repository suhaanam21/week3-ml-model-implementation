# Week 3: Model Implementation and Code Documentation

## Project overview
This project demonstrates a supervised machine-learning classification workflow using Python and a Decision Tree Classifier. It uses the Iris dataset bundled with scikit-learn, so a separate dataset download is not needed.

## Problem statement
Predict an iris flower's species from four measurements: sepal length, sepal width, petal length, and petal width.

## Algorithm
A Decision Tree Classifier learns feature-based decision rules from labelled examples. The model is trained on a training subset and evaluated on a separate test subset.

## Files
- `model.py` — documented Python implementation and evaluation.
- `requirements.txt` — Python dependency.
- `.gitignore` — excludes common temporary files.

## Run the project
1. Install Python 3.10 or later.
2. Open a terminal in this folder.
3. Install the dependency:
   ```bash
   python -m pip install -r requirements.txt
   ```
4. Run:
   ```bash
   python model.py
   ```

The program prints test accuracy, precision/recall/F1 metrics, a confusion matrix, and an example prediction. Results are generated when the code runs.

## Methodology
1. Load the Iris dataset.
2. Separate features (`X`) from target labels (`y`).
3. Split the data into training and test sets using stratification.
4. Train a Decision Tree Classifier.
5. Predict labels for the test set.
6. Evaluate using accuracy, a classification report, and a confusion matrix.

## Limitations
Iris is a small educational dataset. Results are for demonstration and do not guarantee performance on other datasets or real-world applications.

## References
- Scikit-learn Decision Trees: https://scikit-learn.org/stable/modules/tree.html
- Scikit-learn Iris dataset: https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
- Python documentation: https://docs.python.org/3/
