# 🩺 Pima Diabetes Classification using VotingClassifier

This project was created to predict whether individuals have diabetes using the [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database). Multiple machine learning models were combined using **VotingClassifier** from the ensemble methods.

---

## 📌 Project Goal

- Building a powerful ensemble model by combining Logistic Regression, Random Forest and KNN models
- Evaluating model performance with **ROC AUC score**
- Comparing ensemble model with single models

---

## 🧠 Used Algorithms

- Logistic Regression
- Random Forest Classifier
- K-Nearest Neighbors Classifier
- Voting Classifier (soft voting)

---

## 🧪 Used Libraries

- pandas, numpy
- scikit-learn
- matplotlib (for ROC curve)

---

## 🔁 Ensemble Method: VotingClassifier

```python
voting_clf = VotingClassifier(
estimators=[
("lr", LogisticRegression()),
("rf", RandomForestClassifier()),
("knn", KNeighborsClassifier())
    ],
    voting="soft"
)
