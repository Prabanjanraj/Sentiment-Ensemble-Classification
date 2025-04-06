# Sentiment-Ensemble-Classification
# Sentiment Classification using Ensemble Learning

## 📌 Problem Statement
The goal of this project is to perform sentiment classification on a given dataset using machine learning models. The focus is on improving model performance using ensemble techniques like voting and stacking.

## 🧠 Approach

We used the following models for classification:

- **Naive Bayes** (Base & Grid Search)
- **Support Vector Machine (SVM)** (Base & with Truncated SVD)
- **Ensemble Models**:
  - Voting Classifier (Hard & Soft)
  - Stacking Classifier

Dimensionality reduction was applied using **TruncatedSVD** for high-dimensional text features. Hyperparameter tuning was performed using **GridSearchCV**.

## 📊 Results

| Model                  | Accuracy |
|------------------------|----------|
| Naive Bayes (Base)     | 0.606    |
| Naive Bayes (Grid)     | 0.606    |
| SVM (Base)             | 0.674    |
| SVM (with SVD)         | 0.590    |
| Voting Classifier (Hard) | 0.642  |
| Voting Classifier (Soft) | 0.680  |
| **Stacking Classifier** | **0.694** ✅ |

> Ensemble methods like **Voting (Soft)** and **Stacking** showed better performance compared to individual models.

## 🧰 Libraries Used

- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `nltk` (for preprocessing text)
- `joblib` (for model saving)

## 📁 Project Structure

