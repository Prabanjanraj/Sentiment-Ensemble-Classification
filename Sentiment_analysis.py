import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                             roc_curve, auc, RocCurveDisplay)
from sklearn.preprocessing import label_binarize
import numpy as np


# Set style for plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]

# Load data
train_df = pd.read_csv('train.csv', encoding='ISO-8859-1')
test_df = pd.read_csv('test.csv', encoding='ISO-8859-1')
train_df.columns = [col.lower() for col in train_df.columns]
test_df.columns = [col.lower() for col in test_df.columns]

# Feature & target
X = train_df['text'].dropna()
y = train_df.loc[X.index, 'sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if binary classification for ROC curve
binary_classification = len(y.unique()) == 2

# ===== Visualization Function =====
def plot_confusion_matrix(y_true, y_pred, model_name, classes=None):
    cm = confusion_matrix(y_true, y_pred)
    if classes is None:
        classes = sorted(set(y_true) | set(y_pred))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_roc_curve(y_true, y_score, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
    plt.title(f'ROC Curve - {model_name}')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.show()

def plot_metrics_comparison(metrics_dict):
    models = list(metrics_dict.keys())
    accuracies = [metrics_dict[model]['accuracy'] for model in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['skyblue', 'darkorange', 'seagreen', 'crimson'])
    
    plt.title('Model Accuracy Comparison', fontsize=16)
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom', fontsize=12)

    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("model_accuracy_comparison.png")
    plt.show()


# Dictionary to store all metrics
all_metrics = {}

# ===== Naive Bayes WITHOUT SVD =====
nb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nb', MultinomialNB())
])

nb_params = {
    'tfidf__max_df': [0.75],
    'tfidf__ngram_range': [(1, 1)],
    'nb__alpha': [1.0, 10.0]
}

print("\nTraining Naive Bayes with Grid Search...")
nb_grid = GridSearchCV(nb_pipeline, nb_params, cv=3, n_jobs=-1)
nb_grid.fit(X_train, y_train)
y_pred_nb = nb_grid.predict(X_test)

print("\n===== Naive Bayes Evaluation =====")
print("Best Params:", nb_grid.best_params_)
nb_accuracy = accuracy_score(y_test, y_pred_nb)
print("Accuracy:", nb_accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))
print("Classification Report:\n", classification_report(y_test, y_pred_nb))

# Store metrics
all_metrics['Naive Bayes (Grid)'] = {
    'accuracy': nb_accuracy,
    'report': classification_report(y_test, y_pred_nb, output_dict=True)
}

# Visualizations
plot_confusion_matrix(y_test, y_pred_nb, "Naive Bayes (Grid Search)")

if binary_classification:
    y_score_nb = nb_grid.predict_proba(X_test)[:, 1]
    plot_roc_curve(y_test, y_score_nb, "Naive Bayes (Grid Search)")

# ===== SVM WITH TruncatedSVD =====
svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svd', TruncatedSVD(n_components=100, random_state=42)),
    ('svm', LinearSVC())
])

svm_params = {
    'tfidf__max_df': [0.75],
    'tfidf__ngram_range': [(1, 1)],
    'svm__C': [1]
}

print("\nTraining SVM with Grid Search...")
svm_grid = GridSearchCV(svm_pipeline, svm_params, cv=3, n_jobs=-1)
svm_grid.fit(X_train, y_train)
y_pred_svm = svm_grid.predict(X_test)

print("\n===== SVM Evaluation (with SVD) =====")
print("Best Params:", svm_grid.best_params_)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print("Accuracy:", svm_accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))

# Store metrics
all_metrics['SVM (SVD)'] = {
    'accuracy': svm_accuracy,
    'report': classification_report(y_test, y_pred_svm, output_dict=True)
}

# Visualizations
plot_confusion_matrix(y_test, y_pred_svm, "SVM with SVD")

if binary_classification:
    y_score_svm = svm_grid.decision_function(X_test)
    plot_roc_curve(y_test, y_score_svm, "SVM with SVD")

# ===== Baseline Models =====
# Naive Bayes baseline
print("\nTraining Naive Bayes Baseline...")
nb_baseline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.75, ngram_range=(1, 1))),
    ('nb', MultinomialNB(alpha=1.0))
])
nb_baseline.fit(X_train, y_train)
y_pred_nb_base = nb_baseline.predict(X_test)

print("\n==== Naive Bayes (Baseline) ====")
nb_base_accuracy = accuracy_score(y_test, y_pred_nb_base)
print("Accuracy:", nb_base_accuracy)
print(classification_report(y_test, y_pred_nb_base))

# Store metrics
all_metrics['Naive Bayes (Base)'] = {
    'accuracy': nb_base_accuracy,
    'report': classification_report(y_test, y_pred_nb_base, output_dict=True)
}

# Visualizations
plot_confusion_matrix(y_test, y_pred_nb_base, "Naive Bayes Baseline")

if binary_classification:
    y_score_nb_base = nb_baseline.predict_proba(X_test)[:, 1]
    plot_roc_curve(y_test, y_score_nb_base, "Naive Bayes Baseline")

# SVM baseline
print("\nTraining SVM Baseline...")
svm_baseline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.75, ngram_range=(1, 1))),
    ('svm', LinearSVC(C=1))
])
svm_baseline.fit(X_train, y_train)
y_pred_svm_base = svm_baseline.predict(X_test)

print("\n==== SVM (Baseline) ====")
svm_base_accuracy = accuracy_score(y_test, y_pred_svm_base)
print("Accuracy:", svm_base_accuracy)
print(classification_report(y_test, y_pred_svm_base))

# Store metrics
all_metrics['SVM (Base)'] = {
    'accuracy': svm_base_accuracy,
    'report': classification_report(y_test, y_pred_svm_base, output_dict=True)
}

# Visualizations
plot_confusion_matrix(y_test, y_pred_svm_base, "SVM Baseline")

if binary_classification:
    y_score_svm_base = svm_baseline.decision_function(X_test)
    plot_roc_curve(y_test, y_score_svm_base, "SVM Baseline")

# ===== Final Comparison =====
plot_metrics_comparison(all_metrics)

# Print all metrics in a table
print("\n===== Model Comparison Summary =====")
comparison_df = pd.DataFrame({
    'Model': list(all_metrics.keys()),
    'Accuracy': [all_metrics[model]['accuracy'] for model in all_metrics],
    'Precision': [all_metrics[model]['report']['weighted avg']['precision'] for model in all_metrics],
    'Recall': [all_metrics[model]['report']['weighted avg']['recall'] for model in all_metrics],
    'F1-Score': [all_metrics[model]['report']['weighted avg']['f1-score'] for model in all_metrics]
})

print(comparison_df.to_markdown(index=False))


from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

# Convert text to vector using TF-IDF (needed before fitting Voting/Stacking)
tfidf_vectorizer = TfidfVectorizer(max_df=0.75, ngram_range=(1, 1))
X_train_vec = tfidf_vectorizer.fit_transform(X_train)
X_test_vec = tfidf_vectorizer.transform(X_test)

# ===== Voting Classifier (Hard Voting) =====
print("\nTraining Voting Classifier...")
voting_clf = VotingClassifier(
    estimators=[
        ('nb', MultinomialNB(alpha=1.0)),
        ('svm', LinearSVC(C=1))
    ],
    voting='hard'
)
voting_clf.fit(X_train_vec, y_train)
y_pred_voting = voting_clf.predict(X_test_vec)

voting_accuracy = accuracy_score(y_test, y_pred_voting)
print("\n==== Voting Classifier (Hard) ====")
print("Accuracy:", voting_accuracy)
print(classification_report(y_test, y_pred_voting))

all_metrics['Voting (Hard)'] = {
    'accuracy': voting_accuracy,
    'report': classification_report(y_test, y_pred_voting, output_dict=True)
}

plot_confusion_matrix(y_test, y_pred_voting, "Voting Classifier (Hard)")

from sklearn.calibration import CalibratedClassifierCV

# Calibrate the LinearSVC for probability outputs
calibrated_svm = CalibratedClassifierCV(LinearSVC(C=1), cv=3)

# Soft Voting Classifier
print("\nTraining Voting Classifier (Soft)...")
voting_clf_soft = VotingClassifier(
    estimators=[
        ('nb', MultinomialNB(alpha=1.0)),
        ('svm', calibrated_svm)
    ],
    voting='soft'
)

voting_clf_soft.fit(X_train_vec, y_train)
y_pred_voting_soft = voting_clf_soft.predict(X_test_vec)

voting_soft_accuracy = accuracy_score(y_test, y_pred_voting_soft)
print("\n==== Voting Classifier (Soft) ====")
print("Accuracy:", voting_soft_accuracy)
print(classification_report(y_test, y_pred_voting_soft))

all_metrics['Voting (Soft)'] = {
    'accuracy': voting_soft_accuracy,
    'report': classification_report(y_test, y_pred_voting_soft, output_dict=True)
}

plot_confusion_matrix(y_test, y_pred_voting_soft, "Voting Classifier (Soft)")

if binary_classification:
    y_score_voting_soft = voting_clf_soft.predict_proba(X_test_vec)[:, 1]
    plot_roc_curve(y_test, y_score_voting_soft, "Voting Classifier (Soft)")



# ===== Stacking Classifier =====
print("\nTraining Stacking Classifier...")
stacking_clf = StackingClassifier(
    estimators=[
        ('nb', MultinomialNB(alpha=1.0)),
        ('svm', LinearSVC(C=1))
    ],
    final_estimator=LogisticRegression(),
    passthrough=True
)
stacking_clf.fit(X_train_vec, y_train)
y_pred_stack = stacking_clf.predict(X_test_vec)

stacking_accuracy = accuracy_score(y_test, y_pred_stack)
print("\n==== Stacking Classifier ====")
print("Accuracy:", stacking_accuracy)
print(classification_report(y_test, y_pred_stack))

all_metrics['Stacking'] = {
    'accuracy': stacking_accuracy,
    'report': classification_report(y_test, y_pred_stack, output_dict=True)
}

plot_confusion_matrix(y_test, y_pred_stack, "Stacking Classifier")

# ===== Final Comparison (Updated) =====
plot_metrics_comparison(all_metrics)

print("\n===== Model Comparison Summary (Updated) =====")
comparison_df = pd.DataFrame({
    'Model': list(all_metrics.keys()),
    'Accuracy': [all_metrics[model]['accuracy'] for model in all_metrics],
    'Precision': [all_metrics[model]['report']['weighted avg']['precision'] for model in all_metrics],
    'Recall': [all_metrics[model]['report']['weighted avg']['recall'] for model in all_metrics],
    'F1-Score': [all_metrics[model]['report']['weighted avg']['f1-score'] for model in all_metrics]
})
print(comparison_df.to_markdown(index=False))