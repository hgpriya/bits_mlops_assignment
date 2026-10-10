# Model comparison (auto-generated)

Selected model: **logistic_regression** (highest mean 5-fold CV ROC-AUC)

| model | cv_accuracy_mean | cv_accuracy_std | cv_precision_mean | cv_precision_std | cv_recall_mean | cv_recall_std | cv_f1_mean | cv_f1_std | cv_roc_auc_mean | cv_roc_auc_std | test_accuracy | test_precision | test_recall | test_f1 | test_roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| logistic_regression | 0.847 | 0.022 | 0.879 | 0.071 | 0.783 | 0.062 | 0.824 | 0.025 | 0.907 | 0.019 | 0.885 | 0.839 | 0.929 | 0.881 | 0.959 |
| random_forest | 0.814 | 0.023 | 0.833 | 0.03 | 0.747 | 0.086 | 0.784 | 0.04 | 0.897 | 0.024 | 0.885 | 0.839 | 0.929 | 0.881 | 0.95 |
| gradient_boosting | 0.826 | 0.039 | 0.831 | 0.075 | 0.792 | 0.075 | 0.806 | 0.043 | 0.876 | 0.026 | 0.902 | 0.867 | 0.929 | 0.897 | 0.955 |

## Best hyper-parameters
* **logistic_regression**: `{'C': 0.3, 'class_weight': 'None'}`
* **random_forest**: `{'max_depth': 'None', 'max_features': 'sqrt', 'min_samples_leaf': 3, 'n_estimators': 200}`
* **gradient_boosting**: `{'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 100, 'subsample': 0.8}`