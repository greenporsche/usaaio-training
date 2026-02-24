# Problem 13 Variations: End-to-End ML (COMPREHENSIVE)

> Original: Breast Cancer classification using kNN with macro-F1 evaluation
> Core Skills: Data preprocessing, feature engineering, model pipelines, hyperparameter tuning, scikit-learn

---

## CATEGORY A: Different Datasets (Same Constraints)

### Variation A1: Wine Quality Classification

You are given the **Wine Quality dataset**, which is a multi-class classification task. All features are numeric.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

# Load dataset
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

X = df.drop("target", axis=1)
y = df["target"]
```

This dataset has 3 classes representing different wine cultivars.

### Submission Requirements

Submit a single Jupyter notebook containing:
1. Data preprocessing (if needed)
2. Model construction
3. Model training
4. Inference logic

### Inference Requirements

```python
def my_prediction(X_test):
    ###INSERT YOUR CODE HERE###
    return y_pred
```

Your predictions will be evaluated using **macro-averaged F1 score (F1-macro)**.

### Model Constraints

You must use **k-Nearest Neighbors (kNN)** as part of your classification approach.

<details>
<summary>Solution A1</summary>

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report

# Load data
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=2026, stratify=y
)

# Create pipeline with preprocessing and kNN
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# Hyperparameter grid
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9, 11, 15],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan', 'minkowski']
}

# Grid search with F1-macro scoring
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV F1-macro: {grid_search.best_score_:.4f}")

# Validate
y_val_pred = grid_search.predict(X_val)
val_f1 = f1_score(y_val, y_val_pred, average='macro')
print(f"Validation F1-macro: {val_f1:.4f}")

# Final model
best_model = grid_search.best_estimator_
best_model.fit(X, y)  # Retrain on all data

def my_prediction(X_test):
    y_pred = best_model.predict(X_test)
    return pd.Series(y_pred, name='target')
```

**Key Insights**:
- Wine dataset has 13 features with different scales → StandardScaler essential
- 3-class problem → stratified split important
- kNN with distance weighting often works better for this dataset
- Best parameters typically: k=5-7, distance weighting, euclidean metric

</details>

### Variation A2: Iris Classification with Noise

The **Iris dataset** with added Gaussian noise:

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

np.random.seed(2026)
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
# Add noise
X = X + np.random.normal(0, 0.5, X.shape)
y = pd.Series(data.target, name='target')
```

### Model Constraints

Use **kNN** with proper handling of the noisy features.

<details>
<summary>Solution A2</summary>

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score

# Load noisy data
np.random.seed(2026)
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
X = X + np.random.normal(0, 0.5, X.shape)
y = pd.Series(data.target, name='target')

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=2026, stratify=y
)

# Pipeline with feature selection (helps with noise)
pipeline = Pipeline([
    ('scaler', RobustScaler()),  # Robust to outliers from noise
    ('feature_selection', SelectKBest(f_classif)),
    ('knn', KNeighborsClassifier())
])

param_grid = {
    'feature_selection__k': [2, 3, 4],  # Test different feature counts
    'knn__n_neighbors': [5, 7, 9, 11],
    'knn__weights': ['uniform', 'distance'],
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")

# Validate
y_val_pred = grid_search.predict(X_val)
print(f"Validation F1-macro: {f1_score(y_val, y_val_pred, average='macro'):.4f}")

# Final model
best_model = grid_search.best_estimator_
best_model.fit(X, y)

def my_prediction(X_test):
    return pd.Series(best_model.predict(X_test), name='target')
```

**Key Insights**:
- Noise in features can degrade kNN performance
- RobustScaler better than StandardScaler for noisy data
- Feature selection (SelectKBest) can filter out noise
- Higher k values provide robustness against noise

</details>

### Variation A3: Digits Classification (Subset)

Use a subset of the **Digits dataset** (0-4 only):

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits

data = load_digits()
# Filter to digits 0-4
mask = data.target < 5
X = pd.DataFrame(data.data[mask])
y = pd.Series(data.target[mask], name='target')
```

This is a 5-class image classification problem with 64 features (8×8 pixels).

<details>
<summary>Solution A3</summary>

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

# Load filtered data
data = load_digits()
mask = data.target < 5
X = pd.DataFrame(data.data[mask])
y = pd.Series(data.target[mask], name='target')

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=2026, stratify=y
)

# PCA reduces dimensionality and noise for image data
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),  # Pixels already 0-16, normalize to 0-1
    ('pca', PCA()),
    ('knn', KNeighborsClassifier())
])

param_grid = {
    'pca__n_components': [10, 20, 30, 40],
    'knn__n_neighbors': [3, 5, 7, 9],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'cosine']
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")

y_val_pred = grid_search.predict(X_val)
print(f"Validation F1-macro: {f1_score(y_val, y_val_pred, average='macro'):.4f}")

best_model = grid_search.best_estimator_
best_model.fit(X, y)

def my_prediction(X_test):
    return pd.Series(best_model.predict(X_test), name='target')
```

**Key Insights**:
- Image data benefits from PCA for kNN (dimensionality reduction)
- ~20-30 PCA components typically capture most variance
- Cosine distance can work well for normalized pixel intensities
- MinMaxScaler appropriate since pixel values are bounded

</details>

---

## CATEGORY B: Different Evaluation Metrics

### Variation B1: Accuracy as Metric

Same Breast Cancer setup, but evaluated on **Accuracy**:

```python
# Dataset loading same as original Problem 13
# ...

# Your predictions will be evaluated using **Accuracy**.
```

<details>
<summary>Solution B1</summary>

```python
# Main difference: Change scoring metric
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',  # Changed from 'f1_macro'
    n_jobs=-1
)

# Key insight: For balanced datasets, accuracy ≈ F1-macro
# For imbalanced datasets, accuracy can be misleading
# Breast cancer is moderately balanced, so similar results expected
```

**When accuracy differs from F1-macro**:
- Imbalanced classes: High accuracy by predicting majority class
- F1-macro weights all classes equally regardless of size
- For this dataset, both metrics usually select similar models

</details>

### Variation B2: Recall-Focused (Medical Context)

In medical diagnosis, **missing a positive case (cancer) is more costly** than a false alarm.

Your predictions will be evaluated using **Recall on the positive class** (class 1 = malignant).

### Additional Constraint

You must achieve **at least 95% recall** on the positive class.

<details>
<summary>Solution B2</summary>

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, recall_score

# Custom scorer for positive class recall
positive_recall = make_scorer(recall_score, pos_label=1)

# Load data (same as original)
# X, y = ...

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=2026, stratify=y
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# Focus on higher k values (smoother decision boundary → higher recall)
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 11, 15, 21],
    'knn__weights': ['distance'],  # Distance weighting helps
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring=positive_recall,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Post-hoc threshold adjustment for higher recall
from sklearn.calibration import CalibratedClassifierCV

# Alternative approach: Use probability thresholding
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Get probabilities on validation set
y_val_proba = best_model.predict_proba(X_val)[:, 1]

# Find threshold that achieves 95% recall
from sklearn.metrics import recall_score
for thresh in np.arange(0.1, 0.9, 0.05):
    y_pred = (y_val_proba >= thresh).astype(int)
    recall = recall_score(y_val, y_pred, pos_label=1)
    if recall >= 0.95:
        optimal_threshold = thresh
        print(f"Threshold {thresh:.2f}: Recall = {recall:.3f}")
        break

def my_prediction(X_test):
    proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= optimal_threshold).astype(int)
    return pd.Series(y_pred, name='target')
```

**Key Insights**:
- Medical contexts often require high recall (sensitivity)
- Trade-off: Higher recall → lower precision (more false positives)
- Probability thresholding allows fine-grained control
- kNN with distance weighting provides probability estimates

</details>

### Variation B3: ROC-AUC Evaluation

Your predictions will be evaluated using **ROC-AUC score**.

Return **probabilities** instead of class labels:

```python
def my_prediction(X_test):
    # Return probability of positive class
    return y_proba  # Values between 0 and 1
```

<details>
<summary>Solution B3</summary>

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=2026, stratify=y
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

param_grid = {
    'knn__n_neighbors': [5, 7, 11, 15, 21, 31],
    'knn__weights': ['distance'],  # Better probability estimates
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

# Validate
y_val_proba = grid_search.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_val_proba)
print(f"Validation ROC-AUC: {val_auc:.4f}")

best_model = grid_search.best_estimator_
best_model.fit(X, y)

def my_prediction(X_test):
    y_proba = best_model.predict_proba(X_test)[:, 1]
    return pd.Series(y_proba, name='probability')
```

**Key Insights**:
- ROC-AUC evaluates ranking ability, not calibration
- Higher k values give smoother probability estimates
- Distance weighting improves probability quality
- kNN probabilities = proportion of k neighbors in each class

</details>

### Variation B4: Weighted F1 Score

For imbalanced classes, evaluated on **Weighted F1 score** (weights by class support).

<details>
<summary>Solution B4</summary>

```python
from sklearn.metrics import f1_score, make_scorer

# Weighted F1 scorer
weighted_f1 = make_scorer(f1_score, average='weighted')

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring=weighted_f1,
    n_jobs=-1
)

# Weighted F1 vs Macro F1:
# - Weighted: Each class contributes proportionally to its sample size
# - Macro: Each class contributes equally regardless of size
#
# For balanced datasets: weighted ≈ macro
# For imbalanced: weighted favors majority class performance
```

</details>

---

## CATEGORY C: Different Model Constraints

### Variation C1: Decision Tree Constraint

You must use **Decision Trees** as your classification approach.

<details>
<summary>Solution C1</summary>

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

# Load data (same as original)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=2026, stratify=y
)

# Note: Decision trees don't need scaling, but it doesn't hurt
pipeline = Pipeline([
    ('tree', DecisionTreeClassifier(random_state=2026))
])

param_grid = {
    'tree__max_depth': [3, 5, 7, 10, 15, None],
    'tree__min_samples_split': [2, 5, 10, 20],
    'tree__min_samples_leaf': [1, 2, 5, 10],
    'tree__criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")

y_val_pred = grid_search.predict(X_val)
print(f"Validation F1-macro: {f1_score(y_val, y_val_pred, average='macro'):.4f}")

best_model = grid_search.best_estimator_
best_model.fit(X, y)

def my_prediction(X_test):
    return pd.Series(best_model.predict(X_test), name='target')
```

**Key Insights**:
- Decision trees don't need feature scaling
- Regularization via max_depth, min_samples_split prevents overfitting
- Decision trees are interpretable (can visualize the tree)
- Often outperformed by ensemble methods (Random Forest)

</details>

### Variation C2: Logistic Regression Constraint

You must use **Logistic Regression** as your classification approach.

<details>
<summary>Solution C2</summary>

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=2026, stratify=y
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Essential for logistic regression
    ('lr', LogisticRegression(max_iter=1000, random_state=2026))
])

param_grid = {
    'lr__C': [0.001, 0.01, 0.1, 1, 10, 100],  # Inverse regularization
    'lr__penalty': ['l1', 'l2'],
    'lr__solver': ['saga'],  # Supports both L1 and L2
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")

y_val_pred = grid_search.predict(X_val)
print(f"Validation F1-macro: {f1_score(y_val, y_val_pred, average='macro'):.4f}")

best_model = grid_search.best_estimator_
best_model.fit(X, y)

def my_prediction(X_test):
    return pd.Series(best_model.predict(X_test), name='target')
```

**Key Insights**:
- Logistic regression REQUIRES feature scaling
- C parameter controls regularization (smaller C = more regularization)
- L1 penalty induces sparsity (feature selection)
- L2 penalty shrinks coefficients (reduces overfitting)
- Provides probability estimates directly

</details>

### Variation C3: Random Forest Constraint

You must use **Random Forest** as your classification approach.

<details>
<summary>Solution C3</summary>

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=2026, stratify=y
)

# Random Forest doesn't need scaling
pipeline = Pipeline([
    ('rf', RandomForestClassifier(random_state=2026, n_jobs=-1))
])

param_grid = {
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [5, 10, 15, None],
    'rf__min_samples_split': [2, 5, 10],
    'rf__max_features': ['sqrt', 'log2', None]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")

y_val_pred = grid_search.predict(X_val)
print(f"Validation F1-macro: {f1_score(y_val, y_val_pred, average='macro'):.4f}")

# Feature importance analysis
best_rf = grid_search.best_estimator_.named_steps['rf']
importances = pd.Series(
    best_rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)
print("\nTop 10 Important Features:")
print(importances.head(10))

best_model = grid_search.best_estimator_
best_model.fit(X, y)

def my_prediction(X_test):
    return pd.Series(best_model.predict(X_test), name='target')
```

**Key Insights**:
- Random Forest is an ensemble of decision trees
- No scaling needed (tree-based)
- Provides feature importance
- Less prone to overfitting than single decision tree
- max_features controls randomness in tree building

</details>

### Variation C4: SVM Constraint

You must use **Support Vector Machine (SVM)** as your classification approach.

<details>
<summary>Solution C4</summary>

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=2026, stratify=y
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Essential for SVM
    ('svm', SVC(random_state=2026, probability=True))
])

param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__kernel': ['rbf', 'linear', 'poly'],
    'svm__gamma': ['scale', 'auto', 0.01, 0.1]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")

y_val_pred = grid_search.predict(X_val)
print(f"Validation F1-macro: {f1_score(y_val, y_val_pred, average='macro'):.4f}")

best_model = grid_search.best_estimator_
best_model.fit(X, y)

def my_prediction(X_test):
    return pd.Series(best_model.predict(X_test), name='target')
```

**Key Insights**:
- SVM REQUIRES feature scaling
- RBF kernel usually best for non-linear boundaries
- C controls trade-off between margin width and misclassification
- gamma controls RBF kernel width (high = complex boundary)
- probability=True enables probability estimates (slower)

</details>

### Variation C5: Naive Bayes Constraint

You must use **Naive Bayes** as your classification approach.

<details>
<summary>Solution C5</summary>

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=2026, stratify=y
)

# Gaussian Naive Bayes assumes features are normally distributed
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Helps normalize distributions
    ('nb', GaussianNB())
])

# GaussianNB has limited hyperparameters
param_grid = {
    'nb__var_smoothing': np.logspace(-12, -6, 7)  # Smoothing parameter
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")

y_val_pred = grid_search.predict(X_val)
print(f"Validation F1-macro: {f1_score(y_val, y_val_pred, average='macro'):.4f}")

best_model = grid_search.best_estimator_
best_model.fit(X, y)

def my_prediction(X_test):
    return pd.Series(best_model.predict(X_test), name='target')
```

**Key Insights**:
- Naive Bayes assumes features are conditionally independent
- GaussianNB assumes features follow Gaussian distribution
- Very fast training and prediction
- Often surprisingly good despite independence assumption
- var_smoothing prevents zero probabilities

</details>

---

## CATEGORY D: Different Preprocessing Requirements

### Variation D1: No Scaling Allowed

You must use kNN, but **you cannot use any scaling/normalization** preprocessing.

<details>
<summary>Solution D1</summary>

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=2026, stratify=y
)

# Without scaling, we need to be careful about feature magnitudes
# Option 1: Use higher k to smooth out scale effects
# Option 2: Use features with similar scales
# Option 3: Use cosine distance (scale-invariant)

knn = KNeighborsClassifier()

param_grid = {
    'n_neighbors': [5, 11, 21, 31, 51],  # Higher k for robustness
    'weights': ['distance'],  # Distance weighting helps
    'metric': ['cosine', 'correlation'],  # Scale-invariant metrics
}

grid_search = GridSearchCV(
    knn,
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")

y_val_pred = grid_search.predict(X_val)
print(f"Validation F1-macro: {f1_score(y_val, y_val_pred, average='macro'):.4f}")

best_model = grid_search.best_estimator_
best_model.fit(X, y)

def my_prediction(X_test):
    return pd.Series(best_model.predict(X_test), name='target')
```

**Key Insights**:
- Without scaling, features with large ranges dominate distance calculations
- Cosine distance is scale-invariant (only considers direction)
- Correlation distance is also scale-invariant
- Alternative: manually identify and exclude outlier features

</details>

### Variation D2: Missing Values Present

The dataset has **missing values** (NaN). You must handle them appropriately.

```python
# Simulate missing values
np.random.seed(2026)
X_missing = X.copy()
mask = np.random.random(X_missing.shape) < 0.1  # 10% missing
X_missing[mask] = np.nan
```

<details>
<summary>Solution D2</summary>

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

# Create missing data
np.random.seed(2026)
X_missing = X.copy()
mask = np.random.random(X_missing.shape) < 0.1
X_missing[mask] = np.nan

X_train, X_val, y_train, y_val = train_test_split(
    X_missing, y, test_size=0.2, random_state=2026, stratify=y
)

# Option 1: Simple imputation (mean/median)
pipeline_simple = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# Option 2: KNN imputation (more sophisticated)
pipeline_knn_impute = Pipeline([
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

param_grid = {
    'knn__n_neighbors': [3, 5, 7, 11],
    'knn__weights': ['uniform', 'distance'],
}

# Compare both approaches
for name, pipeline in [('SimpleImputer', pipeline_simple),
                        ('KNNImputer', pipeline_knn_impute)]:
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    y_val_pred = grid_search.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred, average='macro')
    print(f"{name}: Best CV F1={grid_search.best_score_:.4f}, Val F1={val_f1:.4f}")

# Use the better one
best_model = grid_search.best_estimator_
best_model.fit(X_missing, y)

def my_prediction(X_test):
    return pd.Series(best_model.predict(X_test), name='target')
```

**Key Insights**:
- SimpleImputer: Fast, uses mean/median/mode
- KNNImputer: Uses similar samples to impute (better for MAR data)
- Imputation must be done before scaling
- Alternative: Use models that handle missing values (tree-based)

</details>

### Variation D3: Categorical Features Present

Some features are categorical and must be encoded:

```python
# Add categorical features
X['category_A'] = np.random.choice(['low', 'medium', 'high'], len(X))
X['category_B'] = np.random.choice(['type1', 'type2'], len(X))
```

<details>
<summary>Solution D3</summary>

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

# Add categorical features
np.random.seed(2026)
X_cat = X.copy()
X_cat['category_A'] = np.random.choice(['low', 'medium', 'high'], len(X))
X_cat['category_B'] = np.random.choice(['type1', 'type2'], len(X))

X_train, X_val, y_train, y_val = train_test_split(
    X_cat, y, test_size=0.2, random_state=2026, stratify=y
)

# Identify column types
numeric_features = X_cat.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_features = X_cat.select_dtypes(include=['object']).columns.tolist()

print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {categorical_features}")

# Column transformer for mixed data types
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('knn', KNeighborsClassifier())
])

param_grid = {
    'knn__n_neighbors': [3, 5, 7, 11],
    'knn__weights': ['uniform', 'distance'],
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")

y_val_pred = grid_search.predict(X_val)
print(f"Validation F1-macro: {f1_score(y_val, y_val_pred, average='macro'):.4f}")

best_model = grid_search.best_estimator_
best_model.fit(X_cat, y)

def my_prediction(X_test):
    return pd.Series(best_model.predict(X_test), name='target')
```

**Key Insights**:
- ColumnTransformer handles different preprocessing per feature type
- OneHotEncoder for categorical → binary columns
- drop='first' avoids dummy variable trap
- kNN can work with mixed features after proper encoding
- Alternative: Target encoding for high-cardinality categoricals

</details>

### Variation D4: Outliers Present

The dataset has **outliers** (extreme values). Handle them appropriately.

<details>
<summary>Solution D4</summary>

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

# Add outliers
np.random.seed(2026)
X_outliers = X.copy()
# Add extreme values to random samples
outlier_indices = np.random.choice(len(X), size=int(0.05 * len(X)), replace=False)
X_outliers.iloc[outlier_indices] *= 10

X_train, X_val, y_train, y_val = train_test_split(
    X_outliers, y, test_size=0.2, random_state=2026, stratify=y
)

# Option 1: RobustScaler (uses median and IQR)
pipeline_robust = Pipeline([
    ('scaler', RobustScaler()),
    ('knn', KNeighborsClassifier())
])

# Option 2: Clip outliers before scaling
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_percentile=5, upper_percentile=95):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def fit(self, X, y=None):
        self.lower_ = np.percentile(X, self.lower_percentile, axis=0)
        self.upper_ = np.percentile(X, self.upper_percentile, axis=0)
        return self

    def transform(self, X):
        X_clipped = np.clip(X, self.lower_, self.upper_)
        return X_clipped

pipeline_clip = Pipeline([
    ('clipper', OutlierClipper()),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

param_grid = {
    'knn__n_neighbors': [3, 5, 7, 11],
    'knn__weights': ['uniform', 'distance'],
}

# Compare approaches
for name, pipeline in [('RobustScaler', pipeline_robust),
                        ('OutlierClipper', pipeline_clip)]:
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    y_val_pred = grid_search.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred, average='macro')
    print(f"{name}: Best CV F1={grid_search.best_score_:.4f}, Val F1={val_f1:.4f}")

best_model = grid_search.best_estimator_
best_model.fit(X_outliers, y)

def my_prediction(X_test):
    return pd.Series(best_model.predict(X_test), name='target')
```

**Key Insights**:
- RobustScaler uses median and IQR (robust to outliers)
- Clipping caps extreme values at percentiles
- kNN is sensitive to outliers (affects distances)
- Alternative: Use Isolation Forest to detect and remove outliers

</details>

---

## CATEGORY E: Multi-Class Extensions

### Variation E1: Multi-Class Classification (>2 classes)

Modify for a **10-class digit classification** problem:

```python
from sklearn.datasets import load_digits

data = load_digits()
X = pd.DataFrame(data.data)
y = pd.Series(data.target, name='target')
```

<details>
<summary>Solution E1</summary>

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report

data = load_digits()
X = pd.DataFrame(data.data)
y = pd.Series(data.target, name='target')

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=2026, stratify=y
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# For multi-class, consider more neighbors for stability
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 11, 15],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan']
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1_macro',  # Macro for multi-class
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV F1-macro: {grid_search.best_score_:.4f}")

y_val_pred = grid_search.predict(X_val)
print(f"\nValidation F1-macro: {f1_score(y_val, y_val_pred, average='macro'):.4f}")
print("\nPer-class report:")
print(classification_report(y_val, y_val_pred))

best_model = grid_search.best_estimator_
best_model.fit(X, y)

def my_prediction(X_test):
    return pd.Series(best_model.predict(X_test), name='target')
```

**Key Insights**:
- Multi-class F1-macro treats all classes equally
- Per-class report reveals which digits are confused
- 8 and 1 are often confused (similar shapes)
- Higher k can help with 10-class stability

</details>

### Variation E2: Imbalanced Multi-Class

Multi-class with **severe class imbalance**:

```python
# Create imbalanced dataset
from sklearn.datasets import load_digits

data = load_digits()
X_all = pd.DataFrame(data.data)
y_all = pd.Series(data.target)

# Keep different amounts per class
np.random.seed(2026)
selected_indices = []
class_sizes = [50, 50, 50, 100, 100, 150, 150, 20, 20, 30]  # Imbalanced
for cls, size in enumerate(class_sizes):
    cls_indices = np.where(y_all == cls)[0]
    selected_indices.extend(np.random.choice(cls_indices, size=size, replace=False))

X = X_all.iloc[selected_indices].reset_index(drop=True)
y = y_all.iloc[selected_indices].reset_index(drop=True)
```

<details>
<summary>Solution E2</summary>

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Create imbalanced data (as above)
# ...

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=2026, stratify=y
)

print("Class distribution in training set:")
print(y_train.value_counts().sort_index())

# Option 1: Class weights (not directly in kNN, but in voting)
# Option 2: Oversampling minority classes (SMOTE)

# Using SMOTE for oversampling
# Note: imblearn Pipeline handles SMOTE before fitting
pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=2026, k_neighbors=3)),
    ('knn', KNeighborsClassifier())
])

param_grid = {
    'knn__n_neighbors': [3, 5, 7],
    'knn__weights': ['distance'],
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV F1-macro: {grid_search.best_score_:.4f}")

y_val_pred = grid_search.predict(X_val)
print(f"\nValidation F1-macro: {f1_score(y_val, y_val_pred, average='macro'):.4f}")
print("\nPer-class report:")
print(classification_report(y_val, y_val_pred))

best_model = grid_search.best_estimator_
best_model.fit(X, y)

def my_prediction(X_test):
    return pd.Series(best_model.predict(X_test), name='target')
```

**Key Insights**:
- SMOTE creates synthetic minority samples
- imblearn Pipeline handles train-time only resampling
- F1-macro gives equal weight to all classes
- Minority classes (7, 8) benefit most from SMOTE
- Alternative: Class-weighted loss (not native to kNN)

</details>

---

## CATEGORY F: Advanced Pipelines

### Variation F1: Feature Selection + kNN

Select the **best k features** before applying kNN:

<details>
<summary>Solution F1</summary>

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=2026, stratify=y
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(score_func=f_classif)),
    ('knn', KNeighborsClassifier())
])

param_grid = {
    'selector__k': [5, 10, 15, 20, 25, 'all'],
    'knn__n_neighbors': [3, 5, 7, 11],
    'knn__weights': ['uniform', 'distance'],
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")

# See which features were selected
best_selector = grid_search.best_estimator_.named_steps['selector']
selected_mask = best_selector.get_support()
selected_features = X.columns[selected_mask].tolist()
print(f"\nSelected features ({len(selected_features)}):")
print(selected_features)

y_val_pred = grid_search.predict(X_val)
print(f"\nValidation F1-macro: {f1_score(y_val, y_val_pred, average='macro'):.4f}")

best_model = grid_search.best_estimator_
best_model.fit(X, y)

def my_prediction(X_test):
    return pd.Series(best_model.predict(X_test), name='target')
```

**Key Insights**:
- SelectKBest uses statistical tests (f_classif for classification)
- Reducing features can improve kNN (curse of dimensionality)
- mutual_info_classif captures non-linear relationships
- Too few features → underfitting; too many → overfitting

</details>

### Variation F2: PCA + kNN

Use **PCA** for dimensionality reduction:

<details>
<summary>Solution F2</summary>

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=2026, stratify=y
)

# First, explore variance explained
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
pca_full = PCA()
pca_full.fit(X_scaled)

# Plot cumulative variance
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
plt.figure(figsize=(10, 4))
plt.plot(range(1, len(cumvar)+1), cumvar, 'b-o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained')
plt.title('PCA Variance Explained')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.legend()
plt.grid(True)
plt.show()

# Find components for 95% variance
n_95 = np.argmax(cumvar >= 0.95) + 1
print(f"Components for 95% variance: {n_95}")

# Build pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('knn', KNeighborsClassifier())
])

param_grid = {
    'pca__n_components': [5, 10, 15, n_95, 'mle'],
    'knn__n_neighbors': [3, 5, 7, 11],
    'knn__weights': ['uniform', 'distance'],
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")

y_val_pred = grid_search.predict(X_val)
print(f"Validation F1-macro: {f1_score(y_val, y_val_pred, average='macro'):.4f}")

best_model = grid_search.best_estimator_
best_model.fit(X, y)

def my_prediction(X_test):
    return pd.Series(best_model.predict(X_test), name='target')
```

**Key Insights**:
- PCA projects to orthogonal components (uncorrelated)
- kNN benefits from uncorrelated features
- 'mle' estimates optimal components using Maximum Likelihood
- Scaling REQUIRED before PCA

</details>

### Variation F3: Ensemble of kNNs

Combine **multiple kNN models** with different hyperparameters:

<details>
<summary>Solution F3</summary>

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=2026, stratify=y
)

# Scale data first
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Create ensemble of kNNs with different k values
ensemble = VotingClassifier(
    estimators=[
        ('knn_3', KNeighborsClassifier(n_neighbors=3, weights='distance')),
        ('knn_5', KNeighborsClassifier(n_neighbors=5, weights='distance')),
        ('knn_7', KNeighborsClassifier(n_neighbors=7, weights='distance')),
        ('knn_11', KNeighborsClassifier(n_neighbors=11, weights='distance')),
        ('knn_15', KNeighborsClassifier(n_neighbors=15, weights='uniform')),
    ],
    voting='soft'  # Use probability averaging
)

ensemble.fit(X_train_scaled, y_train)

y_val_pred = ensemble.predict(X_val_scaled)
print(f"Ensemble Validation F1-macro: {f1_score(y_val, y_val_pred, average='macro'):.4f}")

# Compare with individual models
for name, clf in ensemble.named_estimators_.items():
    clf.fit(X_train_scaled, y_train)
    pred = clf.predict(X_val_scaled)
    print(f"{name} F1-macro: {f1_score(y_val, pred, average='macro'):.4f}")

# Final model (retrain on all data)
X_all_scaled = scaler.fit_transform(X)
ensemble.fit(X_all_scaled, y)

def my_prediction(X_test):
    X_test_scaled = scaler.transform(X_test)
    return pd.Series(ensemble.predict(X_test_scaled), name='target')
```

**Key Insights**:
- VotingClassifier combines multiple models
- 'soft' voting averages probabilities (better than hard)
- Different k values capture different locality scales
- Ensemble often outperforms best individual model

</details>

### Variation F4: Stacking with kNN

Use **kNN as a base model** in a stacking ensemble:

<details>
<summary>Solution F4</summary>

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=2026, stratify=y
)

# Create pipeline with scaling
base_estimators = [
    ('knn', Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=5, weights='distance'))
    ])),
    ('tree', DecisionTreeClassifier(max_depth=5, random_state=2026)),
    ('lr', Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=2026, max_iter=1000))
    ]))
]

# Meta-learner (combines base predictions)
stacking = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(random_state=2026),
    cv=5,
    stack_method='predict_proba'  # Use probabilities
)

stacking.fit(X_train, y_train)

y_val_pred = stacking.predict(X_val)
print(f"Stacking Validation F1-macro: {f1_score(y_val, y_val_pred, average='macro'):.4f}")

# Compare with individual models
for name, clf in stacking.named_estimators_.items():
    pred = clf.predict(X_val)
    print(f"{name} F1-macro: {f1_score(y_val, pred, average='macro'):.4f}")

# Final model
stacking.fit(X, y)

def my_prediction(X_test):
    return pd.Series(stacking.predict(X_test), name='target')
```

**Key Insights**:
- Stacking uses base model outputs as features for meta-learner
- 'predict_proba' provides richer information than 'predict'
- Diverse base models → better stacking performance
- kNN captures local patterns, tree captures global splits

</details>

---

## CATEGORY G: Coding Constraints

### Variation G1: Implement kNN from Scratch

You must implement kNN **from scratch using NumPy only**. No scikit-learn kNN allowed.

<details>
<summary>Solution G1</summary>

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from collections import Counter

class KNNFromScratch:
    """k-Nearest Neighbors implemented from scratch."""

    def __init__(self, n_neighbors=5, weights='uniform', metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric

    def fit(self, X, y):
        """Store training data."""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.classes_ = np.unique(self.y_train)
        return self

    def _compute_distances(self, X):
        """Compute pairwise distances between X and training data."""
        X = np.array(X)

        if self.metric == 'euclidean':
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
            X_sq = np.sum(X**2, axis=1).reshape(-1, 1)
            train_sq = np.sum(self.X_train**2, axis=1).reshape(1, -1)
            cross = 2 * X @ self.X_train.T
            distances = np.sqrt(np.maximum(X_sq + train_sq - cross, 0))

        elif self.metric == 'manhattan':
            # Sum of absolute differences
            distances = np.sum(
                np.abs(X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]),
                axis=2
            )
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        return distances

    def predict(self, X):
        """Predict class labels for X."""
        distances = self._compute_distances(X)
        predictions = []

        for i in range(len(X)):
            # Find k nearest neighbors
            k_indices = np.argsort(distances[i])[:self.n_neighbors]
            k_labels = self.y_train[k_indices]
            k_distances = distances[i][k_indices]

            if self.weights == 'uniform':
                # Simple majority vote
                counts = Counter(k_labels)
                pred = counts.most_common(1)[0][0]

            elif self.weights == 'distance':
                # Weighted by inverse distance
                weights = 1 / (k_distances + 1e-10)  # Avoid division by zero
                class_weights = {}
                for cls in self.classes_:
                    mask = k_labels == cls
                    class_weights[cls] = np.sum(weights[mask])
                pred = max(class_weights, key=class_weights.get)

            predictions.append(pred)

        return np.array(predictions)

    def predict_proba(self, X):
        """Predict class probabilities for X."""
        distances = self._compute_distances(X)
        n_classes = len(self.classes_)
        probas = np.zeros((len(X), n_classes))

        for i in range(len(X)):
            k_indices = np.argsort(distances[i])[:self.n_neighbors]
            k_labels = self.y_train[k_indices]
            k_distances = distances[i][k_indices]

            if self.weights == 'uniform':
                for j, cls in enumerate(self.classes_):
                    probas[i, j] = np.sum(k_labels == cls) / self.n_neighbors

            elif self.weights == 'distance':
                weights = 1 / (k_distances + 1e-10)
                total_weight = np.sum(weights)
                for j, cls in enumerate(self.classes_):
                    mask = k_labels == cls
                    probas[i, j] = np.sum(weights[mask]) / total_weight

        return probas


# Test the implementation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=2026, stratify=y
)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Test our implementation
knn_scratch = KNNFromScratch(n_neighbors=5, weights='distance')
knn_scratch.fit(X_train_scaled, y_train)
y_pred_scratch = knn_scratch.predict(X_val_scaled)
f1_scratch = f1_score(y_val, y_pred_scratch, average='macro')
print(f"From-scratch kNN F1-macro: {f1_scratch:.4f}")

# Compare with sklearn
from sklearn.neighbors import KNeighborsClassifier
knn_sklearn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn_sklearn.fit(X_train_scaled, y_train)
y_pred_sklearn = knn_sklearn.predict(X_val_scaled)
f1_sklearn = f1_score(y_val, y_pred_sklearn, average='macro')
print(f"Sklearn kNN F1-macro: {f1_sklearn:.4f}")

# Final model
X_all_scaled = scaler.fit_transform(X)
knn_final = KNNFromScratch(n_neighbors=5, weights='distance')
knn_final.fit(X_all_scaled, y)

def my_prediction(X_test):
    X_test_scaled = scaler.transform(X_test)
    return pd.Series(knn_final.predict(X_test_scaled), name='target')
```

**Key Insights**:
- Distance computation is vectorized for efficiency
- Euclidean: ||a-b||² = ||a||² + ||b||² - 2a·b
- Weights='distance' requires careful handling of zero distances
- From-scratch should match sklearn results

</details>

### Variation G2: Memory-Constrained kNN

Your solution must work with **limited memory**. Use batch processing for large test sets.

<details>
<summary>Solution G2</summary>

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=2026, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(X_train_scaled, y_train)

# Memory-efficient batch prediction
def batch_predict(model, X, batch_size=100):
    """Predict in batches to limit memory usage."""
    n_samples = len(X)
    predictions = []

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = X[start:end]
        batch_pred = model.predict(batch)
        predictions.extend(batch_pred)

        # Optional: Force garbage collection
        # import gc; gc.collect()

    return np.array(predictions)

# Test batch prediction
y_val_pred = batch_predict(knn, X_val_scaled, batch_size=50)
print(f"Batch prediction F1-macro: {f1_score(y_val, y_val_pred, average='macro'):.4f}")

# Final model
X_all_scaled = scaler.fit_transform(X)
knn.fit(X_all_scaled, y)

def my_prediction(X_test, batch_size=100):
    """Memory-efficient prediction for large test sets."""
    X_test_scaled = scaler.transform(X_test)
    y_pred = batch_predict(knn, X_test_scaled, batch_size)
    return pd.Series(y_pred, name='target')
```

**Key Insights**:
- Batch processing reduces peak memory usage
- kNN stores all training data → memory scales with n_train
- For very large datasets, consider approximate kNN (e.g., Annoy, FAISS)
- batch_size trades off memory vs. speed

</details>

### Variation G3: Time-Constrained Prediction

Your `my_prediction` function must complete in **under 1 second** for 1000 test samples.

<details>
<summary>Solution G3</summary>

```python
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import BallTree, KDTree
from sklearn.metrics import f1_score

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=2026, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Compare different algorithms
algorithms = {
    'brute': KNeighborsClassifier(n_neighbors=5, algorithm='brute'),
    'kd_tree': KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree'),
    'ball_tree': KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree'),
}

print("Timing comparison (100 samples):")
for name, knn in algorithms.items():
    knn.fit(X_train_scaled, y_train)

    # Time prediction
    start = time.time()
    for _ in range(10):  # Average over 10 runs
        _ = knn.predict(X_val_scaled[:100])
    elapsed = (time.time() - start) / 10

    f1 = f1_score(y_val[:100], knn.predict(X_val_scaled[:100]), average='macro')
    print(f"  {name}: {elapsed*1000:.2f}ms, F1={f1:.4f}")

# Use fastest algorithm
knn_fast = KNeighborsClassifier(
    n_neighbors=5,
    algorithm='ball_tree',  # Usually fastest for moderate dimensions
    leaf_size=40  # Tune for speed
)
knn_fast.fit(X_train_scaled, y_train)

# Verify timing constraint
X_test_1000 = np.random.randn(1000, X.shape[1])
X_test_1000_scaled = scaler.transform(X_test_1000)

start = time.time()
_ = knn_fast.predict(X_test_1000_scaled)
elapsed = time.time() - start
print(f"\n1000 samples prediction time: {elapsed:.3f}s")
assert elapsed < 1.0, f"Too slow: {elapsed:.3f}s > 1.0s"

# Final model
X_all_scaled = scaler.fit_transform(X)
knn_fast.fit(X_all_scaled, y)

def my_prediction(X_test):
    X_test_scaled = scaler.transform(X_test)
    return pd.Series(knn_fast.predict(X_test_scaled), name='target')
```

**Key Insights**:
- 'ball_tree' and 'kd_tree' are faster than 'brute' for small k
- 'kd_tree' degrades in high dimensions (curse of dimensionality)
- 'ball_tree' more robust for higher dimensions
- leaf_size affects tree depth vs. linear search trade-off

</details>

---

## CATEGORY H: Report and Documentation

### Variation H1: Feature Importance Analysis

Include a **feature importance analysis** in your report.

<details>
<summary>Solution H1</summary>

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=2026, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(X_train_scaled, y_train)

# Permutation importance (works for any model)
perm_importance = permutation_importance(
    knn, X_val_scaled, y_val,
    n_repeats=10,
    random_state=2026,
    scoring='f1_macro'
)

# Create importance DataFrame
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)

print("Feature Importance (Permutation):")
print(importance_df.head(15))

# Visualize
plt.figure(figsize=(12, 8))
top_n = 15
plt.barh(range(top_n),
         importance_df['importance_mean'].head(top_n)[::-1],
         xerr=importance_df['importance_std'].head(top_n)[::-1])
plt.yticks(range(top_n), importance_df['feature'].head(top_n)[::-1])
plt.xlabel('Permutation Importance (F1-macro decrease)')
plt.title('Top 15 Important Features for kNN')
plt.tight_layout()
plt.show()

# Report section
print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)
print("""
Methodology: Permutation importance measures how much the model's
F1-macro score decreases when each feature is randomly shuffled.

Key Findings:
1. Top 3 features: {}, {}, {}
2. These features contribute {:.1f}% of total importance
3. Features with near-zero importance could be removed

Interpretation:
- High importance: Feature is crucial for kNN distance calculations
- Low importance: Feature provides little discriminative information
- Negative importance: Feature may introduce noise
""".format(
    importance_df.iloc[0]['feature'],
    importance_df.iloc[1]['feature'],
    importance_df.iloc[2]['feature'],
    100 * importance_df.head(3)['importance_mean'].sum() /
          importance_df['importance_mean'].sum()
))
```

**Key Insights**:
- Permutation importance is model-agnostic
- Shows feature importance in model's decision making
- Different from feature selection (tests removal, not shuffling)
- Can identify potentially removable features

</details>

### Variation H2: Hyperparameter Sensitivity Analysis

Include a **hyperparameter sensitivity analysis** showing how k affects performance.

<details>
<summary>Solution H2</summary>

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=2026, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Test different k values
k_values = list(range(1, 31, 2))  # 1, 3, 5, ..., 29
results = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')

    # Cross-validation scores
    cv_scores = cross_val_score(knn, X_train_scaled, y_train,
                                 cv=5, scoring='f1_macro')

    results.append({
        'k': k,
        'mean_f1': cv_scores.mean(),
        'std_f1': cv_scores.std()
    })

results_df = pd.DataFrame(results)

# Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.errorbar(results_df['k'], results_df['mean_f1'],
             yerr=results_df['std_f1'], marker='o', capsize=5)
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('F1-macro Score')
plt.title('kNN Performance vs. k')
plt.grid(True, alpha=0.3)

# Mark best k
best_idx = results_df['mean_f1'].idxmax()
best_k = results_df.loc[best_idx, 'k']
best_f1 = results_df.loc[best_idx, 'mean_f1']
plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best k={best_k}')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(results_df['k'], results_df['std_f1'], marker='s', color='orange')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Std Dev of F1-macro')
plt.title('Prediction Stability vs. k')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Report
print("\n" + "="*60)
print("HYPERPARAMETER SENSITIVITY ANALYSIS")
print("="*60)
print(f"""
k-Nearest Neighbors Hyperparameter: k (number of neighbors)

Tested Range: k ∈ {{{min(k_values)}, {max(k_values)}}}

Key Findings:
1. Best k = {best_k} with F1-macro = {best_f1:.4f}
2. Performance is relatively stable for k ∈ [{results_df[results_df['mean_f1'] > best_f1-0.01]['k'].min()},
                                              {results_df[results_df['mean_f1'] > best_f1-0.01]['k'].max()}]
3. Very small k (k=1) shows higher variance (overfitting)
4. Very large k shows decreasing performance (underfitting)

Interpretation:
- Small k: Captures local patterns, but noisy
- Large k: Smooths decision boundary, may miss local structure
- Optimal k balances bias-variance trade-off

Recommendation: k = {best_k} provides best trade-off for this dataset.
""")
```

**Key Insights**:
- k=1 is essentially memorization (high variance)
- Large k approaches global majority voting (high bias)
- Optimal k depends on data complexity and noise level
- Error bars show prediction stability (lower is better)

</details>

---

## KEY FORMULAS SUMMARY

| Concept | Formula/Definition |
|---------|-------------------|
| **Euclidean Distance** | d(x,y) = √(Σᵢ (xᵢ - yᵢ)²) |
| **Manhattan Distance** | d(x,y) = Σᵢ \|xᵢ - yᵢ\| |
| **Cosine Distance** | d(x,y) = 1 - (x·y)/(‖x‖·‖y‖) |
| **kNN Prediction (uniform)** | ŷ = mode(y_neighbors) |
| **kNN Prediction (distance)** | ŷ = argmax_c Σᵢ (1/dᵢ) · 𝟙(yᵢ=c) |
| **F1 Score** | F1 = 2 · (precision · recall)/(precision + recall) |
| **F1-macro** | F1_macro = (1/C) · Σc F1_c |
| **F1-weighted** | F1_weighted = Σc (nᵢ/N) · F1_c |
| **StandardScaler** | x' = (x - μ)/σ |
| **MinMaxScaler** | x' = (x - min)/(max - min) |
| **RobustScaler** | x' = (x - median)/IQR |

---

## ATOMIC SKILLS CHECKLIST

- [ ] Load and explore dataset structure
- [ ] Split data with stratification
- [ ] Apply appropriate scaling (Standard, MinMax, Robust)
- [ ] Build scikit-learn Pipeline
- [ ] Use GridSearchCV with appropriate scoring
- [ ] Interpret F1-macro vs F1-weighted vs accuracy
- [ ] Handle missing values (SimpleImputer, KNNImputer)
- [ ] Handle categorical features (OneHotEncoder)
- [ ] Handle outliers (RobustScaler, clipping)
- [ ] Perform feature selection (SelectKBest)
- [ ] Apply dimensionality reduction (PCA)
- [ ] Build ensemble models (Voting, Stacking)
- [ ] Analyze feature importance (permutation)
- [ ] Analyze hyperparameter sensitivity
- [ ] Write clear report with justification

---

## COMMON MISCONCEPTIONS

1. **kNN doesn't need scaling**: WRONG. kNN is distance-based, so features with larger scales dominate. Always scale before kNN (except with scale-invariant metrics like cosine).

2. **Higher k is always better**: WRONG. Higher k smooths the decision boundary but may miss local patterns. Optimal k depends on data complexity.

3. **F1-macro = accuracy for balanced datasets**: APPROXIMATELY true, but not exactly. F1 is harmonic mean of precision and recall.

4. **GridSearchCV uses test data**: WRONG. GridSearchCV only uses training data with internal cross-validation. Test data should be held out separately.

5. **Feature selection removes unimportant features**: It removes features that don't help THIS model. Different models may need different features.

6. **Stratified split only matters for imbalanced data**: Good practice for all classification. Ensures consistent class proportions in train/val/test.

7. **Pipeline automatically handles new data**: Yes, but only if the pipeline is fitted on training data first. The scaler must be fit on training only.

8. **kNN probabilities are well-calibrated**: NOT necessarily. kNN probabilities are proportion of k neighbors in each class, which may not be calibrated.
