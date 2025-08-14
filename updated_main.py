# ===============================================
# Google Colab Data Science Project Template
# With Interactive Hyperparameter Tuning & AutoML
# ===============================================

# Install dependencies
!pip install pandas numpy matplotlib seaborn scikit-learn shap lazypredict ipywidgets --quiet

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lazypredict.Supervised import LazyClassifier
from sklearn.metrics import classification_report, confusion_matrix
import ipywidgets as widgets
from IPython.display import display

# ===============================================
# 1. Load Dataset
# ===============================================
# Example dataset: Titanic
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# ===============================================
# 2. Auto Data Cleaning
# ===============================================
df = df.dropna(subset=['Survived'])  # Remove rows without target
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)  # Fill categorical NA
for col in df.select_dtypes(include=np.number).columns:
    df[col].fillna(df[col].mean(), inplace=True)  # Fill numeric NA

# ===============================================
# 3. Preprocessing
# ===============================================
df = pd.get_dummies(df, drop_first=True)
target = "Survived"
X = df.drop(columns=[target])
y = df[target]

# ===============================================
# 4. Customizable Train-Test Split and Model Training
# ===============================================
def run_model(test_size, n_estimators, max_depth):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train Model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Predictions & Report
    y_pred = model.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    # Feature Importance using SHAP
    print("\nGenerating SHAP summary plot...")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)

        print(f"Shape of X_train: {X_train.shape}")
        print(f"Shape of shap_values: {shap_values[1].shape if isinstance(shap_values, list) else shap_values.shape}")
        print(f"Number of features in X_train: {X_train.shape[1]}")
        print(f"Number of features in shap_values: {shap_values[1].shape[1] if isinstance(shap_values, list) else shap_values.shape[1]}")


        # For classification, shap_values is a list of arrays, one for each class.
        # We need to plot the shap values for the positive class (usually index 1).
        if isinstance(shap_values, list):
             shap.summary_plot(shap_values[1], X_train, feature_names=X_train.columns.tolist())
        else:
             # For regression or other cases where shap_values is a single array
             shap.summary_plot(shap_values, X_train, feature_names=X_train.columns.tolist())

    except Exception as e:
        print(f"An error occurred during SHAP plot generation: {e}")


# Interactive UI
widgets.interact(
    run_model,
    test_size=widgets.FloatSlider(value=0.2, min=0.1, max=0.5, step=0.05),
    n_estimators=widgets.IntSlider(value=100, min=10, max=500, step=10),
    max_depth=widgets.IntSlider(value=5, min=1, max=20, step=1)
);

# ===============================================
# 5. AutoML Option
# ===============================================
print("\n=== Running AutoML with LazyPredict ===")
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
X_train_auto, X_test_auto, y_train_auto, y_test_auto = train_test_split(X, y, test_size=0.2, random_state=42)
models, predictions = clf.fit(X_train_auto, X_test_auto, y_train_auto, y_test_auto)
print(models)
