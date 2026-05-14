
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("dataset/careers_dataset.csv")

# Features and target
X = df.drop("career_path", axis=1)
y = df["career_path"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

# Train Decision Tree model
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, "model/model.pkl")
joblib.dump(label_encoder, "model/label_encoder.pkl")

print("\nModel saved successfully!")

# -----------------------------
# Visualization Section
# -----------------------------

# Career Distribution Chart
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="career_path")
plt.xticks(rotation=10)
plt.title("Career Distribution")
plt.tight_layout()
plt.savefig("static/images/career_distribution.png")
plt.close()

# Correlation Heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(df.drop("career_path", axis=1).corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("static/images/correlation_heatmap.png")
plt.close()

# Feature Importance
importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("static/images/feature_importance.png")
plt.close()

# Decision Tree Visualization
plt.figure(figsize=(20, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=label_encoder.classes_,
    filled=True,
    fontsize=7
)

plt.title("Decision Tree Visualization")
plt.savefig("static/images/decision_tree.png")
plt.close()

print("\nVisualization images saved successfully!")
