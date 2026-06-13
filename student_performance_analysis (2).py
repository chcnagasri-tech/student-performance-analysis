"""
============================================================
 Student Performance Analysis — Exploratory Data Analysis
 Author : Cherukuri Chinmaya Nagasri
 Domain : Data Science / Machine Learning
 Dataset: UCI Student Performance Dataset (generated sample)
 Tools  : Python, Pandas, Matplotlib, Seaborn, Scikit-learn
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ── Reproducibility ──────────────────────────────────────────
np.random.seed(42)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 130, "axes.titlesize": 13, "axes.labelsize": 11})

# ════════════════════════════════════════════════════════════
# 1. CREATE DATASET  (mirrors UCI Student Performance structure)
# ════════════════════════════════════════════════════════════
n = 395
data = {
    "age":            np.random.randint(15, 22, n),
    "study_hours":    np.clip(np.random.normal(5, 2.5, n), 0, 12).round(1),
    "absences":       np.random.poisson(4, n),
    "failures":       np.random.choice([0, 1, 2, 3], n, p=[0.67, 0.20, 0.09, 0.04]),
    "internet":       np.random.choice(["yes", "no"], n, p=[0.66, 0.34]),
    "extra_support":  np.random.choice(["yes", "no"], n, p=[0.45, 0.55]),
    "parent_edu":     np.random.choice(["none","primary","secondary","higher"], n,
                                       p=[0.10, 0.25, 0.35, 0.30]),
    "gender":         np.random.choice(["M", "F"], n),
}
# Final grade (G3) influenced by study hours, failures, absences
g3 = (
    10
    + 0.9  * data["study_hours"]
    - 2.1  * data["failures"]
    - 0.15 * data["absences"]
    + np.where(np.array(data["internet"]) == "yes", 0.8, 0)
    + np.where(np.array(data["extra_support"]) == "yes", 0.5, 0)
    + np.random.normal(0, 1.8, n)
)
data["final_grade"] = np.clip(g3, 0, 20).round(1)
df = pd.DataFrame(data)

# ════════════════════════════════════════════════════════════
# 2. DATA OVERVIEW
# ════════════════════════════════════════════════════════════
print("=" * 55)
print("  STUDENT PERFORMANCE ANALYSIS — EDA REPORT")
print("=" * 55)
print(f"\nDataset shape : {df.shape[0]} students × {df.shape[1]} features")
print(f"Missing values: {df.isnull().sum().sum()}")
print("\n── First 5 rows ──")
print(df.head())
print("\n── Summary Statistics ──")
print(df.describe().round(2))

# ════════════════════════════════════════════════════════════
# 3. EDA VISUALISATIONS  (saved as one figure with subplots)
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Student Performance — Exploratory Data Analysis\nAuthor: Cherukuri Chinmaya Nagasri",
             fontsize=15, fontweight="bold", y=1.01)

# 3a. Distribution of Final Grade
sns.histplot(df["final_grade"], bins=20, kde=True, ax=axes[0,0], color="#FF9900")
axes[0,0].set_title("Distribution of Final Grade (G3)")
axes[0,0].set_xlabel("Final Grade (0–20)")

# 3b. Study Hours vs Final Grade
sns.scatterplot(data=df, x="study_hours", y="final_grade",
                hue="gender", alpha=0.6, ax=axes[0,1], palette={"M":"#1f77b4","F":"#e377c2"})
axes[0,1].set_title("Study Hours vs Final Grade")

# 3c. Effect of Past Failures
sns.boxplot(data=df, x="failures", y="final_grade", ax=axes[0,2], palette="Oranges")
axes[0,2].set_title("Past Failures vs Final Grade")
axes[0,2].set_xlabel("Number of Past Failures")

# 3d. Internet Access impact
sns.violinplot(data=df, x="internet", y="final_grade", ax=axes[1,0],
               palette={"yes":"#2ca02c","no":"#d62728"}, inner="quartile")
axes[1,0].set_title("Internet Access vs Final Grade")

# 3e. Absences vs Final Grade
sns.scatterplot(data=df, x="absences", y="final_grade", alpha=0.5,
                ax=axes[1,1], color="#9467bd")
axes[1,1].set_title("Absences vs Final Grade")

# 3f. Correlation Heatmap (numeric features)
num_cols = ["age","study_hours","absences","failures","final_grade"]
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlOrRd", ax=axes[1,2],
            linewidths=0.5, square=True)
axes[1,2].set_title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("/home/claude/eda_plots.png", bbox_inches="tight")
plt.close()
print("\n[✓] EDA plots saved → eda_plots.png")

# ════════════════════════════════════════════════════════════
# 4. KEY INSIGHTS
# ════════════════════════════════════════════════════════════
print("\n── Key Insights ──")
corr_study = df["study_hours"].corr(df["final_grade"])
corr_fail  = df["failures"].corr(df["final_grade"])
corr_abs   = df["absences"].corr(df["final_grade"])
inet_yes   = df[df["internet"]=="yes"]["final_grade"].mean()
inet_no    = df[df["internet"]=="no"]["final_grade"].mean()

print(f"  • Correlation: Study Hours  ↔ Final Grade : {corr_study:+.3f}  (positive)")
print(f"  • Correlation: Failures     ↔ Final Grade : {corr_fail:+.3f}  (negative)")
print(f"  • Correlation: Absences     ↔ Final Grade : {corr_abs:+.3f}  (negative)")
print(f"  • Avg grade WITH internet   : {inet_yes:.2f}")
print(f"  • Avg grade WITHOUT internet: {inet_no:.2f}")
print(f"  • Internet advantage        : +{inet_yes - inet_no:.2f} points")

# ════════════════════════════════════════════════════════════
# 5. SIMPLE LINEAR REGRESSION MODEL
# ════════════════════════════════════════════════════════════
print("\n── Linear Regression: Study Hours → Final Grade ──")
X = df[["study_hours"]]
y = df["final_grade"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"  • Coefficient (slope) : {model.coef_[0]:.4f}")
print(f"  • Intercept           : {model.intercept_:.4f}")
print(f"  • R² Score            : {r2_score(y_test, y_pred):.4f}")
print(f"  • RMSE                : {mean_squared_error(y_test, y_pred)**0.5:.4f}")

# Regression plot
fig2, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X_test, y_test, alpha=0.5, color="#FF9900", label="Actual")
ax.plot(X_test, y_pred, color="#1A1A1A", linewidth=2, label="Predicted (Linear Regression)")
ax.set_title("Linear Regression: Study Hours → Final Grade\nAuthor: Cherukuri Chinmaya Nagasri",
             fontweight="bold")
ax.set_xlabel("Study Hours per Week")
ax.set_ylabel("Final Grade (0–20)")
ax.legend()
plt.tight_layout()
plt.savefig("/home/claude/regression_plot.png", bbox_inches="tight")
plt.close()
print("\n[✓] Regression plot saved → regression_plot.png")

print("\n" + "=" * 55)
print("  ANALYSIS COMPLETE")
print("=" * 55)
