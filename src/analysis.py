print("Script started...")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Defining Base Directory

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
IMG_DIR = os.path.join(BASE_DIR, "images")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# 1. Load Dataset
print("Loading dataset...")

iris = sns.load_dataset('iris')

# Save dataset to data folder
if not os.path.exists("data"):
    os.makedirs("data")

iris.to_csv("data/iris.csv", index=False)

# 2. Basic Exploration

print("\nDataset Shape:")
print(iris.shape)

print("\nColumn Names:")
print(iris.columns)

print("\nFirst 5 Rows:")
print(iris.head())

print("\nDataset Info:")
print(iris.info())

print("\nStatistical Summary:")
print(iris.describe())

# 3. Scatter Plot

plt.figure(figsize=(6,4))
sns.scatterplot(x='sepal_length',
                y='petal_length',
                hue='species',
                data=iris)

plt.title("Sepal Length vs Petal Length")
plt.savefig(os.path.join(IMG_DIR, "scatter_plot.png"))
plt.show()

# 4. Histograms


iris.hist(figsize=(8,6))
plt.suptitle("Feature Distributions")
plt.savefig(os.path.join(IMG_DIR, "histograms.png"))
plt.show()

# 5. Box Plot

plt.figure(figsize=(8,5))
sns.boxplot(data=iris)
plt.title("Box Plot of Iris Features")
plt.savefig(os.path.join(IMG_DIR, "boxplot.png"))
plt.show()

# 6. Pairplot (Advanced)

sns.pairplot(iris, hue='species')
plt.savefig(os.path.join(IMG_DIR, "pairplot.png"))
plt.show()

print("\nAnalysis Complete. Plots saved in images folder.")

# ------------------------------
# 7. Correlation Heatmap
# ------------------------------

print("Generating correlation heatmap...")

plt.figure(figsize=(6,4))
sns.heatmap(iris.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig(os.path.join(IMG_DIR, "correlation_heatmap.png"))
plt.show()

print("Heatmap done")