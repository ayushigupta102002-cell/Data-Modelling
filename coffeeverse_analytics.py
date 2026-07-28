
"""
CoffeeVerse Analytics - End-to-End Data Science Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, confusion_matrix, classification_report)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

N = 5000
rng = np.random.default_rng(42)

cities = ["Delhi","Mumbai","Bengaluru","Hyderabad","Chennai","Pune","Gurgaon","Kolkata"]
products = {
    "Espresso":"Coffee","Cappuccino":"Coffee","Latte":"Coffee","Americano":"Coffee",
    "Mocha":"Coffee","Cold Coffee":"Coffee","Sandwich":"Food","Brownie":"Dessert",
    "Cookie":"Dessert","Tea":"Tea","Green Tea":"Tea","Pasta":"Food"
}
payments = ["UPI","Cash","Credit Card","Debit Card"]
memberships = ["Gold","Silver","Bronze","Non-member"]

# NumPy phase
transaction_ids = np.arange(1, N + 1)
customer_ids = rng.integers(1000, 2000, size=N)
product_names = rng.choice(list(products.keys()), size=N)
quantities = rng.integers(1, 6, size=N)
prices = rng.integers(50, 400, size=N)
discounts = rng.choice([0,5,10,15,20], size=N)
ratings = rng.integers(1, 6, size=N)
city_data = rng.choice(cities, size=N)
membership_data = rng.choice(memberships, size=N)
payment_data = rng.choice(payments, size=N)
purchase_type = rng.choice(["Store", "Delivery"], size=N)
dates = pd.date_range("2026-01-01", periods=N, freq="H")

revenue = quantities * prices * (1 - discounts/100)

print("Average Revenue:", np.mean(revenue))
print("Highest Revenue:", np.max(revenue))
print("Unique Customers:", len(np.unique(customer_ids)))

# Pandas phase
df = pd.DataFrame({
    "TransactionID": transaction_ids,
    "CustomerID": customer_ids,
    "City": city_data,
    "Product": product_names,
    "Category": [products[p] for p in product_names],
    "Quantity": quantities,
    "Price": prices,
    "Discount": discounts,
    "Date": dates,
    "Payment": payment_data,
    "Rating": ratings,
    "Membership": membership_data,
    "PurchaseType": purchase_type,
    "Revenue": revenue
})

# Introduce missing values
df.loc[df.sample(frac=0.01, random_state=1).index, "City"] = np.nan
df.loc[df.sample(frac=0.01, random_state=2).index, "Rating"] = np.nan

# Cleaning
df["City"] = df["City"].fillna(df["City"].mode()[0])
df["Rating"] = df["Rating"].fillna(df["Rating"].median())

# Feature engineering
df["Month"] = df["Date"].dt.month
df["Weekend"] = df["Date"].dt.dayofweek >= 5
purchase_counts = df.groupby("CustomerID")["TransactionID"].transform("count")
df["NumberOfPurchases"] = purchase_counts
df["RepeatCustomer"] = np.where(df["NumberOfPurchases"] > 3, "YES", "NO")

# Save CSV
df.to_csv("cleaned_sales_data.csv", index=False)

# Matplotlib dashboard
plt.figure(figsize=(8,4))
monthly = df.groupby("Month")["Revenue"].sum()
plt.plot(monthly.index, monthly.values, marker="o")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.title("Monthly Revenue")
plt.tight_layout()
plt.savefig("sales_dashboard.png")
plt.close()

# Seaborn heatmap
numeric_df = df[["Quantity","Price","Discount","Rating","Revenue","NumberOfPurchases"]]
plt.figure(figsize=(6,4))
sns.heatmap(numeric_df.corr(numeric_only=True), annot=True)
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

# Regression
X = df[["Quantity","Price","Discount","City","Membership","Rating","Month"]]
y = df["Revenue"]

num_cols = ["Quantity","Price","Discount","Rating","Month"]
cat_cols = ["City","Membership"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

reg_model = Pipeline([
    ("prep", preprocessor),
    ("model", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
reg_model.fit(X_train, y_train)
pred = reg_model.predict(X_test)

print("\nRegression Metrics")
print("MAE:", mean_absolute_error(y_test, pred))
print("RMSE:", mean_squared_error(y_test, pred) ** 0.5)
print("R2:", r2_score(y_test, pred))

# Classification
clf_y = (df["RepeatCustomer"] == "YES").astype(int)
clf = Pipeline([
    ("prep", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, clf_y, test_size=0.2, random_state=42
)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print("\nClassification Metrics")
print("Accuracy:", accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

# Cross validation
scores = cross_val_score(reg_model, X, y, cv=3, scoring="r2")
print("Cross Validation R2:", scores.mean())

# KMeans clustering
cluster_features = df[["Revenue","Quantity","NumberOfPurchases"]]
scaled = StandardScaler().fit_transform(cluster_features)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled)
df["Cluster"] = clusters

# PCA
pca = PCA(n_components=2)
components = pca.fit_transform(scaled)

plt.figure(figsize=(6,4))
plt.scatter(components[:,0], components[:,1], c=clusters)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Customer Segments (PCA)")
plt.tight_layout()
plt.savefig("customer_segments.png")
plt.close()

print("\nGenerated files:")
print("- cleaned_sales_data.csv")
print("- sales_dashboard.png")
print("- correlation_heatmap.png")
print("- customer_segments.png")
