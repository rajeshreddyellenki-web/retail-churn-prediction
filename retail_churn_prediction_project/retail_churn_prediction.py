"""
Retail Churn Prediction
-----------------------
Predicts which retail customers are likely to stop buying.

How it works:
    1. Pick a cutoff date. Features are built from purchases BEFORE the cutoff.
    2. A customer is labelled churned (1) if they made NO purchase in the
       CHURN_WINDOW_DAYS days AFTER the cutoff, otherwise retained (0).
    3. A model is trained on those labelled customers and evaluated.
    4. Every customer gets a churn probability and a risk level.

Usage:
    python retail_churn_prediction.py --demo
    python retail_churn_prediction.py --transactions transactions.csv

Input file:
    transactions.csv -> customer_id, order_date (YYYY-MM-DD), amount
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------- Settings ----------------
CHURN_WINDOW_DAYS = 90   # no purchase in this many days after cutoff -> churned
RANDOM_STATE = 42


# ---------------- Demo data ----------------
def create_demo_data(n_customers=1500, seed=42):
    """Simulate purchase history. Some customers stop buying part-way through."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=540)
    rows = []

    for cid in range(1, n_customers + 1):
        join = start + pd.Timedelta(days=int(rng.integers(0, 300)))
        loyal = rng.random() > 0.35                     # ~35% will churn
        gap = rng.integers(10, 45) if loyal else rng.integers(20, 90)   # days between orders
        avg_spend = rng.integers(20, 200)
        stop_date = end if loyal else join + pd.Timedelta(days=int(rng.integers(60, 330)))

        date = join
        while date < min(stop_date, end):
            amount = max(5, rng.normal(avg_spend, avg_spend * 0.3))
            rows.append((f"C{cid:04d}", date, round(amount, 2)))
            date += pd.Timedelta(days=int(max(1, rng.normal(gap, gap * 0.3))))

    return pd.DataFrame(rows, columns=["customer_id", "order_date", "amount"])


# ---------------- Feature engineering ----------------
def build_features(transactions, cutoff):
    """RFM-style features using only purchases before the cutoff."""
    past = transactions[transactions["order_date"] < cutoff]
    g = past.groupby("customer_id")

    feats = pd.DataFrame({
        "recency_days": (cutoff - g["order_date"].max()).dt.days,
        "frequency": g.size(),
        "monetary": g["amount"].sum(),
        "avg_order_value": g["amount"].mean(),
        "tenure_days": (cutoff - g["order_date"].min()).dt.days,
    })
    # Average days between orders (tenure / orders), and recent activity
    feats["avg_days_between_orders"] = feats["tenure_days"] / feats["frequency"]
    last90 = past[past["order_date"] >= cutoff - pd.Timedelta(days=90)]
    feats["orders_last_90d"] = last90.groupby("customer_id").size()
    feats["orders_last_90d"] = feats["orders_last_90d"].fillna(0)
    # Is the customer overdue compared with their usual buying rhythm?
    feats["overdue_ratio"] = feats["recency_days"] / feats["avg_days_between_orders"].clip(lower=1)
    return feats


def add_churn_label(feats, transactions, cutoff):
    future = transactions[(transactions["order_date"] >= cutoff) &
                          (transactions["order_date"] < cutoff + pd.Timedelta(days=CHURN_WINDOW_DAYS))]
    active = set(future["customer_id"])
    feats = feats.copy()
    feats["churned"] = (~feats.index.isin(active)).astype(int)
    return feats


# ---------------- Modelling ----------------
def train_and_evaluate(data):
    X = data.drop(columns="churned")
    y = data["churned"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    scaler = StandardScaler().fit(X_train)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8,
                                                class_weight="balanced",
                                                random_state=RANDOM_STATE),
    }

    results = {}
    for name, model in models.items():
        use_scaled = name == "Logistic Regression"
        Xtr = scaler.transform(X_train) if use_scaled else X_train
        Xte = scaler.transform(X_test) if use_scaled else X_test
        model.fit(Xtr, y_train)
        proba = model.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results[name] = {
            "model": model, "scaled": use_scaled,
            "accuracy": accuracy_score(y_test, pred),
            "auc": roc_auc_score(y_test, proba),
            "pred": pred,
        }

    print("\n===== MODEL COMPARISON =====")
    for name, r in results.items():
        print(f"{name:20s} accuracy={r['accuracy']:.3f}  ROC-AUC={r['auc']:.3f}")

    best_name = max(results, key=lambda n: results[n]["auc"])
    best = results[best_name]
    print(f"\nBest model: {best_name}")
    print("\nConfusion matrix (rows = actual, cols = predicted) [retained, churned]")
    print(confusion_matrix(y_test, best["pred"]))
    print("\nClassification report")
    print(classification_report(y_test, best["pred"], target_names=["retained", "churned"]))

    if best_name == "Random Forest":
        imp = pd.Series(best["model"].feature_importances_, index=X.columns)
    else:
        imp = pd.Series(np.abs(best["model"].coef_[0]), index=X.columns)
    print("Feature importance")
    print(imp.sort_values(ascending=False).round(3).to_string())

    return best["model"], best["scaled"], scaler, X.columns


# ---------------- Scoring ----------------
def risk_level(p):
    if p >= 0.7:
        return "HIGH"
    if p >= 0.4:
        return "MEDIUM"
    return "LOW"


def score_current_customers(transactions, model, scaled, scaler, columns, today):
    """Score all customers using their full history up to today."""
    feats = build_features(transactions, today)[columns]
    X = scaler.transform(feats) if scaled else feats
    out = feats.copy()
    out["churn_probability"] = model.predict_proba(X)[:, 1].round(3)
    out["risk_level"] = out["churn_probability"].apply(risk_level)
    out["suggestion"] = out["risk_level"].map({
        "HIGH": "Send win-back offer / personal outreach",
        "MEDIUM": "Send reminder or loyalty reward",
        "LOW": "No action needed",
    })
    return out.sort_values("churn_probability", ascending=False)


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Retail Churn Prediction")
    parser.add_argument("--transactions", help="Path to transactions CSV")
    parser.add_argument("--demo", action="store_true", help="Use generated demo data")
    parser.add_argument("--out", default="churn_predictions.csv", help="Output CSV path")
    args = parser.parse_args()

    if args.demo or not args.transactions:
        print("Using demo data...")
        transactions = create_demo_data()
        transactions.to_csv("transactions.csv", index=False)
    else:
        transactions = pd.read_csv(args.transactions)

    transactions["order_date"] = pd.to_datetime(transactions["order_date"])
    today = transactions["order_date"].max().normalize() + pd.Timedelta(days=1)
    cutoff = today - pd.Timedelta(days=CHURN_WINDOW_DAYS)

    print(f"Transactions: {len(transactions):,} | Customers: {transactions['customer_id'].nunique():,}")
    print(f"Cutoff date: {cutoff.date()} (churn window = {CHURN_WINDOW_DAYS} days)")

    data = add_churn_label(build_features(transactions, cutoff), transactions, cutoff)
    print(f"Churn rate in training data: {data['churned'].mean():.1%}")

    model, scaled, scaler, columns = train_and_evaluate(data)

    scores = score_current_customers(transactions, model, scaled, scaler, columns, today)
    scores.to_csv(args.out)

    print("\n===== CURRENT CUSTOMER RISK =====")
    print(scores["risk_level"].value_counts().to_string())
    print("\nTop 10 customers at risk")
    print(scores[["recency_days", "frequency", "monetary", "churn_probability", "risk_level"]]
          .head(10).to_string())
    print(f"\nFull predictions saved to {args.out}")


if __name__ == "__main__":
    main()
