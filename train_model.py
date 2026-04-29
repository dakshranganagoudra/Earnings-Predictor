import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from xgboost import XGBClassifier
from collections import Counter
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────

def load_data():
    df = pd.read_csv('data/cleaned/feature_matrix.csv', parse_dates=['earnings_date'])
    df = df.sort_values('earnings_date').reset_index(drop=True)

    feature_cols = [
        'prev_surprise_1q', 'avg_surprise_4q', 'beat_streak', 'surprise_consistency',
        'eps_estimate_revision_pct', 'num_analysts',
        'revenue_growth', 'gross_margin', 'revenue_growth_accel', 'gross_margin_trend',
        'price_return_30d', 'price_return_5d', 'price_volatility_30d', 'pct_from_52w_high',
        'fiscal_quarter', 'ticker_encoded'
    ]

    X = df[feature_cols]
    y = df['label']
    return df, X, y, feature_cols


# ─────────────────────────────────────────
# TRAIN / TEST SPLIT (TIME BASED)
# ─────────────────────────────────────────

def time_split(X, y, df, split_ratio=0.8):
    split_idx = int(len(df) * split_ratio)
    split_date = df.iloc[split_idx]['earnings_date']
    print(f"Training on data before: {split_date.date()}")
    print(f"Train size: {split_idx} | Test size: {len(df) - split_idx}")
    return (
        X.iloc[:split_idx], X.iloc[split_idx:],
        y.iloc[:split_idx], y.iloc[split_idx:]
    )


# ─────────────────────────────────────────
# HANDLE CLASS IMBALANCE
# ─────────────────────────────────────────

def get_sample_weights(y_encoded):
    counter = Counter(y_encoded)
    total = sum(counter.values())
    class_weights = {cls: total / (len(counter) * count) for cls, count in counter.items()}
    print(f"Class weights: {class_weights}")
    return np.array([class_weights[c] for c in y_encoded])


# ─────────────────────────────────────────
# TRAIN ALL 3 MODELS
# ─────────────────────────────────────────

def train_models(X_train, X_test, y_train_enc, y_test_enc, sample_weights):
    models = {}

    # Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train, y_train_enc)
    models['Logistic Regression'] = (lr, lr.predict(X_test))

    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=6,
                                 class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train_enc)
    models['Random Forest'] = (rf, rf.predict(X_test))

    # XGBoost
    print("Training XGBoost...")
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         use_label_encoder=False, eval_metric='mlogloss',
                         random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train_enc, sample_weight=sample_weights,
            eval_set=[(X_test, y_test_enc)], verbose=False)
    models['XGBoost'] = (xgb, xgb.predict(X_test))

    return models


# ─────────────────────────────────────────
# COMPARE MODEL RESULTS
# ─────────────────────────────────────────

def compare_models(models, y_test_enc, le):
    print("\n" + "="*50)
    results = {}
    for name, (model, preds) in models.items():
        acc = accuracy_score(y_test_enc, preds)
        f1 = f1_score(y_test_enc, preds, average='weighted')
        results[name] = {'Accuracy': round(acc, 3), 'F1 (weighted)': round(f1, 3)}
        print(f"\n=== {name} ===")
        print(classification_report(y_test_enc, preds, target_names=le.classes_))

    print("\n📊 Summary:")
    print(pd.DataFrame(results).T)
    return results


# ─────────────────────────────────────────
# PLOT CONFUSION MATRIX
# ─────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, model_name, le):
    os.makedirs('outputs', exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'{model_name} — Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'outputs/{model_name.replace(" ", "_")}_confusion_matrix.png', dpi=150)
    plt.close()
    print(f"✅ Saved confusion matrix for {model_name}")


# ─────────────────────────────────────────
# TUNE XGBOOST
# ─────────────────────────────────────────

def tune_xgboost(X_train, y_train_enc, sample_weights):
    print("\n🔧 Tuning XGBoost with RandomizedSearchCV...")
    tscv = TimeSeriesSplit(n_splits=5)

    param_grid = {
        'n_estimators': [200, 300, 400],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9]
    }

    search = RandomizedSearchCV(
        XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        param_distributions=param_grid,
        n_iter=15,
        cv=tscv,
        scoring='f1_weighted',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train, y_train_enc, sample_weight=sample_weights)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV score: {search.best_score_:.3f}")
    return search.best_estimator_


# ─────────────────────────────────────────
# PLOT FEATURE IMPORTANCE
# ─────────────────────────────────────────

def plot_feature_importance(model, feature_cols):
    os.makedirs('outputs', exist_ok=True)
    feat_list = list(feature_cols)
    imp_list = list(model.feature_importances_)
    min_len = min(len(feat_list), len(imp_list))
    importance = pd.DataFrame({
        'feature': feat_list[:min_len],
        'importance': imp_list[:min_len]
    }).sort_values('importance', ascending=True)
    plt.figure(figsize=(8, 7))
    median_imp = importance['importance'].median()
    colors = ['#2ecc71' if imp > median_imp else '#95a5a6'
              for imp in importance['importance']]
    plt.barh(importance['feature'], importance['importance'], color=colors)
    plt.xlabel('Feature Importance Score')
    plt.title('XGBoost — What Drives Earnings Surprise Predictions?')
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png', dpi=150)
    plt.close()
    print("✅ Saved feature importance chart")

    print("\nTop features:")
    print(importance.sort_values('importance', ascending=False).head(8).to_string(index=False))


# ─────────────────────────────────────────
# RUN EVERYTHING
# ─────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    print("📥 Loading feature matrix...")
    df, X, y, feature_cols = load_data()
    print(f"Dataset shape: {X.shape}")
    print(f"Label distribution:\n{y.value_counts()}")

    # Split
    X_train, X_test, y_train, y_test = time_split(X, y, df)

    # Fill missing values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    joblib.dump(imputer, 'models/imputer.pkl')

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    print(f"\nClasses: {le.classes_}")

    # Sample weights
    sample_weights = get_sample_weights(y_train_enc)

    # Train models
    print("\n🤖 Training models...")
    models = train_models(X_train, X_test, y_train_enc, y_test_enc, sample_weights)

    # Compare
    compare_models(models, y_test_enc, le)

    # Plot confusion matrices
    for name, (model, preds) in models.items():
        plot_confusion_matrix(y_test_enc, preds, name, le)

   # Use the XGBoost model that already trained
    best_xgb = models['XGBoost'][0]
    best_preds = models['XGBoost'][1]
    print(f"\nXGBoost Accuracy: {accuracy_score(y_test_enc, best_preds):.3f}")
    print(f"XGBoost F1: {f1_score(y_test_enc, best_preds, average='weighted'):.3f}")

    # Feature importance
    plot_feature_importance(best_xgb, list(feature_cols))

        # Save everything
    joblib.dump(best_xgb, 'models/xgb_earnings_predictor.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    joblib.dump(feature_cols, 'models/feature_cols.pkl')

    print("\n✅ All models and charts saved!")
    print("  models/xgb_earnings_predictor.pkl")
    print("  models/label_encoder.pkl")
    print("  models/feature_cols.pkl")
    print("  outputs/feature_importance.png")
