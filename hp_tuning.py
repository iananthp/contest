import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
import gc

# Configuration
DATA_PATH = r"/home/amma/Documents/stain/contest_main/data"
SEED = 42
N_FOLDS = 5
NAN_VALUE = -127
TARGET = "modben"


def load_and_preprocess():
    train_feat = pd.read_csv(f"{DATA_PATH}/train_features.csv")
    outcomes = pd.read_csv(f"{DATA_PATH}/train_outcomes_functional.csv")
    metadata = pd.read_csv(f"{DATA_PATH}/metadata.csv")
    test_feat = pd.read_csv(f"{DATA_PATH}/test_features.csv")

    outcomes = outcomes[outcomes['time'] == 26].drop(columns=['time'])
    week1_cols = [c for c in train_feat.columns if c.endswith('01') or c == 'PID']
    train_feat = train_feat[week1_cols]
    test_feat = test_feat[week1_cols]

    full_train = train_feat.merge(metadata, on='PID').merge(outcomes, on='PID')
    test_data = test_feat.merge(metadata, on='PID')

    full_train.replace(['ND', 'NaN', np.nan], NAN_VALUE, inplace=True)
    test_data.replace(['ND', 'NaN', np.nan], NAN_VALUE, inplace=True)

    for df in [full_train, test_data]:
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype('category')

    return full_train, test_data


def prepare_lgb_data(df, target_col=None):
    if target_col:
        y = df[target_col]
        X = df.drop(columns=[target_col, 'PID'])
    else:
        y = None
        X = df.drop(columns=['PID'])
    return X, y


def tune_lgb_hyperparams():
    def objective(trial):
        train_df, _ = load_and_preprocess()
        train_df = train_df[train_df[TARGET].between(1, 7)].copy()
        train_df[TARGET] -= 1

        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        oof_preds = np.zeros(len(train_df))

        params = {
            'objective': 'multiclass',
            'num_class': 7,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'n_estimators': 1000,
            'random_state': SEED,
            'device': 'gpu'
        }

        for train_idx, val_idx in kf.split(train_df):
            X_train, y_train = prepare_lgb_data(train_df.iloc[train_idx], TARGET)
            X_val, y_val = prepare_lgb_data(train_df.iloc[val_idx], TARGET)

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='multi_logloss',
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            oof_preds[val_idx] = model.predict(X_val)
            del model; gc.collect()

        score = spearmanr(train_df[TARGET], oof_preds).correlation
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, timeout=3600)
    print("Best LightGBM hyperparameters:", study.best_params)
    return study.best_params


def tune_cat_hyperparams():
    def objective(trial):
        train_df, _ = load_and_preprocess()
        train_df = train_df[train_df[TARGET].between(1, 7)].copy()
        train_df[TARGET] -= 1

        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        oof_preds = np.zeros(len(train_df))

        params = {
            'iterations': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'depth': trial.suggest_int('depth', 4, 10),
            'loss_function': 'MultiClass',
            'eval_metric': 'MultiClass',
            'task_type': 'GPU',
            'random_seed': SEED,
            'verbose': False,
            'early_stopping_rounds': 50
        }

        for train_idx, val_idx in kf.split(train_df):
            X_train = train_df.iloc[train_idx].drop(columns=[TARGET, 'PID'])
            y_train = train_df.iloc[train_idx][TARGET]
            X_val = train_df.iloc[val_idx].drop(columns=[TARGET, 'PID'])
            y_val = train_df.iloc[val_idx][TARGET]

            model = CatBoostClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                cat_features=np.where(X_train.dtypes == 'category')[0]
            )
            oof_preds[val_idx] = model.predict(X_val).astype(int).flatten()
            del model; gc.collect()

        score = spearmanr(train_df[TARGET], oof_preds).correlation
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, timeout=3600)
    print("Best CatBoost hyperparameters:", study.best_params)
    return study.best_params


if __name__ == "__main__":
    print("\n--- Tuning LightGBM ---")
    best_lgb_params = tune_lgb_hyperparams()

    print("\n--- Tuning CatBoost ---")
    best_cat_params = tune_cat_hyperparams()
