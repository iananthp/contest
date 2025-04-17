import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
import xgboost as xgb
import lightgbm as lgb
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

    # Convert object columns to category
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


def train_ensemble_model(best_params_xgb):
    train_df, test_df = load_and_preprocess()
    train_df = train_df[train_df[TARGET].between(1, 7)].copy()
    train_df[TARGET] -= 1

    xgb_oof = np.zeros(len(train_df))
    lgb_oof = np.zeros(len(train_df))
    xgb_test_preds = np.zeros(len(test_df))
    lgb_test_preds = np.zeros(len(test_df))

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    best_params_xgb.update({
        'objective': 'multi:softprob',
        'num_class': 7,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'device': 'cuda',
        'seed': SEED
    })

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        # XGBoost
        X_train_xgb = train_df.iloc[train_idx].drop(columns=[TARGET, 'PID'])
        y_train_xgb = train_df.iloc[train_idx][TARGET]
        X_val_xgb = train_df.iloc[val_idx].drop(columns=[TARGET, 'PID'])
        y_val_xgb = train_df.iloc[val_idx][TARGET]
        X_test_xgb = test_df.drop(columns=['PID'])

        dtrain = xgb.DMatrix(X_train_xgb, y_train_xgb, enable_categorical=True)
        dval = xgb.DMatrix(X_val_xgb, y_val_xgb, enable_categorical=True)
        dtest = xgb.DMatrix(X_test_xgb, enable_categorical=True)

        xgb_model = xgb.train(
            best_params_xgb,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        xgb_oof[val_idx] = xgb_model.predict(dval).argmax(axis=1)
        xgb_test_preds += xgb_model.predict(dtest).argmax(axis=1) / N_FOLDS
        del xgb_model; gc.collect()

        # LightGBM
        X_train_lgb, y_train_lgb = prepare_lgb_data(train_df.iloc[train_idx], TARGET)
        X_val_lgb, y_val_lgb = prepare_lgb_data(train_df.iloc[val_idx], TARGET)
        X_test_lgb, _ = prepare_lgb_data(test_df)

        lgb_model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=7,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_estimators=1000,
            random_state=SEED,
            device='gpu'
        )

        lgb_model.fit(
            X_train_lgb, y_train_lgb,
            eval_set=[(X_val_lgb, y_val_lgb)],
            eval_metric='multi_logloss',
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        lgb_oof[val_idx] = lgb_model.predict(X_val_lgb)
        lgb_test_preds += lgb_model.predict(X_test_lgb) / N_FOLDS
        del lgb_model; gc.collect()

    # Average predictions
    oof_ensemble = (xgb_oof + lgb_oof) / 2
    test_ensemble = (xgb_test_preds + lgb_test_preds) / 2

    score = spearmanr(train_df[TARGET], oof_ensemble).correlation
    print(f"Ensemble CV Spearman Correlation: {score:.4f}")

    template = pd.read_csv(f"{DATA_PATH}/test_outcomes_Fun_template_update.csv")
    test_df['modben'] = test_ensemble.round().astype(int) + 1
    submission = template[['PID']].merge(test_df[['PID', 'modben']], on='PID', how='left')
    submission['modben'] = submission['modben'].fillna(1).astype(int)

    assert len(submission) == 118
    assert list(submission.columns) == ['PID', 'modben']
    assert submission['modben'].between(1, 7).all()

    submission.to_csv("submission_ensemble_xgb_lgb.csv", index=False)
    print("Submission saved as submission_ensemble_xgb_lgb.csv")
    return submission


if __name__ == "__main__":
    best_params = {
        'learning_rate': 0.1920247721178956,
        'max_depth': 3,
        'min_child_weight': 1,
        'subsample': 0.6949549276554778,
        'colsample_bytree': 0.659117757260395,
        'gamma': 0.015120746132691234,
        'lambda': 3.46804277546913,
        'alpha': 0.9906152112060733
    }
    train_ensemble_model(best_params)