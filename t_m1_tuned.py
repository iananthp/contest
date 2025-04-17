# import pandas as pd
# import numpy as np
# from sklearn.model_selection import KFold
# from scipy.stats import spearmanr
# import xgboost as xgb
# import optuna
# import gc

# # Configuration
# DATA_PATH = r"/home/amma/Documents/stain/contest_main/data"
# SEED = 42
# N_FOLDS = 5
# NAN_VALUE = -127
# TARGET = "modben"


# def load_and_preprocess():
#     train_feat = pd.read_csv(f"{DATA_PATH}/train_features.csv")
#     outcomes = pd.read_csv(f"{DATA_PATH}/train_outcomes_functional.csv")
#     metadata = pd.read_csv(f"{DATA_PATH}/metadata.csv")
#     test_feat = pd.read_csv(f"{DATA_PATH}/test_features.csv")

#     outcomes = outcomes[outcomes['time'] == 26].drop(columns=['time'])
#     week1_cols = [c for c in train_feat.columns if c.endswith('01') or c == 'PID']
#     train_feat = train_feat[week1_cols]
#     test_feat = test_feat[week1_cols]

#     full_train = train_feat.merge(metadata, on='PID').merge(outcomes, on='PID')
#     test_data = test_feat.merge(metadata, on='PID')

#     full_train.replace(['ND', 'NaN', np.nan], NAN_VALUE, inplace=True)
#     test_data.replace(['ND', 'NaN', np.nan], NAN_VALUE, inplace=True)

#     return full_train, test_data


# def prepare_dmatrix(df, target_col=None):
#     if target_col:
#         y = df[target_col]
#         X = df.drop(columns=[target_col, 'PID'])
#     else:
#         y = None
#         X = df.drop(columns=['PID'])

#     cat_cols = X.select_dtypes(include='object').columns.tolist()
#     for col in cat_cols:
#         X[col] = X[col].astype('category')

#     return xgb.DMatrix(X, y, enable_categorical=True)


# def objective(trial):
#     train_df, _ = load_and_preprocess()
#     train_df = train_df[train_df[TARGET].between(1, 7)].copy()
#     train_df[TARGET] -= 1

#     kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
#     oof_preds = np.zeros(len(train_df))

#     params = {
#         'objective': 'multi:softprob',
#         'num_class': 7,
#         'eval_metric': 'mlogloss',
#         'tree_method': 'gpu_hist',
#         'predictor': 'gpu_predictor',
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'gamma': trial.suggest_float('gamma', 0, 5.0),
#         'lambda': trial.suggest_float('lambda', 0, 5.0),
#         'alpha': trial.suggest_float('alpha', 0, 5.0),
#         'seed': SEED
#     }

#     for train_idx, val_idx in kf.split(train_df):
#         X_train = prepare_dmatrix(train_df.iloc[train_idx], TARGET)
#         X_val = prepare_dmatrix(train_df.iloc[val_idx], TARGET)

#         model = xgb.train(
#             params,
#             X_train,
#             num_boost_round=1000,
#             evals=[(X_val, 'val')],
#             early_stopping_rounds=50,
#             verbose_eval=False
#         )

#         oof_preds[val_idx] = model.predict(X_val).argmax(axis=1)
#         del model; gc.collect()

#     score = spearmanr(train_df[TARGET], oof_preds).correlation
#     return score


# def tune_hyperparams():
#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective, n_trials=30, timeout=3600)  # 1 hour or 30 trials max
#     print("Best hyperparameters:", study.best_params)
#     return study.best_params


# def train_final_model(best_params):
#     train_df, test_df = load_and_preprocess()
#     train_df = train_df[train_df[TARGET].between(1, 7)].copy()
#     train_df[TARGET] -= 1

#     oof_preds = np.zeros(len(train_df))
#     test_preds = np.zeros(len(test_df))
#     kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

#     best_params.update({
#         'objective': 'multi:softprob',
#         'num_class': 7,
#         'eval_metric': 'mlogloss',
#         'tree_method': 'gpu_hist',
#         'predictor': 'gpu_predictor',
#         'seed': SEED
#     })

#     for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
#         X_train = prepare_dmatrix(train_df.iloc[train_idx], TARGET)
#         X_val = prepare_dmatrix(train_df.iloc[val_idx], TARGET)

#         model = xgb.train(
#             best_params,
#             X_train,
#             num_boost_round=1000,
#             evals=[(X_train, 'train'), (X_val, 'val')],
#             early_stopping_rounds=50,
#             verbose_eval=100
#         )

#         oof_preds[val_idx] = model.predict(X_val).argmax(axis=1)
#         test_preds += model.predict(prepare_dmatrix(test_df)).argmax(axis=1) / N_FOLDS
#         del model; gc.collect()

#     score = spearmanr(train_df[TARGET], oof_preds).correlation
#     print(f"Optimized CV Spearman Correlation: {score:.4f}")

#     template = pd.read_csv(f"{DATA_PATH}/test_outcomes_Fun_template_update.csv")
#     test_df['modben'] = test_preds.round().astype(int) + 1
#     submission = template[['PID']].merge(test_df[['PID', 'modben']], on='PID', how='left')
#     submission['modben'] = submission['modben'].fillna(1).astype(int)

#     assert len(submission) == 118
#     assert list(submission.columns) == ['PID', 'modben']
#     assert submission['modben'].between(1, 7).all()

#     submission.to_csv("submission_tuned_gpu.csv", index=False)
#     print("Submission saved as submission_tuned_gpu.csv")
#     return submission


# if __name__ == "__main__":
#     best_params = tune_hyperparams()
#     train_final_model(best_params)


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
import xgboost as xgb
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

    return full_train, test_data


def prepare_dmatrix(df, target_col=None):
    if target_col:
        y = df[target_col]
        X = df.drop(columns=[target_col, 'PID'])
    else:
        y = None
        X = df.drop(columns=['PID'])

    cat_cols = X.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype('category')

    return xgb.DMatrix(X, y, enable_categorical=True)


def objective(trial):
    train_df, _ = load_and_preprocess()
    train_df = train_df[train_df[TARGET].between(1, 7)].copy()
    train_df[TARGET] -= 1

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(train_df))

    params = {
        'objective': 'multi:softprob',
        'num_class': 7,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'device': 'cuda',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5.0),
        'lambda': trial.suggest_float('lambda', 0, 5.0),
        'alpha': trial.suggest_float('alpha', 0, 5.0),
        'seed': SEED
    }

    for train_idx, val_idx in kf.split(train_df):
        X_train = prepare_dmatrix(train_df.iloc[train_idx], TARGET)
        X_val = prepare_dmatrix(train_df.iloc[val_idx], TARGET)

        model = xgb.train(
            params,
            X_train,
            num_boost_round=1000,
            evals=[(X_val, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        oof_preds[val_idx] = model.predict(X_val).argmax(axis=1)
        del model; gc.collect()

    score = spearmanr(train_df[TARGET], oof_preds).correlation
    return score


def tune_hyperparams():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, timeout=3600)  # 1 hour or 30 trials max
    print("Best hyperparameters:", study.best_params)
    return study.best_params


def train_final_model(best_params):
    train_df, test_df = load_and_preprocess()
    train_df = train_df[train_df[TARGET].between(1, 7)].copy()
    train_df[TARGET] -= 1

    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    best_params.update({
        'objective': 'multi:softprob',
        'num_class': 7,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'device': 'cuda',
        'seed': SEED
    })

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        X_train = prepare_dmatrix(train_df.iloc[train_idx], TARGET)
        X_val = prepare_dmatrix(train_df.iloc[val_idx], TARGET)

        model = xgb.train(
            best_params,
            X_train,
            num_boost_round=1000,
            evals=[(X_train, 'train'), (X_val, 'val')],
            early_stopping_rounds=50,
            verbose_eval=100
        )

        oof_preds[val_idx] = model.predict(X_val).argmax(axis=1)
        test_preds += model.predict(prepare_dmatrix(test_df)).argmax(axis=1) / N_FOLDS
        del model; gc.collect()

    score = spearmanr(train_df[TARGET], oof_preds).correlation
    print(f"Optimized CV Spearman Correlation: {score:.4f}")

    template = pd.read_csv(f"{DATA_PATH}/test_outcomes_Fun_template_update.csv")
    test_df['modben'] = test_preds.round().astype(int) + 1
    submission = template[['PID']].merge(test_df[['PID', 'modben']], on='PID', how='left')
    submission['modben'] = submission['modben'].fillna(1).astype(int)

    assert len(submission) == 118
    assert list(submission.columns) == ['PID', 'modben']
    assert submission['modben'].between(1, 7).all()

    submission.to_csv("submission_tuned_gpu_change_1.csv", index=False)
    print("Submission saved as submission_tuned_gpu.csv")
    return submission


if __name__ == "__main__":
    best_params = tune_hyperparams()
    train_final_model(best_params)
