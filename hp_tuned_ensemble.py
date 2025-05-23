# # import pandas as pd
# # import numpy as np
# # from sklearn.model_selection import KFold
# # from scipy.stats import spearmanr
# # import xgboost as xgb
# # import lightgbm as lgb
# # from catboost import CatBoostClassifier
# # import gc

# # # Configuration
# # DATA_PATH = r"/home/amma/Documents/stain/contest_main/data"
# # SEED = 42
# # N_FOLDS = 5
# # NAN_VALUE = -127
# # TARGET = "modben"


# # def load_and_preprocess():
# #     train_feat = pd.read_csv(f"{DATA_PATH}/train_features.csv")
# #     outcomes = pd.read_csv(f"{DATA_PATH}/train_outcomes_functional.csv")
# #     metadata = pd.read_csv(f"{DATA_PATH}/metadata.csv")
# #     test_feat = pd.read_csv(f"{DATA_PATH}/test_features.csv")

# #     outcomes = outcomes[outcomes['time'] == 26].drop(columns=['time'])
# #     week1_cols = [c for c in train_feat.columns if c.endswith('01') or c == 'PID']
# #     train_feat = train_feat[week1_cols]
# #     test_feat = test_feat[week1_cols]

# #     full_train = train_feat.merge(metadata, on='PID').merge(outcomes, on='PID')
# #     test_data = test_feat.merge(metadata, on='PID')

# #     full_train.replace(['ND', 'NaN', np.nan], NAN_VALUE, inplace=True)
# #     test_data.replace(['ND', 'NaN', np.nan], NAN_VALUE, inplace=True)

# #     for df in [full_train, test_data]:
# #         for col in df.select_dtypes(include='object').columns:
# #             df[col] = df[col].astype('category')

# #     return full_train, test_data


# # def prepare_lgb_data(df, target_col=None):
# #     if target_col:
# #         y = df[target_col]
# #         X = df.drop(columns=[target_col, 'PID'])
# #     else:
# #         y = None
# #         X = df.drop(columns=['PID'])
# #     return X, y


# # def train_final_ensemble():
# #     train_df, test_df = load_and_preprocess()
# #     train_df = train_df[train_df[TARGET].between(1, 7)].copy()
# #     train_df[TARGET] -= 1

# #     xgb_oof = np.zeros(len(train_df))
# #     lgb_oof = np.zeros(len(train_df))
# #     cat_oof = np.zeros(len(train_df))
# #     xgb_test_preds = np.zeros(len(test_df))
# #     lgb_test_preds = np.zeros(len(test_df))
# #     cat_test_preds = np.zeros(len(test_df))

# #     kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# #     # Predefined tuned hyperparameters
# #     best_xgb_params = {
# #         'learning_rate': 0.1920247721178956,
# #         'max_depth': 3,
# #         'min_child_weight': 1,
# #         'subsample': 0.6949549276554778,
# #         'colsample_bytree': 0.659117757260395,
# #         'gamma': 0.015120746132691234,
# #         'lambda': 3.46804277546913,
# #         'alpha': 0.9906152112060733,
# #         'objective': 'multi:softprob',
# #         'num_class': 7,
# #         'eval_metric': 'mlogloss',
# #         'tree_method': 'hist',
# #         'device': 'cuda',
# #         'seed': SEED
# #     }

# #     best_lgb_params = {
# #         'learning_rate': 0.18792971616282866,
# #         'max_depth': 3,
# #         'subsample': 0.8472317407359186,
# #         'colsample_bytree': 0.762107533514624,
# #         'objective': 'multiclass',
# #         'num_class': 7,
# #         'n_estimators': 1000,
# #         'random_state': SEED,
# #         'device': 'gpu'
# #     }

# #     best_cat_params = {
# #         'learning_rate': 0.08612208241623023,
# #         'depth': 6,
# #         'iterations': 1000,
# #         'loss_function': 'MultiClass',
# #         'eval_metric': 'MultiClass',
# #         'task_type': 'GPU',
# #         'random_seed': SEED,
# #         'verbose': False,
# #         'early_stopping_rounds': 50
# #     }

# #     for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
# #         X_train = train_df.iloc[train_idx].drop(columns=[TARGET, 'PID'])
# #         y_train = train_df.iloc[train_idx][TARGET]
# #         X_val = train_df.iloc[val_idx].drop(columns=[TARGET, 'PID'])
# #         y_val = train_df.iloc[val_idx][TARGET]
# #         X_test = test_df.drop(columns=['PID'])

# #         # XGBoost
# #         dtrain = xgb.DMatrix(X_train, y_train, enable_categorical=True)
# #         dval = xgb.DMatrix(X_val, y_val, enable_categorical=True)
# #         dtest = xgb.DMatrix(X_test, enable_categorical=True)
# #         xgb_model = xgb.train(best_xgb_params, dtrain, num_boost_round=1000, evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=False)
# #         xgb_oof[val_idx] = xgb_model.predict(dval).argmax(axis=1)
# #         xgb_test_preds += xgb_model.predict(dtest).argmax(axis=1) / N_FOLDS
# #         del xgb_model; gc.collect()

# #         # LightGBM
# #         X_train_lgb, y_train_lgb = prepare_lgb_data(train_df.iloc[train_idx], TARGET)
# #         X_val_lgb, y_val_lgb = prepare_lgb_data(train_df.iloc[val_idx], TARGET)
# #         X_test_lgb, _ = prepare_lgb_data(test_df)
# #         lgb_model = lgb.LGBMClassifier(**best_lgb_params)
# #         lgb_model.fit(X_train_lgb, y_train_lgb, eval_set=[(X_val_lgb, y_val_lgb)], eval_metric='multi_logloss', callbacks=[lgb.early_stopping(50, verbose=False)])
# #         lgb_oof[val_idx] = lgb_model.predict(X_val_lgb)
# #         lgb_test_preds += lgb_model.predict(X_test_lgb) / N_FOLDS
# #         del lgb_model; gc.collect()

# #         # CatBoost
# #         cat_model = CatBoostClassifier(**best_cat_params)
# #         cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=np.where(X_train.dtypes == 'category')[0])
# #         cat_oof[val_idx] = cat_model.predict(X_val).astype(int).flatten()
# #         cat_test_preds += cat_model.predict(X_test).astype(int).flatten() / N_FOLDS
# #         del cat_model; gc.collect()

# #     oof_ensemble = (xgb_oof + lgb_oof + cat_oof) / 3
# #     test_ensemble = (xgb_test_preds + lgb_test_preds + cat_test_preds) / 3

# #     score = spearmanr(train_df[TARGET], oof_ensemble).correlation
# #     print(f"Ensemble CV Spearman Correlation: {score:.4f}")

# #     template = pd.read_csv(f"{DATA_PATH}/test_outcomes_Fun_template_update.csv")
# #     test_df['modben'] = test_ensemble.round().astype(int) + 1
# #     submission = template[['PID']].merge(test_df[['PID', 'modben']], on='PID', how='left')
# #     submission['modben'] = submission['modben'].fillna(1).astype(int)

# #     assert len(submission) == 118
# #     assert list(submission.columns) == ['PID', 'modben']
# #     assert submission['modben'].between(1, 7).all()

# #     submission.to_csv("submission_hp_ensemble.csv", index=False)
# #     print("Submission saved as submission_final_ensemble.csv")
# #     return submission


# # if __name__ == "__main__":
# #     train_final_ensemble()







# import pandas as pd
# import numpy as np
# from sklearn.model_selection import KFold
# from scipy.stats import spearmanr
# import xgboost as xgb
# import lightgbm as lgb
# from catboost import CatBoostClassifier
# import gc

# # Configuration
# DATA_PATH = r"/home/amma/Documents/stain/contest_main/data"
# SEED = 42
# N_FOLDS = 10
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

#     for df in [full_train, test_data]:
#         for col in df.select_dtypes(include='object').columns:
#             df[col] = df[col].astype('category')

#     return full_train, test_data


# def prepare_lgb_data(df, target_col=None):
#     if target_col:
#         y = df[target_col]
#         X = df.drop(columns=[target_col, 'PID'])
#     else:
#         y = None
#         X = df.drop(columns=['PID'])
#     return X, y


# def train_final_ensemble():
#     train_df, test_df = load_and_preprocess()
#     train_df = train_df[train_df[TARGET].between(1, 7)].copy()
#     train_df[TARGET] -= 1

#     xgb_oof = np.zeros(len(train_df))
#     lgb_oof = np.zeros(len(train_df))
#     cat_oof = np.zeros(len(train_df))
#     xgb_test_preds = np.zeros(len(test_df))
#     lgb_test_preds = np.zeros(len(test_df))
#     cat_test_preds = np.zeros(len(test_df))

#     kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

#     # Predefined tuned hyperparameters
#     best_xgb_params = {
#         'learning_rate': 0.1920247721178956,
#         'max_depth': 3,
#         'min_child_weight': 1,
#         'subsample': 0.6949549276554778,
#         'colsample_bytree': 0.659117757260395,
#         'gamma': 0.015120746132691234,
#         'lambda': 3.46804277546913,
#         'alpha': 0.9906152112060733,
#         'objective': 'multi:softprob',
#         'num_class': 7,
#         'eval_metric': 'mlogloss',
#         'tree_method': 'hist',
#         'device': 'cuda',
#         'seed': SEED
#     }

#     best_lgb_params = {
#         'learning_rate': 0.18792971616282866,
#         'max_depth': 3,
#         'subsample': 0.8472317407359186,
#         'colsample_bytree': 0.762107533514624,
#         'objective': 'multiclass',
#         'num_class': 7,
#         'n_estimators': 1000,
#         'random_state': SEED,
#         'device': 'gpu'
#     }

#     best_cat_params = {
#         'learning_rate': 0.08612208241623023,
#         'depth': 6,
#         'iterations': 1000,
#         'loss_function': 'MultiClass',
#         'eval_metric': 'MultiClass',
#         'task_type': 'GPU',
#         'random_seed': SEED,
#         'verbose': False,
#         'early_stopping_rounds': 50
#     }

#     for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
#         X_train = train_df.iloc[train_idx].drop(columns=[TARGET, 'PID'])
#         y_train = train_df.iloc[train_idx][TARGET]
#         X_val = train_df.iloc[val_idx].drop(columns=[TARGET, 'PID'])
#         y_val = train_df.iloc[val_idx][TARGET]
#         X_test = test_df.drop(columns=['PID'])

#         # XGBoost
#         dtrain = xgb.DMatrix(X_train, y_train, enable_categorical=True)
#         dval = xgb.DMatrix(X_val, y_val, enable_categorical=True)
#         dtest = xgb.DMatrix(X_test, enable_categorical=True)
#         xgb_model = xgb.train(best_xgb_params, dtrain, num_boost_round=1000, evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=False)
#         xgb_oof[val_idx] = xgb_model.predict(dval).argmax(axis=1)
#         xgb_test_preds += xgb_model.predict(dtest).argmax(axis=1) / N_FOLDS
#         del xgb_model; gc.collect()

#         # LightGBM
#         X_train_lgb, y_train_lgb = prepare_lgb_data(train_df.iloc[train_idx], TARGET)
#         X_val_lgb, y_val_lgb = prepare_lgb_data(train_df.iloc[val_idx], TARGET)
#         X_test_lgb, _ = prepare_lgb_data(test_df)
#         lgb_model = lgb.LGBMClassifier(**best_lgb_params)
#         lgb_model.fit(X_train_lgb, y_train_lgb, eval_set=[(X_val_lgb, y_val_lgb)], eval_metric='multi_logloss', callbacks=[lgb.early_stopping(50, verbose=False)])
#         lgb_oof[val_idx] = lgb_model.predict(X_val_lgb)
#         lgb_test_preds += lgb_model.predict(X_test_lgb) / N_FOLDS
#         del lgb_model; gc.collect()

#         # CatBoost
#         cat_model = CatBoostClassifier(**best_cat_params)
#         cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=np.where(X_train.dtypes == 'category')[0])
#         cat_oof[val_idx] = cat_model.predict(X_val).astype(int).flatten()
#         cat_test_preds += cat_model.predict(X_test).astype(int).flatten() / N_FOLDS
#         del cat_model; gc.collect()

#     oof_ensemble = (xgb_oof + lgb_oof + cat_oof) / 3
#     test_ensemble = (xgb_test_preds + lgb_test_preds + cat_test_preds) / 3

#     score = spearmanr(train_df[TARGET], oof_ensemble).correlation
#     print(f"Ensemble CV Spearman Correlation: {score:.4f}")

#     template = pd.read_csv(f"{DATA_PATH}/test_outcomes_Fun_template_update.csv")
#     test_df['modben'] = test_ensemble.round().astype(int) + 1
#     submission = template[['PID']].merge(test_df[['PID', 'modben']], on='PID', how='left')
#     submission['modben'] = submission['modben'].fillna(1).astype(int)

#     assert len(submission) == 118
#     assert list(submission.columns) == ['PID', 'modben']
#     assert submission['modben'].between(1, 7).all()

#     submission.to_csv("submission_10fold_ensembled.csv", index=False)
#     print("Submission saved as submission_final_ensemble.csv")
#     return submission


# if __name__ == "__main__":
#     train_final_ensemble()


















import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import gc

# Configuration
DATA_PATH = r"/home/amma/Documents/stain/contest_main/data"
SEED = 42
N_FOLDS = 10
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


def train_final_ensemble():
    train_df, test_df = load_and_preprocess()
    train_df = train_df[train_df[TARGET].between(1, 7)].copy()
    train_df[TARGET] -= 1

    xgb_oof = np.zeros((len(train_df), 7))
    lgb_oof = np.zeros((len(train_df), 7))
    cat_oof = np.zeros((len(train_df), 7))
    xgb_test_preds = np.zeros((len(test_df), 7))
    lgb_test_preds = np.zeros((len(test_df), 7))
    cat_test_preds = np.zeros((len(test_df), 7))

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    best_xgb_params = {
        'learning_rate': 0.1920247721178956,
        'max_depth': 3,
        'min_child_weight': 1,
        'subsample': 0.6949549276554778,
        'colsample_bytree': 0.659117757260395,
        'gamma': 0.015120746132691234,
        'lambda': 3.46804277546913,
        'alpha': 0.9906152112060733,
        'objective': 'multi:softprob',
        'num_class': 7,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'device': 'cuda',
        'seed': SEED
    }

    best_lgb_params = {
        'learning_rate': 0.18792971616282866,
        'max_depth': 3,
        'subsample': 0.8472317407359186,
        'colsample_bytree': 0.762107533514624,
        'objective': 'multiclass',
        'num_class': 7,
        'n_estimators': 1000,
        'random_state': SEED,
        'device': 'gpu'
    }

    best_cat_params = {
        'learning_rate': 0.08612208241623023,
        'depth': 6,
        'iterations': 1000,
        'loss_function': 'MultiClass',
        'eval_metric': 'MultiClass',
        'task_type': 'GPU',
        'random_seed': SEED,
        'verbose': False,
        'early_stopping_rounds': 50
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        X_train = train_df.iloc[train_idx].drop(columns=[TARGET, 'PID'])
        y_train = train_df.iloc[train_idx][TARGET]
        X_val = train_df.iloc[val_idx].drop(columns=[TARGET, 'PID'])
        y_val = train_df.iloc[val_idx][TARGET]
        X_test = test_df.drop(columns=['PID'])

        # XGBoost
        dtrain = xgb.DMatrix(X_train, y_train, enable_categorical=True)
        dval = xgb.DMatrix(X_val, y_val, enable_categorical=True)
        dtest = xgb.DMatrix(X_test, enable_categorical=True)
        xgb_model = xgb.train(best_xgb_params, dtrain, num_boost_round=1000, evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=False)
        xgb_oof[val_idx] = xgb_model.predict(dval)
        xgb_test_preds += xgb_model.predict(dtest) / N_FOLDS
        del xgb_model; gc.collect()

        # LightGBM
        X_train_lgb, y_train_lgb = prepare_lgb_data(train_df.iloc[train_idx], TARGET)
        X_val_lgb, y_val_lgb = prepare_lgb_data(train_df.iloc[val_idx], TARGET)
        X_test_lgb, _ = prepare_lgb_data(test_df)
        lgb_model = lgb.LGBMClassifier(**best_lgb_params)
        lgb_model.fit(X_train_lgb, y_train_lgb, eval_set=[(X_val_lgb, y_val_lgb)], eval_metric='multi_logloss', callbacks=[lgb.early_stopping(50, verbose=False)])
        lgb_oof[val_idx] = lgb_model.predict_proba(X_val_lgb)
        lgb_test_preds += lgb_model.predict_proba(X_test_lgb) / N_FOLDS
        del lgb_model; gc.collect()

        # CatBoost
        cat_model = CatBoostClassifier(**best_cat_params)
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=np.where(X_train.dtypes == 'category')[0])
        cat_oof[val_idx] = cat_model.predict_proba(X_val)
        cat_test_preds += cat_model.predict_proba(X_test) / N_FOLDS
        del cat_model; gc.collect()

    final_oof = (xgb_oof + lgb_oof + cat_oof) / 3
    final_test = (xgb_test_preds + lgb_test_preds + cat_test_preds) / 3

    oof_preds = final_oof.argmax(axis=1)
    test_preds = final_test.argmax(axis=1)

    score = spearmanr(train_df[TARGET], oof_preds).correlation
    print(f"Ensemble CV Spearman Correlation: {score:.4f}")

    template = pd.read_csv(f"{DATA_PATH}/test_outcomes_Fun_template_update.csv")
    test_df['modben'] = test_preds + 1
    submission = template[['PID']].merge(test_df[['PID', 'modben']], on='PID', how='left')
    submission['modben'] = submission['modben'].fillna(1).astype(int)

    assert len(submission) == 118
    assert list(submission.columns) == ['PID', 'modben']
    assert submission['modben'].between(1, 7).all()

    submission.to_csv("submission_e_6.csv", index=False)
    print("Submission saved as submission_final_ensemble.csv")
    return submission


if __name__ == "__main__":
    train_final_ensemble()
