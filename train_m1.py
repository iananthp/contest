# import pandas as pd
# import numpy as np
# from sklearn.model_selection import KFold
# from scipy.stats import spearmanr
# import xgboost as xgb
# import gc

# # Configuration
# DATA_PATH = r"/home/amma/Documents/stain/contest_main/data"
# SEED = 42
# N_FOLDS = 5
# NAN_VALUE = -127
# TARGET = "modben"

# def load_and_preprocess():
#     # Load datasets
#     train_feat = pd.read_csv(f"{DATA_PATH}/train_features.csv")
#     outcomes = pd.read_csv(f"{DATA_PATH}/train_outcomes_functional.csv")
#     metadata = pd.read_csv(f"{DATA_PATH}/metadata.csv")
#     test_feat = pd.read_csv(f"{DATA_PATH}/test_features.csv")
    
#     # Filter outcomes to target timepoint (26 weeks)
#     outcomes = outcomes[outcomes['time'] == 26].drop(columns=['time'])
    
#     # Select week 1 features only (ending with '01')
#     week1_cols = [c for c in train_feat.columns if c.endswith('01') or c == 'PID']
#     train_feat = train_feat[week1_cols]
#     test_feat = test_feat[week1_cols]
    
#     # Merge data
#     full_train = train_feat.merge(metadata, on='PID').merge(outcomes, on='PID')
#     test_data = test_feat.merge(metadata, on='PID')
    
#     # Handle missing values
#     full_train.replace(['ND', 'NaN', np.nan], NAN_VALUE, inplace=True)
#     test_data.replace(['ND', 'NaN', np.nan], NAN_VALUE, inplace=True)
    
#     return full_train, test_data

# def prepare_dmatrix(df, target_col=None):
#     """Convert dataframe to optimized DMatrix format"""
#     if target_col:
#         y = df[target_col]
#         X = df.drop(columns=[target_col, 'PID'])
#     else:
#         y = None
#         X = df.drop(columns=['PID'])
    
#     # Convert categoricals
#     cat_cols = X.select_dtypes(include='object').columns.tolist()
#     for col in cat_cols:
#         X[col] = X[col].astype('category')
    
#     return xgb.DMatrix(X, y, enable_categorical=True)

# def train_xgb_model():
#     # Load and preprocess data
#     train_df, test_df = load_and_preprocess()
    
#     # Remove invalid target values
#     train_df = train_df[train_df[TARGET].between(1, 7)].copy()
#     train_df[TARGET] -= 1  # Convert to 0-6 for XGBoost
    
#     # Initialize storage
#     oof_preds = np.zeros(len(train_df))
#     test_preds = np.zeros(len(test_df))
    
#     # Cross-validation
#     kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
#     for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
#         # Data prep
#         X_train = prepare_dmatrix(train_df.iloc[train_idx], TARGET)
#         X_val = prepare_dmatrix(train_df.iloc[val_idx], TARGET)
        
#         # Model configuration
#         params = {
#             'objective': 'multi:softprob',
#             'num_class': 7,
#             'eval_metric': 'mlogloss',
#             'learning_rate': 0.05,
#             'max_depth': 6,
#             'subsample': 0.8,
#             'colsample_bytree': 0.8,
#             'seed': SEED
#         }
        
#         # Training
#         model = xgb.train(
#             params,
#             X_train,
#             num_boost_round=1000,
#             evals=[(X_train, 'train'), (X_val, 'val')],
#             early_stopping_rounds=50,
#             verbose_eval=100
#         )
        
#         # Predictions
#         oof_preds[val_idx] = model.predict(X_val).argmax(axis=1)
#         test_preds += model.predict(prepare_dmatrix(test_df)).argmax(axis=1) / N_FOLDS
        
#         del model; gc.collect()
    
#     # Calculate validation score
#     score = spearmanr(train_df[TARGET], oof_preds).correlation
#     print(f"CV Spearman Correlation: {score:.4f}")
    
#     # Prepare submission
#     test_df['modben'] = test_preds.round().astype(int) + 1
#     submission = test_df[['PID', 'modben']]
#     submission.to_csv("submission.csv", index=False)
    
#     return submission

# if __name__ == "__main__":
#     train_xgb_model()

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
import xgboost as xgb
import gc

# Configuration
DATA_PATH = r"/home/amma/Documents/stain/contest_main/data"
SEED = 42
N_FOLDS = 5
NAN_VALUE = -127
TARGET = "modben"

def load_and_preprocess():
    # Load datasets
    train_feat = pd.read_csv(f"{DATA_PATH}/train_features.csv")
    outcomes = pd.read_csv(f"{DATA_PATH}/train_outcomes_functional.csv")
    metadata = pd.read_csv(f"{DATA_PATH}/metadata.csv")
    test_feat = pd.read_csv(f"{DATA_PATH}/test_features.csv")
    
    # Filter outcomes to target timepoint (26 weeks)
    outcomes = outcomes[outcomes['time'] == 26].drop(columns=['time'])
    
    # Select week 1 features only (ending with '01')
    week1_cols = [c for c in train_feat.columns if c.endswith('01') or c == 'PID']
    train_feat = train_feat[week1_cols]
    test_feat = test_feat[week1_cols]
    
    # Merge data
    full_train = train_feat.merge(metadata, on='PID').merge(outcomes, on='PID')
    test_data = test_feat.merge(metadata, on='PID')
    
    # Handle missing values
    full_train.replace(['ND', 'NaN', np.nan], NAN_VALUE, inplace=True)
    test_data.replace(['ND', 'NaN', np.nan], NAN_VALUE, inplace=True)
    
    return full_train, test_data

def prepare_dmatrix(df, target_col=None):
    """Convert dataframe to optimized DMatrix format"""
    if target_col:
        y = df[target_col]
        X = df.drop(columns=[target_col, 'PID'])
    else:
        y = None
        X = df.drop(columns=['PID'])
    
    # Convert categoricals
    cat_cols = X.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype('category')
    
    return xgb.DMatrix(X, y, enable_categorical=True)

def train_xgb_model():
    # Load and preprocess data
    train_df, test_df = load_and_preprocess()
    
    # Remove invalid target values
    train_df = train_df[train_df[TARGET].between(1, 7)].copy()
    train_df[TARGET] -= 1  # Convert to 0-6 for XGBoost
    
    # Initialize storage
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    
    # Cross-validation
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        # Data prep
        X_train = prepare_dmatrix(train_df.iloc[train_idx], TARGET)
        X_val = prepare_dmatrix(train_df.iloc[val_idx], TARGET)
        
        # Model configuration
        params = {
            'objective': 'multi:softprob',
            'num_class': 7,
            'eval_metric': 'mlogloss',
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': SEED
        }
        
        # Training
        model = xgb.train(
            params,
            X_train,
            num_boost_round=1000,
            evals=[(X_train, 'train'), (X_val, 'val')],
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        # Predictions
        oof_preds[val_idx] = model.predict(X_val).argmax(axis=1)
        test_preds += model.predict(prepare_dmatrix(test_df)).argmax(axis=1) / N_FOLDS
        
        del model; gc.collect()
    
    # Calculate validation score
    score = spearmanr(train_df[TARGET], oof_preds).correlation
    print(f"CV Spearman Correlation: {score:.4f}")
    
    # Prepare submission using official template
    template = pd.read_csv(f"{DATA_PATH}/test_outcomes_Fun_template_update.csv")
    test_df['modben'] = test_preds.round().astype(int) + 1
    
    # Merge predictions with template
    submission = template[['PID']].merge(test_df[['PID', 'modben']], on='PID', how='left')
    
    # Handle any missing predictions (shouldn't occur if PIDs match)
    submission['modben'] = submission['modben'].fillna(1).astype(int)
    
    # Final validation
    assert len(submission) == 118, f"Submission must have 118 rows, got {len(submission)}"
    assert list(submission.columns) == ['PID', 'modben'], "Invalid column names"
    assert submission['modben'].between(1,7).all(), "Invalid modben values"
    
    submission.to_csv("submission_e_1.csv", index=False)
    print("Submission created with correct 118 rows")
    return submission

if __name__ == "__main__":
    train_xgb_model()