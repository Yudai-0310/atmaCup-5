import os
import pathlib
import string
import pylab as pl
import missingno as msno
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import lightgbm as lgb
import pandas_profiling as pdp
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.metrics import average_precision_score
from IPython.display import display
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm
pd.set_option('display.max_columns', 100)

file_path = '../input/'


# train, test と spectrum　を結合
train = pd.read_csv(file_path + 'train.csv')
test =  pd.read_csv(file_path + 'test.csv')
fitting = pd.read_csv(file_path + 'fitting.csv')

train = pd.merge(train, fitting, on="spectrum_id", how="left")
test = pd.merge(test, fitting, on="spectrum_id", how="left")

p_temp = pathlib.Path('../input/spectrum_raw')

spec = []
for file in p_temp.iterdir():
    spec_df = pd.read_csv(file, sep='\t', header=None)
    spec_df.columns = ["wavelength", "intensity"]
    spec_df["spectrum_filename"] = file.stem + ".dat"
    spec.append(spec_df)

spec_df = pd.concat(spec, axis=0)

spec_agg = spec_df.groupby("spectrum_filename")["intensity"].agg(["max", "min", "mean", "std"])
spec_agg.columns = ["intensity_" + c for c in spec_agg.columns]

train = pd.merge(train, spec_agg.reset_index(), on="spectrum_filename", how="left")
test = pd.merge(test, spec_agg.reset_index(), on="spectrum_filename", how="left")

print(f'train shape: {train.shape}')
print(f'test shape: {test.shape}')


# カテゴリエンコーディング
categorical_columns = [
    'exc_wl',
    'layout_a',
]

for col in categorical_columns:
    le = LabelEncoder()
    train[col] = le.fit_transform(list(train[col].values))

for col in categorical_columns:
    le = LabelEncoder()
    test[col] = le.fit_transform(list(test[col].values))


# 波形から作成した大量の特徴量をくっつける
feature = pd.read_csv('../output/feature.csv')
feature = feature.rename(columns={'id': 'spectrum_filename'})

train = train.merge(feature, how="left", on="spectrum_filename")
test = test.merge(feature, how="left", on="spectrum_filename")


# 列名を変更
train_col = list(train.columns)
for i, j in enumerate(train_col):
    train_col[i] = j.translate(str.maketrans( '', '',string.punctuation))

test_col = list(test.columns)
for i, j in enumerate(test_col):
    test_col[i] = j.translate(str.maketrans( '', '',string.punctuation))
    
train.columns = train_col
test.columns = test_col


# 学習
categorical_columns = [
    'excwl',
    'layouta'
]

useless_columns_train = [
    'spectrumid',
    'spectrumfilename',
    'chipid',
    'target'
]

useless_columns_test = [
    'spectrumid',
    'spectrumfilename',
    'chipid'
]

lgb_model_params = {
    'objective': 'binary',
    'metric': 'binary',
    'boosting': 'gbdt',
    'max_depth': 5,
    'num_leaves': 50,
    'learning_rate': 0.1,
    'colsample_bytree': 0.7,
    'subsample': 0.7,
    'subsample_freq': 1,
    'seed': 71,
    'bagging_seed': 71,
    'feature_fraction_seed': 71,
    'drop_seed': 71,
    'verbose': -1
}

lgb_train_params = {
    'num_boost_round': 1000,
    'early_stopping_rounds': 100,
    'verbose_eval': 50
}

def pr_auc(preds, data):
    y_true = data.get_label()
    score = average_precision_score(y_true, preds)
    return "pr_auc", score, True

X = train.drop(columns=useless_columns_train, axis=1)
y = train['target']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = []
oof_pred = np.zeros_like(y, dtype=np.float)

for i, (idx_train, idx_valid) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
    X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]
    
    train_lgb = lgb.Dataset(X_train, label=y_train)
    valid_lgb = lgb.Dataset(X_valid, label=y_valid)

    model = lgb.train(
        lgb_model_params,
        train_lgb,
        valid_sets=[train_lgb, valid_lgb],
        valid_names=['train', 'valid'],
        categorical_feature=categorical_columns,
        **lgb_train_params,
        feval=pr_auc
    )

    pred_valid = model.predict(X_valid)
    oof_pred[idx_valid] = pred_valid
    models.append(model)

    print(f'Fold {i+1} PR-AUC: {average_precision_score(y_valid, pred_valid):.4f}')
    print("")

score = average_precision_score(y, oof_pred)
print("")
print('FINISHED score: {:.4f}'.format(score))

categorical_columns = [
    'excwl',
    'layouta'
]

useless_columns_train = [
    'spectrumid',
    'spectrumfilename',
    'chipid',
    'target'
]

useless_columns_test = [
    'spectrumid',
    'spectrumfilename',
    'chipid'
]

lgb_model_params = {
    'objective': 'binary',
    'metric': 'binary',
    'boosting': 'gbdt',
    'max_depth': 5,
    'num_leaves': 50,
    'learning_rate': 0.1,
    'colsample_bytree': 0.7,
    'subsample': 0.7,
    'subsample_freq': 1,
    'seed': 71,
    'bagging_seed': 71,
    'feature_fraction_seed': 71,
    'drop_seed': 71,
    'verbose': -1
}

lgb_train_params = {
    'num_boost_round': 1000,
    'early_stopping_rounds': 100,
    'verbose_eval': 50
}

def pr_auc(preds, data):
    y_true = data.get_label()
    score = average_precision_score(y_true, preds)
    return "pr_auc", score, True

X = train.drop(columns=useless_columns_train, axis=1)
y = train['target']

models = []
score_random_average = []
oof_pred = np.zeros_like(y, dtype=np.float)

for j in tqdm(range(0,90)):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=j)

    for i, (idx_train, idx_valid) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
        X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]

        train_lgb = lgb.Dataset(X_train, label=y_train)
        valid_lgb = lgb.Dataset(X_valid, label=y_valid)

        model = lgb.train(
            lgb_model_params,
            train_lgb,
            valid_sets=[train_lgb, valid_lgb],
            valid_names=['train', 'valid'],
            categorical_feature=categorical_columns,
            **lgb_train_params,
            feval=pr_auc
        )

        pred_valid = model.predict(X_valid)
        oof_pred[idx_valid] = pred_valid
        models.append(model)
        
        print("")

    score = average_precision_score(y, oof_pred)
    score_random_average.append(score)

print('正答率の平均：', np.mean(score_random_average))
print('正答率の標準偏差', np.std(score_random_average))


# サブミット
pred_test = np.array([model.predict(test.drop(columns=useless_columns_test, axis=1)) for model in models])
pred_test = np.mean(pred_test, axis=0)

submission = pd.read_csv(file_path + 'atmaCup5__sample_submission.csv')
submission['target'] = pred_test

submission.to_csv('../output/0604_1.csv', index=False)


# LB:0.8746
