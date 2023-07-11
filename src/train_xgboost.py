import pickle
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import sklearn
import sklearn.feature_extraction
import sklearn.metrics
import sklearn.preprocessing
import xgboost

from utils.constants import VALID_NAME_CHARS_DICT
from utils.paths import FINAL_PATH
from utils.utils import make_representative

sklearn.set_config(transform_output="pandas")

EncodingMethod = Literal["decimal", "one_hot"]


def encode_name(name: str, max_name_len: int = 30) -> list[int]:
    res = []
    name_len = len(name)
    for i in range(max_name_len + 1):
        if i < name_len:
            res.append(VALID_NAME_CHARS_DICT[name[i]])
        else:
            res.append(0)

    return res


def encode_name_one_hot(name: str, max_name_len: int = 30) -> list[int]:
    res = []
    name_len = len(name)
    for i in range(max_name_len + 1):
        if i < name_len:
            name_c = name[i]
        else:
            name_c = 1  # just a sentinel value

        for c in VALID_NAME_CHARS_DICT:
            res.append(int(c == name_c))

    return res


def process_data(df: pd.DataFrame, name_encoding: EncodingMethod = "one_hot"):
    X = df[
        [
            "first_name",
            "last_name",
            # "pct_asian_zcta",
            # "pct_black_zcta",
            # "pct_hispanic_zcta",
            # "pct_white_zcta",
        ]
    ]
    y = df["race_ethnicity"]

    label_encoder = sklearn.preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y)

    for col in ("first_name", "last_name"):
        df["num_chars"] = df[col].str.len()
        df["num_vowels"] = df[col].str.count(r"[aeiou]")
        df[f"pct_vowels_{col}"] = df["num_vowels"] / df["num_chars"]
        df = df.drop(columns=["num_chars", "num_vowels"])

    if name_encoding == "decimal":
        encode_method = encode_name
    elif name_encoding == "one_hot":
        encode_method = encode_name_one_hot

    fn = pd.DataFrame(df["first_name"].apply(encode_method).to_list())
    fn.columns = [f"fn{col}" for col in fn.columns]

    ln = pd.DataFrame(df["last_name"].apply(encode_method).to_list())
    ln.columns = [f"ln{col}" for col in ln.columns]

    X = pd.concat([fn, ln, X.drop(columns=["first_name", "last_name"])], axis=1)

    # X = category_encoders.TargetEncoder(cols=["first_name", "last_name"]).fit_transform(
    #     X, y
    # )

    return X, y


def train(name_encoding: EncodingMethod = "decimal"):
    df = pl.read_parquet(FINAL_PATH / "fl_imb_deduped_train.parquet").to_pandas()
    test = pl.scan_parquet(FINAL_PATH / "ppp_test_sample.parquet").collect().to_pandas()

    X_train, y_train = process_data(df, name_encoding)
    X_test, y_test = process_data(test, name_encoding)

    # Train the XGBoost model
    xgb_model = xgboost.XGBClassifier(
        gamma=5,
        learning_rate=0.01,
        max_depth=3,
        min_child_weight=500,
        n_estimators=10_000,
        subsample=0.8,
        objective="multi:softprob",
        num_class=4,
        tree_method="gpu_hist",
        early_stopping_rounds=40,
    )
    xgb_model = xgb_model.fit(
        X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True
    )
    xgb_model.save_model(FINAL_PATH / "xgboost1/one_hot.model")

    # Predict on the test set
    y_pred = xgb_model.predict_proba(X_test)
    y_pred_bool = np.argmax(y_pred, axis=1)

    # Calculate accuracy
    report = sklearn.metrics.classification_report(y_test, y_pred_bool)
    print(report)


def f1_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    err = 1 - sklearn.metrics.f1_score(y_true, np.round(y_pred))

    return "f1_err", err


def process_data_binary(df: pd.DataFrame, race: str):
    df["is_race"] = df["race_ethnicity"] == race

    for col in ("first_name", "last_name"):
        df["num_chars"] = df[col].str.len()
        df["num_vowels"] = df[col].str.count(r"[aeiou]")
        df[f"pct_vowels_{col}"] = df["num_vowels"] / df["num_chars"]
        df = df.drop(columns=["num_chars", "num_vowels"])

    fn = pd.DataFrame(df["first_name"].apply(encode_name_one_hot).to_list())
    fn.columns = [f"fn{col}" for col in fn.columns]

    ln = pd.DataFrame(df["last_name"].apply(encode_name_one_hot).to_list())
    ln.columns = [f"ln{col}" for col in ln.columns]

    X = df[
        [
            # "first_name",
            # "last_name",
            "pct_asian_zcta",
            "pct_black_zcta",
            "pct_hispanic_zcta",
            "pct_white_zcta",
            "pct_vowels_first_name",
            "pct_vowels_last_name",
        ]
    ]
    y = df["is_race"]

    X = pd.concat([fn, ln, X], axis=1)

    # X = category_encoders.TargetEncoder(cols=["first_name", "last_name"]).fit_transform(
    #     X, y
    # )

    return X, y


def train_binary(race: str):
    df = make_representative(
        pl.read_parquet(FINAL_PATH / "flz_dups_train.parquet").sample(500_000),
        distribution={"asian": 0.5, "black": 0.5},
    ).to_pandas()
    print(df["race_ethnicity"].value_counts())

    test = (
        pl.scan_parquet(FINAL_PATH / "ppp_test_sample.parquet")
        .filter(pl.col("race_ethnicity").is_in(["black", "asian"]))
        .collect()
        .to_pandas()
    )

    X_train, y_train = process_data_binary(df, race)
    X_test, y_test = process_data_binary(test, race)

    # Train the XGBoost model
    xgb_model = xgboost.XGBClassifier(
        gamma=5,
        learning_rate=0.01,
        max_depth=3,
        min_child_weight=500,
        n_estimators=20_000,
        subsample=0.8,
        objective="binary:logistic",
        tree_method="gpu_hist",
        early_stopping_rounds=40,
    )
    xgb_model = xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=True,
    )
    xgb_model.save_model(FINAL_PATH / f"xgboost2/{race}.model")

    # tst = xgboost.XGBClassifier().load_model(FINAL_PATH / "xgboost1/model.model")

    # Predict on the test set
    y_pred = xgb_model.predict_proba(X_test)
    y_pred_bool = np.argmax(y_pred, axis=1)

    # Calculate accuracy
    report = sklearn.metrics.classification_report(y_test, y_pred_bool)
    print(report)

    test["pred"] = y_pred_bool
    test[["first_name", "last_name", "race_ethnicity", "pred"]].to_csv("temp.csv")

    xgboost.plot_importance(xgb_model)
    plt.show()

    # results = xgb_model.evals_result()
    # plot learning curves
    # plt.plot(results["validation_0"]["logloss"], label="train")
    # plt.plot(results["validation_1"]["logloss"], label="test")


def main():
    train(name_encoding="decimal")
    # train_binary("asian")
    # train_binary("black")
    # train_binary("hispanic")
    # train_binary("white")


if __name__ == "__main__":
    main()
