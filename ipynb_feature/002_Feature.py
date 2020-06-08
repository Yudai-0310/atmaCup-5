import pandas as pd
from pathlib import Path
from fastprogress import progress_bar, master_bar

data_dir = Path("../input/")
spectrum_dir = data_dir / "spectrum_raw"

mb = master_bar(["train", "test"])
spectrum_dfs = {}

for phase in mb:
    df = pd.read_csv(data_dir / (phase + ".csv"))
    dfs = []

    for filename in progress_bar(df.spectrum_filename, parent=mb):
        spec = pd.read_csv(
            spectrum_dir / filename,
            sep="\t",
            header=None)
        spec.columns = ["wl", "intensity"]
        spec["spectrum_filename"] = filename
        dfs.append(spec)

    spectrums = pd.concat(dfs, axis=0).reset_index(drop=True)
    spectrum_dfs[phase] = spectrums

from tsfresh import extract_features, extract_relevant_features
from tsfresh.feature_extraction import settings

TEST = True

if TEST:
    spec_train = spectrum_dfs["train"]
    uniq_filenames = spec_train.spectrum_filename.unique()
    df = spec_train[spec_train.spectrum_filename.isin(uniq_filenames)]
else:
    df = spectrum_dfs["train"]

X = extract_features(df, column_id="spectrum_filename", column_sort="wl", n_jobs=8)

train = pd.read_csv(data_dir / "train.csv")

y = df.merge(train, how="left", on="spectrum_filename").set_index("spectrum_filename").target

y = y.groupby("spectrum_filename").mean()

X = extract_relevant_features(df, y, column_id="spectrum_filename", column_sort="wl")

TEST = True

if TEST:
    spec_test = spectrum_dfs["test"]
    uniq_filenames = spec_test.spectrum_filename.unique()
    df_test = spec_test[spec_test.spectrum_filename.isin(uniq_filenames)]
else:
    df_test = spectrum_dfs["test"]

X_test = extract_features(df_test, column_id="spectrum_filename", column_sort="wl", n_jobs=8)

fe = pd.concat([X, X_test[X.columns]], axis = 0)

fe.to_csv('../output/feature.csv')