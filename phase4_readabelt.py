from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from readmepp import ReadMe
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


#1 Define input and output files

# 1 Define input and output files


BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = Path(
    "/Users/nataliakolbina/Documents/workspace/haptic/"
    "CLEAR_original_grouped_split_2026-09-25/"
    "CLEAR_original_test.csv"
)

OUTPUT_CSV = (
    BASE_DIR / "clear_test_readabert_predictions.csv"
)

OUTPUT_SCATTER = (
    BASE_DIR / "clear_test_readabert_scatter.png"
)

OUTPUT_DISTRIBUTION = (
    BASE_DIR / "clear_test_readabert_distribution.png"
)


#2 Load the new held-out CLEAR test set


test = pd.read_csv(
    INPUT_PATH
)

test = test.dropna(
    subset=[
        "Excerpt",
        "BT.Easiness"
    ]
).copy()

test["Excerpt"] = (
    test["Excerpt"]
    .astype(str)
)

test["BT.Easiness"] = (
    test["BT.Easiness"]
    .astype(float)
)

print(
    "Test shape:",
    test.shape
)

print(
    test[
        [
            "Excerpt",
            "BT.Easiness"
        ]
    ].head()
)



#3 Load the pretrained English ReadaBERT model


predictor = ReadMe(
    lang="en"
)

print(
    "\nPretrained ReadaBERT loaded."
)


#4 Generate zero-shot ReadaBERT predictions on CLEAR


tqdm.pandas()

test["readabert_level"] = (
    test["Excerpt"]
    .progress_apply(
        lambda text:
            predictor.predict(
                str(text)
            )
    )
)

print(
    "\nFirst 10 predictions:"
)

print(
    test[
        [
            "Excerpt",
            "BT.Easiness",
            "readabert_level"
        ]
    ].head(10)
)



# 5. Save the pretrained ReadaBERT predictions


test.to_csv(
    OUTPUT_CSV,
    index=False
)

print(
    "\nSaved:",
    OUTPUT_CSV
)



# 6. Compare ReadaBERT levels with CLEAR BT.Easiness


valid = test[
    [
        "BT.Easiness",
        "readabert_level"
    ]
].dropna()

spearman_r, spearman_p = (
    spearmanr(
        valid["BT.Easiness"],
        valid["readabert_level"]
    )
)

pearson_r, pearson_p = (
    pearsonr(
        valid["BT.Easiness"],
        valid["readabert_level"]
    )
)

print(
    "\nZERO-SHOT CORRELATIONS"
)

print(
    "Spearman correlation:",
    spearman_r
)

print(
    "Spearman p-value:",
    spearman_p
)

print(
    "Pearson correlation:",
    pearson_r
)

print(
    "Pearson p-value:",
    pearson_p
)



# 7. Plot ReadaBERT levels against CLEAR BT.Easiness


plt.figure(
    figsize=(7, 5)
)

plt.scatter(
    test["readabert_level"],
    test["BT.Easiness"],
    alpha=0.4
)

plt.xlabel(
    "ReadaBERT Readability Level"
)

plt.ylabel(
    "CLEAR BT.Easiness"
)

plt.title(
    "Pretrained ReadaBERT vs CLEAR BT.Easiness"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_SCATTER,
    dpi=300,
    bbox_inches="tight"
)

plt.close()



# 8. Summarise BT.Easiness within each ReadaBERT level


level_summary = (
    test
    .groupby(
        "readabert_level"
    )["BT.Easiness"]
    .agg(
        [
            "count",
            "mean",
            "median",
            "std"
        ]
    )
    .reset_index()
)

print(
    "\nBT.Easiness by ReadaBERT level:"
)

print(
    level_summary
)



# 9. Examine the distribution of ReadaBERT predictions


level_counts = (
    test[
        "readabert_level"
    ]
    .value_counts()
    .sort_index()
)

print(
    "\nReadaBERT level counts:"
)

print(
    level_counts
)

plt.figure(
    figsize=(7, 5)
)

level_counts.plot(
    kind="bar"
)

plt.xlabel(
    "ReadaBERT Level"
)

plt.ylabel(
    "Number of CLEAR Excerpts"
)

plt.title(
    "Distribution of Pretrained ReadaBERT Predictions on CLEAR"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DISTRIBUTION,
    dpi=300,
    bbox_inches="tight"
)

plt.close()



# 10. Confirm completion of the zero-shot evaluation


print(
    "\nPHASE 4 COMPLETE"
)

