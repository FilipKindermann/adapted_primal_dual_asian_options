######################################################################################
######################################################################################
#
# Description: combines several runs of a test into one aggregated combination file
#
######################################################################################
######################################################################################

import pandas as pd
from pathlib import Path

# folder name for the results
NAME = "roughness"
# column name of the varied parameter
COLUMN = "Hurst parameter"

# finds and combines all results in the folder
files = Path(f"results/rep_{NAME}").glob("*.csv")
df_raw = pd.concat(
    (pd.read_csv(file) for file in files),
    ignore_index=True
)
df = df_raw.drop_duplicates()
if df_raw.shape != df.shape:
    print(f"{df_raw.shape[0] - df.shape[0]} duplicates dropped")

# Define columns that are grouped - rest are aggregated
group_cols = [col for col in df.columns if col not in ["Lower Bound", "Lower Bound Std", "Upper Bound", "Upper Bound Std", "Duality Gap"]]

# creates aggregated results with average bound, se, sd, duality gap
data = (
    df.groupby(
        group_cols,
        as_index=False
    )
    .agg(
        lower_bound=("Lower Bound", "mean"),
        lower_bound_se=("Lower Bound Std", "mean"),
        lower_bound_sd=("Lower Bound", "std"),
        upper_bound=("Upper Bound", "mean"),
        upper_bound_se=("Upper Bound Std", "mean"),
        upper_bound_sd=("Upper Bound", "std"),
    )
)
data["duality_gap"] = (data["upper_bound"]-data["lower_bound"]) / data["upper_bound"]

# creates data frames that will be seperate sheets in the result csv
lower_bound = data.pivot(index=COLUMN, columns="Strike", values="lower_bound")
lower_bound_se = data.pivot(index=COLUMN, columns="Strike", values="lower_bound_se")
lower_bound_sd = data.pivot(index=COLUMN, columns="Strike", values="lower_bound_sd")
upper_bound = data.pivot(index=COLUMN, columns="Strike", values="upper_bound")
upper_bound_se = data.pivot(index=COLUMN, columns="Strike", values="upper_bound_se")
upper_bound_sd = data.pivot(index=COLUMN, columns="Strike", values="upper_bound_sd")
duality_gap = data.pivot(index=COLUMN, columns="Strike", values="duality_gap")

# stores results as csv
with pd.ExcelWriter(f"results/rep_{NAME}/output_{NAME}.xlsx", engine="openpyxl") as writer:
    data.to_excel(writer, sheet_name="data", index=False)
    lower_bound.to_excel(writer, sheet_name="lower_bound")
    lower_bound_se.to_excel(writer, sheet_name="lower_bound_se")
    lower_bound_sd.to_excel(writer, sheet_name="lower_bound_sd")
    upper_bound.to_excel(writer, sheet_name="upper_bound")
    upper_bound_se.to_excel(writer, sheet_name="upper_bound_se")
    upper_bound_sd.to_excel(writer, sheet_name="upper_bound_sd")
    duality_gap.to_excel(writer, sheet_name="duality_gap")