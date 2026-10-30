from pathlib import Path
import pandas as pd

p = Path("reports/research/spxw_0dte_iron_fly/2026-10-27/trades.csv")
df = pd.read_csv(p)
print(df.columns.tolist())
print(df.head(2).to_string())
print(df["fill"].value_counts().to_dict())
print("n", len(df))
