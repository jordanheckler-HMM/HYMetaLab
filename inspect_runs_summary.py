import pandas as pd

# ---------- Load the file ----------
file_path = "runs_summary_merged.csv"  # adjust if in another folder
df = pd.read_csv(file_path)
print("\n✅ File loaded successfully.\n")

# ---------- Basic info ----------
print("Shape:", df.shape)
print("\nColumns:")
for i, col in enumerate(df.columns):
    print(f"{i+1:>3}. {col}")

print("\nFirst 5 rows:")
print(df.head(5).to_string(index=False))

# ---------- Non-zero / non-NaN counts ----------
summary = []
for col in df.columns:
    series = pd.to_numeric(df[col], errors="coerce")
    non_null = series.notna().sum()
    non_zero = (series != 0).sum()
    summary.append((col, non_null, non_zero))
summary_df = pd.DataFrame(summary, columns=["Column", "Non-Null", "Non-Zero"])
print("\n--- Value Presence Summary ---")
print(summary_df.to_string(index=False))

# ---------- Quick suggestions ----------
expected = ["R_mean", "kE", "kI", "n_agents", "openness", "lam"]
missing = [c for c in expected if c not in df.columns]
if missing:
    print("\n⚠️ Missing expected columns:", missing)
else:
    print("\n✅ All expected columns found.")

# ---------- Save column summary for reference ----------
summary_df.to_csv("validation/column_check.csv", index=False)
print("\nSaved: validation/column_check.csv")
