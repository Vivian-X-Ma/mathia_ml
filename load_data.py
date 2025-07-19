import pandas as pd

# Load all three datasets
df1 = pd.read_csv("student_features_moyo.csv") # moyo features
df2 = pd.read_csv("student_features_aggregated.csv") # my features
df3 = pd.read_csv("features_siqi.csv") # siqi features

print(df1.shape)
print(df2.shape)
print(df3.shape)

combined = pd.merge(
    df1,
    df2,
    on="Anon.Student.Id",
    how="left"
)

combined = pd.merge(
    combined,
    df3,
    on="Anon.Student.Id",
    how="left",
)

combined = combined.fillna(0)

print(f"Final student count: {len(combined)}")  # Should be 557

# Save final dataset
combined.to_csv("final_combined_features.csv", index=False)