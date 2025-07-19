import pandas as pd
from datetime import datetime

# Load data
df_log = pd.read_csv("training_set_with_formatted_time.csv")

# Convert datetime
df_log["timestamp"] = pd.to_datetime(df_log["datetime"], format="mixed")

# Sort by student and timestamp
df_log = df_log.sort_values(["Anon.Student.Id", "Problem.Name", "timestamp"])

# --- Feature Engineering ---
# 1. Time spent per problem
time_per_problem = df_log.groupby(["Anon.Student.Id", "Problem.Name"])[
    "timestamp"
].apply(lambda x: (x.max() - x.min()).total_seconds())
time_per_problem.name = "Time_Per_Problem"

# 2. Hint counts by level per problem
hint_counts = (
    df_log[df_log["Action"] == "Hint Request"]
    .groupby(["Anon.Student.Id", "Problem.Name", "Help.Level"])
    .size()
    .unstack(fill_value=0)
    .add_prefix("Level")
    .add_suffix("_Hints")
)

# 3. Hint frequency (hints per minute)
hint_freq = hint_counts.sum(axis=1) / (time_per_problem / 60)  # hints per minute
hint_freq.name = "Hint_Request_Frequency"

# 4. Combine features at PROBLEM level
problem_features = pd.concat(
    [hint_counts, hint_freq, time_per_problem], axis=1
).reset_index()

# --- AGGREGATE TO STUDENT LEVEL ---
student_features = problem_features.groupby("Anon.Student.Id").agg(
    {
        "Level1_Hints": "mean",
        "Level2_Hints": "mean",
        "Level3_Hints": "mean",
        "Hint_Request_Frequency": "mean",
        "Time_Per_Problem": ["mean", "median"],  # Both mean and median time
    }
)

# Flatten column names
student_features.columns = [
    "Level1_Hints_Per_Problem",
    "Level2_Hints_Per_Problem",
    "Level3_Hints_Per_Problem",
    "Hint_Request_Frequency",
    "Avg_Time_Per_Problem",
    "Median_Time_Per_Problem",
]

# --- Add Additional Features ---
# Total problems attempted
student_features["Total_Problems_Attempted"] = problem_features.groupby(
    "Anon.Student.Id"
)["Problem.Name"].nunique()

# Total hint requests
student_features["Total_Hint_Requests"] = (
    problem_features.groupby("Anon.Student.Id")[
        ["Level1_Hints", "Level2_Hints", "Level3_Hints"]
    ]
    .sum()
    .sum(axis=1)
)

# Save
student_features.to_csv("student_features_aggregated.csv")
print(f"Aggregated features for {len(student_features)} unique students")
