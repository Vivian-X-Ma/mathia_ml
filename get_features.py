import pandas as pd
from datetime import datetime

df_log = pd.read_csv("training_set_with_formatted_time.csv")

df_log["timestamp"] = pd.to_datetime(df_log["datetime"], format="mixed")

# Calculate time spent per problem (in seconds)
df_log = df_log.sort_values(['Anon.Student.Id', 'Problem.Name', 'timestamp'])
time_per_problem = (df_log.groupby(['Anon.Student.Id', 'Problem.Name'])['timestamp']
                     .apply(lambda x: (x.max() - x.min()).total_seconds()))
time_per_problem.name = 'Time_Per_Problem'

# Count hints by level per problem
hint_counts = df_log[df_log['Action'] == 'Hint Request'].groupby(
    ['Anon.Student.Id', 'Problem.Name', 'Help.Level']
).size().unstack(fill_value=0).add_prefix('Level').add_suffix('_Hints')

# Calculate hint frequency (hints per minute)
hint_freq = hint_counts.sum(axis=1) / (time_per_problem / 60)  # Convert to minutes
hint_freq.name = 'Hint_Request_Frequency'

# Combine all features
features = pd.concat([
    hint_counts,
    hint_freq,
    time_per_problem
], axis=1).reset_index()

# aggregate 
student_features = features.groupby('Anon.Student.Id').agg({
    'Level1_Hints': 'mean',
    'Level2_Hints': 'mean',
    'Level3_Hints': 'mean',
    'Hint_Request_Frequency': 'mean',
    'Time_Per_Problem': ['mean', 'median']  # Both mean and median time
})

student_features.columns = [
    'Level1_Hints_Per_Problem',
    'Level2_Hints_Per_Problem',
    'Level3_Hints_Per_Problem',
    'Hint_Request_Frequency',
    'Avg_Time_Per_Problem',
    'Median_Time_Per_Problem'
]

student_features.to_csv('student_features.csv')
