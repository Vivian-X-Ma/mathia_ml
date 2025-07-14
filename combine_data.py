import pandas as pd

df_summary = pd.read_csv('workspace_summary_train.csv')
student_features = pd.read_csv('student_features.csv')

student_features = student_features.reset_index()

combined_df = pd.merge(
    df_summary,
    student_features,
    on='Anon.Student.Id',
    how='left',  # Use 'outer' to keep all students from both datasets
    indicator=True  # Shows which dataset each row came from
)

combined_df = combined_df.drop(columns='_merge')

combined_df.to_csv('combined_student_data.csv', index=False)

print(combined_df.head())