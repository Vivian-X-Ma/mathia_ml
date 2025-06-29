import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV


# Load data 
df_workspace = pd.read_csv('workspace_summary_train.csv')
df_outcomes = pd.read_csv('student_scores_train.csv')  # Contains 'Anon.Student.Id', 'PreMath', 'PostMath'

# Merge workspace data with outcomes
df = pd.merge(df_workspace, df_outcomes, on='Anon.Student.Id', how='inner')

# Drop columns not useful for prediction
df = df.drop(['workspace_started_on', 'workspace_ended_on'], axis=1, errors='ignore')

# Convert categorical columns
df['workspace_progress_status'] = df['workspace_progress_status'].map({'GRADUATED': 1, 'PROMOTED': 0})
df['workspace_type'] = df['workspace_type'].map({'Concept Builder': 0, 'Mastery': 1})

# Handle missing data
def handle_missing(df):
    # Numeric columns: Fill NA with median
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    # Categorical columns: Fill NA with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df

df = handle_missing(df)

# print(df.head())


#duplicated_rows = df[df['Anon.Student.Id'].duplicated(keep=False)].sort_values('Anon.Student.Id')
#print(duplicated_rows.head())



# Aggregate workspace-level features per student
def engineer_features(df):
    # Group by student (if multiple workspaces per student)
    agg_rules = {
        'workspace_total_time_seconds': ['sum', 'mean'],
        'problems_completed': 'sum',
        'error_count': 'sum',
        'hint_count': 'sum',
        'skills_mastered': 'sum',
        'skills_encountered': 'sum',
        'aplse_earned': 'sum',
        'aplse_possible': 'sum',
        'workspace_progress_status': 'mean',  # Graduation rate
        'PreMath': 'first'  # Keep PreMath score as a baseline
    }
    df_student = df.groupby('Anon.Student.Id').agg(agg_rules)
    
    # Flatten multi-index columns
    df_student.columns = ['_'.join(col).strip() for col in df_student.columns]
    
    # Create derived features
    df_student['skill_mastery_ratio'] = df_student['skills_mastered_sum'] / df_student['skills_encountered_sum']
    df_student['aplse_score'] = df_student['aplse_earned_sum'] / df_student['aplse_possible_sum']
    df_student['hint_rate'] = df_student['hint_count_sum'] / df_student['problems_completed_sum']
    df_student['error_rate'] = df_student['error_count_sum'] / df_student['problems_completed_sum']
    df_student['time_per_problem'] = df_student['workspace_total_time_seconds_sum'] / df_student['problems_completed_sum']
    
    # Merge back PostMath (target)
    df_student = pd.merge(df_student, df_outcomes[['Anon.Student.Id', 'PostMath']], 
                          on='Anon.Student.Id', how='left')
    return df_student

df_student = engineer_features(df)
df_student = df_student.dropna(subset=['PostMath'])

# Define features and target
X = df_student.drop(['PostMath', 'Anon.Student.Id'], axis=1, errors='ignore')
y = df_student['PostMath']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Model candidates
models = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'XGBoost': GradientBoostingRegressor(random_state=42)
}

# Evaluate models
results = {}
for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('imputer', SimpleImputer(strategy='median')),  # Handle any remaining NA
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    results[name] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred)
    }

# Print results
for model, metrics in results.items():
    print(f"{model}: RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.2f}, MAE={metrics['MAE']:.2f}")