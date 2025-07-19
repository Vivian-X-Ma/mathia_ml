#!/usr/bin/env python
# coding: utf-8

# In[18]:


# import
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns


# In[19]:


df1 = pd.read_csv("workspace_summary_train.csv")
df2 = pd.read_csv("student_scores_train.csv")
df1.head()


# In[20]:


df2.head()


# In[21]:


dfmerge = pd.merge(df1, df2, on="Anon.Student.Id", how="outer")  # Keep all students from df2
dfmerge.head()


# In[22]:


print(dfmerge.columns)


# In[23]:


agg_dfmerge = (
    dfmerge.groupby("Anon.Student.Id")
    .agg(
        {
            "workspace_total_time_seconds": "sum",
            "step_by_step_problems_completed": "sum",
            "problems_completed": "sum",
            "hint_count": "sum",
            "error_count": "sum",
            "skills_encountered": "sum",
            "skills_mastered": "sum",
            "aplse_earned": "sum",
            "aplse_possible": "sum",
            "PreMath": "first",
            "PostMath": "first",
        }
    )
    .reset_index()
)


# In[24]:


agg_dfmerge["skills"] = agg_dfmerge["skills_mastered"] / agg_dfmerge[
    "skills_encountered"
].replace(0, np.nan)
agg_dfmerge["aplse"] = agg_dfmerge["aplse_earned"] / agg_dfmerge[
    "aplse_possible"
].replace(0, np.nan)


agg_dfmerge = agg_dfmerge.fillna(0)

print(f"Final row count: {len(agg_dfmerge)}")  # Should now be 557



# In[25]:


agg_dfmerge = agg_dfmerge.fillna(0)
agg_dfmerge.head()


# In[26]:


features = [
    "workspace_total_time_seconds",
    "step_by_step_problems_completed",
    "problems_completed",
    "hint_count",
    "error_count",
    "skills",
    "aplse",
]

X = agg_dfmerge[features]
y = agg_dfmerge["PostMath"]


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R^2: {r2:.4f}")
print(f"MAE: {mae: .4f}")
# model = RandomForestRegressor(random_state=42)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)


# In[28]:


plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Real PostMath")
plt.ylabel("Pred PostMath")
plt.title("Real vs. Pred PostMath")
plt.grid(True)
plt.show


# In[29]:


agg_dfmerge["total_attempts"] = agg_dfmerge["problems_completed"]

agg_dfmerge["correct_attempts"] = agg_dfmerge["skills_mastered"]

agg_dfmerge["correct_attempt_ratio"] = np.where(
    agg_dfmerge["total_attempts"] > 0,
    agg_dfmerge["correct_attempts"] / agg_dfmerge["total_attempts"],
    0,
)

print(
    agg_dfmerge[
        [
            "Anon.Student.Id",
            "total_attempts",
            "correct_attempts",
            "correct_attempt_ratio",
        ]
    ].head()
)


# In[30]:


features = [
    "workspace_total_time_seconds",
    "step_by_step_problems_completed",
    "problems_completed",
    "hint_count",
    "error_count",
    "skills",
    "aplse",
    "total_attempts",
    "correct_attempts",
    "correct_attempt_ratio",
]

# Export extracted features with Anon.Student.Id to CSV
agg_dfmerge[["Anon.Student.Id"] + features].to_csv(
    "student_features_moyo.csv", index=False
)


# In[ ]:
