#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[17]:


df1 = pd.read_csv(r"C:\Users\mfaso\Downloads\workspace_summary_train.csv")
df2 = pd.read_csv(r"C:\Users\mfaso\Downloads\student_scores_train.csv")
dfmerge = pd.merge(df1, df2, on="Anon.Student.Id")


# In[18]:


agg_dfmerge = dfmerge.groupby("Anon.Student.Id").agg({
    'workspace_total_time_seconds': 'sum',
    'step_by_step_problems_completed': 'sum',
    'problems_completed': 'sum',
    'hint_count': 'sum',
    'error_count': 'sum',
    'skills_encountered': 'sum',
    'skills_mastered': 'sum',
    'aplse_earned': 'sum',
    'aplse_possible': 'sum',
    'PreMath': 'first',
    'PostMath': 'first'
}).reset_index()

agg_dfmerge['skills'] = agg_dfmerge['skills_mastered'] / agg_dfmerge['skills_encountered'].replace(0, np.nan)
agg_dfmerge['aplse'] = agg_dfmerge['aplse_earned'] / agg_dfmerge['aplse_possible'].replace(0, np.nan)
agg_dfmerge = agg_dfmerge.fillna(0)


# In[19]:


bins = [0, 0.5, 0.8, 1.0]
labels = ['Low', 'Medium', 'High']

agg_dfmerge['performance_level'] = pd.cut(agg_dfmerge['PostMath'], bins=bins, labels=labels, include_lowest=True)
print(agg_dfmerge['performance_level'].value_counts())


# In[20]:


sns.countplot(data=agg_dfmerge, x='performance_level', order=labels)
plt.title('Performance Levels Based on PostMath')
plt.xlabel('Performance Level')
plt.ylabel('Number of Students')
plt.show()


# In[21]:


# Performance levels
bins = [0, 0.4, 0.6, 0.75, 0.9, 1.0]
labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']

agg_dfmerge['performance_level'] = pd.cut(
    agg_dfmerge['PostMath'],
    bins=bins,
    labels=labels,
    include_lowest=True
)


# In[22]:


# Create 5 quantile-based ordinal bins
agg_dfmerge['performance_level'] = pd.qcut(
    agg_dfmerge['PostMath'],
    q=5,
    labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
)


# In[23]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='performance_level', data=agg_dfmerge, order=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
plt.title('Distribution of Ordinal Performance Levels')
plt.xlabel('Performance Level')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[24]:


features_to_plot = ['hint_count', 'error_count', 'skills', 'aplse']

for feature in features_to_plot:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='performance_level', y=feature, data=agg_dfmerge, order=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    plt.title(f'{feature} vs Performance Level')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# In[26]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X = agg_dfmerge[features_to_plot]
y = agg_dfmerge['performance_level_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))


# In[28]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, cohen_kappa_score

# Example: Using features to predict performance level (ordinal)
features = ['workspace_total_time_seconds', 'step_by_step_problems_completed', 
            'problems_completed', 'hint_count', 'error_count', 'skills', 'aplse']

# Encode ordinal target
ordinal_mapping = {
    'Very Low': 1,
    'Low': 2,
    'Medium': 3,
    'High': 4,
    'Very High': 5
}
agg_dfmerge['performance_level_encoded'] = agg_dfmerge['performance_level'].map(ordinal_mapping)

X = agg_dfmerge[features]
y = agg_dfmerge['performance_level_encoded']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)


# In[29]:


from sklearn.metrics import cohen_kappa_score

# QWK (weights='quadratic')
qwk = cohen_kappa_score(y_test, y_pred, weights='quadratic')
print(f"Quadratic Weighted Kappa: {qwk:.4f}")


# In[30]:


print(classification_report(y_test, y_pred))
sns.heatmap(pd.crosstab(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix for Ordinal Prediction")
plt.show()


# In[31]:


import matplotlib.pyplot as plt

# Plot predicted class distribution
plt.figure(figsize=(6, 4))
pd.Series(y_pred).value_counts().sort_index().plot(kind='bar')
plt.xticks(rotation=0)
plt.title("Distribution of Predicted Performance Levels")
plt.xlabel("Performance Level")
plt.ylabel("Count")
plt.show()


# In[33]:


residuals = y_pred - y_test
-> I tried doing something here, but I forgot, so we could delete this


# In[ ]:




