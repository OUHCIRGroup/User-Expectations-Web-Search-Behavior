# Import necessary libraries
import os
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Set the working directory
os.chdir('data_path')

# Load the datasets
df_task = pd.read_csv('df_task.csv')
df_page_SERP = pd.read_csv('df_SERP_clean.csv')
df_query = pd.read_csv('df_query.csv')
df_query_page = pd.read_csv('query_page.csv')

# Define feature lists
fea_query = ['useful_pages', 'clicking_results', 'spending_time', 'useful_information',
             'effort', 'satisfaction_x', 'session', 'belong_query_id', 'task_type']
fea_page = ['dwell_time_new', 'clicked_results', 'usefulness_dict']

# Data processing
df_query_page['dwell_time_new'] = df_query_page['end_timestamp'] - df_query_page['start_timestamp']

df_query_act = pd.DataFrame()
for q in df_query_page['belong_query_id'].unique():
    df_q = df_query_page[df_query_page['belong_query_id'] == q][fea_query + fea_page]
    dict_q = df_q[fea_query].iloc[0].to_dict()
    dict_q['dwell_time'] = df_q['dwell_time_new'].sum()
    dict_q['click_number'] = df_q['clicked_results'].apply(lambda x: len(ast.literal_eval(x) if type(x) != list else x)).sum()
    use_dict = df_q['usefulness_dict'].apply(ast.literal_eval).explode().value_counts().to_dict()
    dict_q['use_number'] = len(use_dict)
    df_query_act = df_query_act.append(dict_q, ignore_index=True)

# Clip data values
df_query_act['useful_pages'] = np.clip(df_query_act['useful_pages'], 0, 10)
df_query_act['clicking_results'] = np.clip(df_query_act['clicking_results'], 0, 10)

# Binning data for visualization
df_query_act['task_type_cat'] = pd.cut(df_query_act['task_type'], 3, labels=["Low", "Medium", "High"])
df_query_act['spending_time_cat'] = pd.cut(df_query_act['spending_time'], 3, labels=["Low", "Medium", "High"])
df_query_act['useful_pages_cat'] = pd.cut(df_query_act['useful_pages'], 3, labels=["Low", "Medium", "High"])
df_query_act['clicking_results_cat'] = pd.cut(df_query_act['clicking_results'], 3, labels=["Low", "Medium", "High"])

# Visualization
plt.rc('font', size=20)
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
sns.pointplot(x="useful_pages_cat", y="use_number", data=df_query_act, estimator=np.mean, ci=95, capsize=.2, ax=ax, label='Mean')
sns.regplot(x="useful_pages", y="use_number", data=df_query_act, x_bins=10, scatter=False, lowess=True, truncate=True, ci=None, ax=ax, color='orange', label='Regression')
ax.set_xticks(range(3))
ax.set_xticklabels(["Low", "Medium", "High"])
ax.set_xlabel('# Expected useful pages')
ax.set_ylabel('# Annotated useful pages')
legend_elements = [Line2D([0], [0], color='tab:blue', marker='o', lw=2, label='Mean', markersize=8)]
ax.legend(legend_elements, ['Mean'], bbox_to_anchor=(1, 1))
plt.show()

