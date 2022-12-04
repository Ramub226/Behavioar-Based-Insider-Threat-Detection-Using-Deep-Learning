#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import numpy as np
import dask.dataframe as dd

TOPIC_NUM = 100

tmp_dir = Path('./')
answers_dir = Path(r"./answers")
answers_file = answers_dir/"insiders.csv"
dataset_version = '5.2'
assert(answers_file.is_file())


# In[3]:


# https://stackoverflow.com/questions/57531388/how-can-i-reduce-the-memory-of-a-pandas-dataframe
def reduce_mem_usage(df, ignore_cols = None ):
    
    """ 
    iterate through all the columns of a dataframe and 
    modify the data type to reduce memory usage.        
    """
    if ignore_cols == None:ignore_cols = []

    start_mem = df.memory_usage().sum() / 1024**2
    
        
    for col in df.columns:
        col_type = df[col].dtype
        print(col, col_type)

        if col in ignore_cols:
            continue
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max <                  np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max <                   np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max <                   np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max <                   np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max <                   np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max <                   np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
        print("\tNew dtype:  ", df[col].dtype)
    end_mem = df.memory_usage().sum() / 1024**2
    print(('Memory usage after optimization is: {:.2f}' 
                              'MB').format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) 
                                             / start_mem))
    
    return df


# In[4]:


preprocessed_dir = Path(f'./CERT_output/')
assert(preprocessed_dir.is_dir())


# In[9]:


processed_files = ['logon_preprocessed.csv', 'http_preprocessed.csv','device_preprocessed.csv',
                  'file_preprocessed.csv']
columns = ['id', 'date', 'user', 'is_work_time', 'subtype']
temp_df=[]

for file in processed_files:
    temp_df.append(pd.read_csv(preprocessed_dir/file,usecols = columns))


# In[10]:


df = pd.concat(temp_df, axis=0)
df.isna().sum()


# In[11]:


for temp in temp_df:
    del temp


# In[12]:


df = reduce_mem_usage(df, ignore_cols=['id', 'date'])


# In[13]:


subtype_encoder = LabelEncoder()
type_encoder = LabelEncoder()
df.subtype = df.subtype.map(str)
df['subtype'] = subtype_encoder.fit_transform(df['subtype'])
#df['type'] = type_encoder.fit_transform(df['type'])
df['type'] = type_encoder.fit_transform(df['subtype'])

#df['action_id'] = df.is_usual_pc.astype(np.int8) * 100 + df.is_work_time.astype(np.int8) * 10 + df.subtype
df['action_id'] =  df.is_work_time.astype(np.int8) * 10 + df.subtype
df['date'] = pd.to_datetime(df.date, format='%Y/%m/%d %H:%M:%S')

df = df[['id', 'date', 'user', 'action_id']]

df.to_pickle(str(tmp_dir / "df.pkl"))
del df


# In[14]:


content_dir = Path(f'./CERT_output/')
content_file = ['email_lda.csv', 'file_lda.csv', 'http_lda.csv']
content_cols = ['id', 'content']
temp_df = []

for file in content_file:
    df = pd.read_csv(content_dir/ file, usecols = content_cols)
    df = reduce_mem_usage(df, ignore_cols=content_cols)
    temp_df.append(df)


# In[16]:



content_df = pd.concat(temp_df, axis=0)

print(content_df)

# In[17]:


for df in temp_df:
    del df


# In[18]:


content_df = reduce_mem_usage(content_df, ignore_cols=['id', 'content'])
content_df.to_csv(str(tmp_dir / 'content_df.csv'))


# In[5]:


content_df = dd.read_csv(str(tmp_dir / 'content_df.csv'))    .set_index('id').drop('Unnamed: 0', axis=1)
df = pd.read_pickle(str(tmp_dir / "df.pkl"))

# Merge the csv files.
df = dd.merge(content_df, df, how='inner', on=['id'])


# In[ ]:


#df = df.reset_index().drop(['index', 'type'], axis=1)
df['day'] = df.date.dt.floor('D')
df.set_index('date')

action_id_lists = df.groupby(['user', 'day'], sort=True)    ['action_id'].apply(list)

content_lists = df.groupby(['user', 'day'], sort=True)    ['content'].apply(list)

action_id_lists = action_id_lists.reset_index()
content_lists = content_lists.reset_index()

df_merged = dd.merge(action_id_lists, content_lists, how='inner', on=['user', 'day'])
df_merged.to_csv(str(tmp_dir / "merged_df.csv"), index=False, single_file=True)


# In[ ]:


main_df = pd.read_csv(answers_file)
main_df = main_df[main_df['dataset'].astype(str) ==
                  str(dataset_version)].drop(['dataset', 'details'], axis=1)


# In[ ]:


df = pd.read_csv(str(tmp_dir / "merged_df.csv"))
df = df.merge(main_df, left_on='user', right_on='user', how='left')
#df = df.drop(['start', 'end', 'day', 'user'], axis=1)
df['day'] = pd.to_datetime(df.day, format='%Y-%m-%d ').dt.floor('D')
df['dayofweek']=df['day'].dt.dayofweek
df['malicious'] = (df['dayofweek'] >= 5)
df = df.drop(['day','user'],axis=1)
df = df.dropna()
df.to_csv(str(tmp_dir / 'merged_answers_df.csv'), index=False)


# In[ ]:


df = pd.read_csv(str(tmp_dir / 'merged_answers_df.csv'),)
df = reduce_mem_usage(df, ignore_cols=['action_id', 'content'])

import csv
f = open('./dataset.csv','w',newline='', encoding='utf-8')
writer = csv.writer(f)
# In[ ]:

df = df.dropna()
import ast
from scipy.sparse import csc_matrix

for idx, row in df.iterrows():
    content = ast.literal_eval(row.content)
    day=row.dayofweek
    scenario=row.scenario
    action=ast.literal_eval(row.action_id)
    mal=row.malicious
    yl=len(content)
    for iv in range(yl):
        y=ast.literal_eval(content[iv])
        ixtt=len(y)
        for ix in range(ixtt):
            ct=y[ix][0]
            prob=y[ix][1]
            print('Values ', ct,',',prob,',',action[iv],',',scenario,',',mal)
            data=[ct,prob,action[iv],scenario,day,mal]
            writer.writerow(data)


f.close()            


