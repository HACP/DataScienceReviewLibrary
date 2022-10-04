import pandas as pd
import numpy as np

def get_df_summary_basic(df):
    # Objective: Summary of main descriptive metrics of a Data Set
    # Returns: df with summary metrics
    # Includes: count of nulls, ratio of nulls, number of unique elements, ratio of unique elements
    df_info = df.isnull().sum().reset_index()
    df_info.columns = ['Feature','Count_Null']
    df_info['Ratio_Null'] = df_info['Count_Null']/len(df)
    df_info['Count_Unique'] = [len(pd.unique(df[col])) for col in df.columns] # Opportunity: Make it more pythonic
    df_info['Ratio_Unique'] = df_info['Count_Unique']/len(df)
    return(df_info)

def get_entropy(df, col):
    # Objective: Computes Shannon entropy for a given feature
    # Literature: https://en.wikipedia.org/wiki/Entropy_(information_theory)
    # Returns: float number. Positive by definition
    # Test Cases:
    ##  Test1: df_test1 = pd.DataFrame([0],columns=['feature']) -> H = 0 --- Only one element in the distribution
    ##  Test2: df_test2 = pd.DataFrame(np.arange(0,10),columns=['feature']) -> H = np.log(10)
    df_sel_freq = df.groupby(col).size().reset_index()
    H = -(np.log(df_sel_freq[0]/np.sum(df_sel_freq[0]))*(df_sel_freq[0]/np.sum(df_sel_freq[0]))).sum()
    return(H)

def get_df_summary_entropy(df):
    return [get_entropy(df, col) for col in df.columns]

def get_mutual_information(df, col1, col2):
    # Objective: Computes Mutual Information between two given features
    # Literature: https://en.wikipedia.org/wiki/Mutual_information
    # Returns: float number. Positive by definition
    # Test Cases:
    ##  Test1: get_mutual_information(df, col1, col2) = get_mutual_information(df, col2, col1) -- symmetry property.
    ##         check with df_test3 = pd.DataFrame(np.transpose([np.arange(0,10), 200*np.arange(0,10)+500]) ,columns=['feature1','feature2'])
    ##  Test2: get_mutual_information(df, col1, col1) = get_entropy(df, col1) by definition of MI
    if col1 != col2:
        df_subset_col1 = df.groupby(col1).size().reset_index()
        df_subset_col1['ProbCol1'] = df_subset_col1[0]/np.sum(df_subset_col1[0])
        df_subset_col2 = df.groupby(col2).size().reset_index()
        df_subset_col2['ProbCol2'] = df_subset_col2[0]/np.sum(df_subset_col2[0])
        df_subset_joint = df.groupby([col1,col2]).size().reset_index()
        df_subset_joint['ProbJoint'] = df_subset_joint[0]/np.sum(df_subset_joint[0])
        df_subset_joint_prob = pd.merge(pd.merge(df_subset_joint, df_subset_col1,on=col1, how='left'),df_subset_col2, on=col2,how='left')
        MI = np.sum(df_subset_joint_prob['ProbJoint']*np.log(df_subset_joint_prob['ProbJoint']/(df_subset_joint_prob['ProbCol1']*df_subset_joint_prob['ProbCol2'])))
        return(MI)
    else:
        return(get_entropy(df, col1))

def get_df_summary_mutual_information_target(df,col_target):
    # col_target is the target variable
    return [get_mutual_information(df, col_target, col) for col in df.columns]

def get_df_summary_advanced_information_theory(df):
    df_info = get_df_summary_basic(df)
    df_info['Entropy'] = get_df_summary_entropy(df)
    df_info['MutualInformationTarget'] = get_df_summary_mutual_information_target(df, 'Survived')
    return(df_info)

df_train = pd.read_csv('../data/titanic/train.csv')

# Entropy Tests
# Test 1
df_test1 = pd.DataFrame([0],columns=['feature'])
get_entropy(df_test1,'feature') + 0 == 0

# Test 2
df_test2 = pd.DataFrame(np.arange(0,10),columns=['feature'])
abs(get_entropy(df_test2,'feature') - np.log(10)) < 0.01

# Mutual Information Test
# Test 1
df_test3 = pd.DataFrame(np.transpose([np.arange(0,10), 200*np.arange(0,10)+500]) ,columns=['feature1','feature2'])
df_test3
get_mutual_information(df_test3, 'feature1','feature2') - get_mutual_information(df_test3, 'feature2','feature1') == 0

# Test 2
df_test3 = pd.DataFrame(np.transpose([np.arange(0,10), 200*np.arange(0,10)+500]) ,columns=['feature1','feature2'])
df_test3
get_mutual_information(df_test3, 'feature1','feature1') == get_entropy(df_test3, 'feature1')
