import pandas as pd
import os
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 400


current_directory = os.path.dirname(__file__)
file_path = os.path.join(current_directory, 'resources//credit_card_clients.xls')
dataset = pd.read_excel(file_path, sheet_name='Sheet1')


def create_index_list():

    df_zero_mask = dataset == 0
    feature_zero_mask = df_zero_mask.iloc[:,1:].all(axis=1)

    return feature_zero_mask

def df_clean_transform():

    feature_zero_mask = create_index_list()

    data_clean = dataset.loc[~feature_zero_mask, :].copy()
    valid_pay_1_mask = data_clean['PAY_1'] !='Not available'
    data_clean_2 = data_clean.loc[valid_pay_1_mask, :].copy()
    data_clean_2['PAY_1'] = data_clean_2['PAY_1'].astype('int64')

    return data_clean_2

def education_encoding():
    data_clean_2 = df_clean_transform()

    data_clean_2['EDUCATION'].replace(to_replace=[0,5,6], value=4, inplace=True)
    data_clean_2.groupby('EDUCATION').agg({'default payment next month':'mean'})
    data_clean_2['EDUCATION_CAT'] = 'none'

    cat_mapping = {1:"graduate school",
                   2:"university",
                   3:"high school",
                   4:"others"}


    data_clean_2['EDUCATION_CAT'] = data_clean_2['EDUCATION'].map(cat_mapping)
    edu_oneHot = pd.get_dummies(data_clean_2['EDUCATION_CAT'])
    data_with_oneHot = pd.concat([data_clean_2, edu_oneHot], axis=1)

    return data_with_oneHot


