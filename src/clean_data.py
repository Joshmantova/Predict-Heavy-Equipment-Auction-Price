import pandas as pd

def hotcode_productgroup(df):
    '''
        hotcode product group
    INPUT:
        df: pandas dataframe
    OUTPUT:
        DataFrame: ProductGroup Hotcode
    '''

    return pd.get_dummies(df, columns=["ProductGroup"], drop_first=True)[['ProductGroup_MG','ProductGroup_SSL','ProductGroup_TEX','ProductGroup_TTT','ProductGroup_WL']]


def hotcode_enclosure(df):
    '''
        hotcode Enclosure
    INPUT:
        df: pandas dataframe
    OUTPUT:
        DataFrame: ProductGroup Hotcode
    '''
    return pd.get_dummies(df, columns=["Enclosure"], dummy_na=True, drop_first=True).iloc[:, 51:55]

def hotcode_state(df):
    '''
        hotcode state
    INPUT:
        df: pandas dataframe
    OUTPUT:
        DataFrame: ProductGroup Hotcode
    ''' 

    return pd.get_dummies(df, columns=["state"], drop_first=True).iloc[:, 51:98]

def hotcode_modelid(df):
    '''
        hotcode productid.
    INPUT:
        df: pandas dataframe
    OUTPUT:
        DataFrame: ProductGroup Hotcode
    ''' 

    return pd.get_dummies(df, columns=["ModelID"], drop_first=False).iloc[:,51:1814]

