import sys
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Loads messages and categories dataset, perform basic cleaning and merge the two dataset

        Parameters
        ----------
        messages_filepath, categories_filepath : str
        The location of the two csv source files

        Returns
        -------
        df: a pandas dataset concatenating categories columns after message columns
    """
    #read messages
    messages = pd.read_csv(messages_filepath)

    #read categories
    #categories csv files consists of two columns (comma separated). Second columns in turn consist of 36 columns (separation is based on ';') 
    categories = pd.read_csv(categories_filepath, skiprows=1,sep='[,;]',engine='python',header=None)
    #column 0 contains the id's to be used for merging. We use first row to define following columns name
    colname={0:'id'}
    row1=categories.iloc[0]
    for i,s in enumerate(row1[1:]):
        colname[i+1]=s.split('-')[0]
    categories.rename(columns=colname,inplace=True)

    #the two datasets are merged based on 'id' column
    df = pd.merge(messages,categories,on='id')
    
    return (df)

def clean_data(df):
    """
        Performs basic data manipulation

        Parameters
        ----------
        df : a pandas dataset concatenating categories columns after message columns
        Following cleaning are needed: 
        - on categories columns (4-39) we must replace 'string-digit' with 'digit' (e.g. "aid_related-0" with "0"). Digit must be casted into an integer
        - repeated rows must be dropped

        Returns
        -------
        df: cleaned version of the input dataset
    """
    
    # remove redundant strings from categories column
    for col in df.columns[4:]:
        df[col]=df[col].apply(lambda x: int(str(x)[-1]))

    # check and remove duplicate duplicates
    dp= df.duplicated().sum()
    if dp >0: 
        df.drop_duplicates(inplace=True)
        print("{} duplicated rows removed".format(dp-df.duplicated().sum()))
    
    return df


def save_data(df, database_filename):
    """
        Save dataframe into a sqllite db 

        Parameters
        ----------
        df : a pandas dataset concatenating categories columns after message columns
        database_filename: filename for the db to be created
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('MsgAndCategories', engine, index=False, if_exists='replace')

    
def main():
    """
        Load, manipulate and save data 
        Expected to be launched from terminal command; command must include path for csv files and for db filename
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()