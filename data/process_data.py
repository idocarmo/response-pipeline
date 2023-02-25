import sys
import re
import numpy as np
import pandas as pd

from sqlalchemy import create_engine

sys.path.append('../')

def load_data(messages_filepath, categories_filepath):
    """Load messages data.
    
    Load messages and its categories data and pandas dataframes and
    merge them.

    Args:
      messages_filepath:
        String with messages file path.
      categories_filepath:
        String with messages categories file path.

    Returns:
      Merged messages and categories dataframe.
    """
    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)

    df = pd.merge(df_messages, df_categories, on='id', how='inner')

    return df
    


def clean_data(df):
    """Clean messages data.

    Clean mesages data, converts categories list to binary columns and
    remove duplicates from data.  

    Args:
        df:
            Dataframe with messages data and categories.

    Returns:
        Cleaned messages dataframe.
    """

    df = df.copy()

    df_categories = df.categories.str.split(';', expand=True)

    row = df_categories.loc[0]
    category_colnames = row.apply(lambda text: re.findall('(\w+)-', text)[0]).values
    df_categories.columns = category_colnames

    for column in df_categories:
        # set each value to be the last character of the string
        df_categories[column] = df_categories[column].apply(lambda text: text[-1])
        
        # convert column from string to numeric
        df_categories[column] = df_categories[column].astype(np.int8)

    df_categories[((df_categories!=0)&(df_categories!=1))] = 1 

    df.drop(['original', 'categories'], axis=1, inplace=True)

    df = pd.concat([df, df_categories], axis=1)

    df.drop_duplicates(subset=['id', 'message'], inplace=True, keep='first')

    df.drop(['child_alone'], axis=1, inplace=True)

    return df


def save_data(df, database_filename):
    """Saves the data as SQLite database.

    Args:
        df:
            Dataframe with cleaned messages and categories data.
        database_filename:
            String with the name of the SQL database to be saved.

    Returns:
        None.
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')
      


def main():
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