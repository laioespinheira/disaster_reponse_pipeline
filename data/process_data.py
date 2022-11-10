import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    input: messages_filepath and categories_filepath, e.g. disaster_messages.csv and disaster_categories.csv
    output: df

    Using the pandas library we import, clean and concat the data into a final dataframe
    '''
    # import csv's
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # split string on categories
    categories = categories.categories.str.split(';', expand=True)
    # create column names based on columns
    row = categories.iloc[0, :]
    categories_colnames = list(map(lambda x: x[:-2], row))
    categories.columns = categories_colnames

    # getting only 0s and 1s from categories and transforming to integers
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype('int64')

    df = pd.concat([messages, categories], axis=1)

    return df


def clean_data(df):
    '''
    input: df
    output: df

    another series of cleasing is done with the clean_data function
    '''
    df = df.drop_duplicates()
    df = df[df['related'] != 2]
    df = df.dropna()

    return df


def save_data(df, database_filename):
    '''
    save_data
    creates a local sqlite database

    Input:
    df - dataframe to be stored as a db
    database_filename - name of the database e.g. DisasterResponse.db
    '''
    # storing the dataframe into a local sqlite database
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('ETL_Pipeline', engine, index=False)


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