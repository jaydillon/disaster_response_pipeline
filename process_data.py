import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Load messages data and categories data

    Arguments:
        messages_filepath: path to the .csv file of messages
        categories_filepath: path to the .csv file of categories

    Output:
        df: combined dataframe having both messages and categories data
    '''

    messages = pd.read_csv('messages.csv')
    categories = pd.read_csv('categories.csv')

    df = messages.merge(categories, how='outer', on='id')

    return df


def clean_categories(df):
    '''
    Clean and transform categories data

    Arguments:
        df: combined dataframe having both messages and categories data

    Output:
        df: combined dataframe with categories data cleaned up
    '''

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[[1]]

    # use this row to extract a list of new column names for categories.
    category_colnames = [category_name.split('-')[0] for category_name in row.values[0]]

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1:]
        categories[column] = categories[column].astype(int)

    # Concate original dataframe with the new categories dataframe, and remove duplicates
    df = df.drop('categories',axis=1)
    df = pd.concat([df,categories],axis=1)
    df = df.drop_duplicates()

    return df


def save_data_to_db(df, database_filename):
    """
    Save processed dataframe to SQLite database

    Arguments:
        df: cleaned dataframe containing messages and categories
        database_filename: Path to SQLite database
    """
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = database_filename.replace(".db","") + "_table"
    df.to_sql(table_name, engine, index=False, if_exists='replace')


def main():
    """
    Main function which will kick off the data processing functions. There are three primary actions taken by this function:
        1) Load messages data and categories data
        2) Clean and transform categories data
        3) Save processed dataframe to SQLite database
    """

    # Execute the ETL pipeline if the count of arguments is matching to 4
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
