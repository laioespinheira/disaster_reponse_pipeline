import sys
import pandas as pd
import pickle
import nltk

nltk.download('omw-1.4')

from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def load_data(database_filepath):
    '''
    input: database_filepath, e.g. "DisasterResponse.db"
    output: X, Y, category_names (Feature and target variables, category_names (all categories names in the dataset)

    On this function we get the database_filename, define X, Y and category_names and return it
    '''

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('ETL_Pipeline', engine)

    X = df.message
    Y = df.iloc[:, 4:].astype('int64')
    category_names = df.columns[4:]

    return X, Y, category_names


def tokenize(text):
    '''
    input: text
    output: clean_tokens

    Using the nltk library(Natural Language Toolkit) this function perform tokenization and cleaning on the text and returns it
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    input: none
    output: cv model

    The pipeline from scikit-learn library is used here to assemble several steps that can be cross-validated together while setting
    different parameters
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__n_estimators': ([50, 100, 200])
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()