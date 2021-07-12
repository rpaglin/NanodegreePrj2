import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from sqlalchemy import create_engine
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import pickle


def load_data(database_filepath):
    """
        load data from sql database

        Parameters
        ----------
        database_filepath: path for database file
        
        Returns
        -------
        X: 1-column dataframe containing text messages
        Y: 36-column dataframe containning target message categories 
        category_names: names of target message categories
    """
   
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('MsgAndCategories',engine)
    X = df['message']
    Y = df[df.columns[4:]]
    category_names=df.columns[4:]
    return X,Y,category_names


def tokenize(text):
    """
        tokenize messages (tokenize, lemmatize, lowercase, remove blanks, punctuation and numbers

        Parameters
        ----------
        text: text message to be tokenized 
        
        Returns
        -------
        clean_tokens: list of tokens obtained from text
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if clean_tok.isalpha():
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model(optimize=False):
    """
        Create the model pipeline
        
        Parameters
        ----------
        optimize: boolean Trigger execution of gridsearch for parameters tuning (to save time in iterations)  
        
        Returns
        -------
        Pipeline model for message classification
    """
    model = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1,1),max_df=0.5,max_features=5000)),
                ('tfidf', TfidfTransformer(use_idf=True)),
                ('MOClass',MultiOutputClassifier(estimator=RandomForestClassifier()))
                ])
    
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        'MOClass__estimator__n_estimators': [50, 100, 200],
        'MOClass__estimator__min_samples_split': [2, 3, 4]
        }
    if optimize:    
        model = GridSearchCV(model, param_grid=parameters)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
        Print model performance on test data
        
        Parameters
        ----------
        model: classification pipeline
        X_test: dataframe for test messages
        Y_test: dataframe for test classification (true values)
        category_names: list of category names
        
        Returns
        -------
        A dataframe with categories as columns and weigthed average Precision, Recall and f1 scores in rows 
    """
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred)
    Y_pred.columns=category_names
    model_score={'Precision': [], 'Recall': [], 'f1':[]}
    print("-----------------------------------------------------------------------------")
    for c in category_names:
        y_true = Y_test[c]
        y_pred = Y_pred[c]
        prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        model_score['Precision'].append(prec)
        model_score['Recall'].append(rec)
        model_score['f1'].append(f1)
        print("Category: {}".format(c))
        print(classification_report(y_true, y_pred,digits=5))
        #print("Confusion Matrix:")
        #print(confusion_matrix(y_true, y_pred))
        print("-----------------------------------------------------------------------------")
    model_perf=pd.DataFrame.from_dict(model_score, orient='index',columns=category_names)
    print("Average performance:")
    print(model_perf.mean(axis=1))
    print("=============================================================================")
    return model_perf
    
def save_model(model, model_filepath):
    """
        # save the model to disk

        # some time later...

        # load the model from disk
        loaded_model = pickle.load(open(filename, 'rb'))
        result = loaded_model.score(X_test, Y_test)
        print(result)
    """
    pickle.dump(model, open(model_filepath, 'wb'))

def save_perf(df, database_filepath):
    """
        Save model performance score on the data db (in a dedicated table) 

        Parameters
        ----------
        df: database containing average precision, recall and s1 scores for each category
        database_filepath: path for database file
        
        Returns
        -------
        None
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df.to_sql('PrecRecallS1', engine, index=False, if_exists = 'replace')


def main():
    """
        Load data, create a pipeline model, train the model and evaluate performance its performance
        Performance are saved in at db table, while the model is saved for web usage using pickle dump
        A local boolena variable named 'optimize' triggers (when True) the addition of a grdsearch step in model creation

        Expected to be launched from terminal command; command must include path for db file and for pickle filename
    """
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        optimize=False
        
        print('Building model...')
        model = build_model(optimize)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        if optimize:
            print(model.best_params_)
        
        print('Evaluating model...')
        model_perf=evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        save_perf(model_perf,database_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()