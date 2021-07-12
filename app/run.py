import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """
        Transform a text into a list of cleaned tokens 

        Parameters
        ----------
        text: string. Text to be tokenized
        
        Returns
        -------
        list of tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def category_distribution(df):
    """
        Calculate the percentage of messages that fall in each category (a messages belonging in more tha one category is counted in each category) 

        Parameters
        ----------
        df: a dataframes having categories in columns 4-40. Columns 4 is assumed to be the 'related' category
        
        Returns
        -------
        percentage of message per category (in a list) and the count of rows in the input dataframe
    """
    count=df['related'].count()
    cat_perc=list(df.iloc[:,4:].sum()/count)
    return cat_perc, count

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('MsgAndCategories', engine) #reads categorized messages db table
dfperf = pd.read_sql_table('PrecRecallS1', engine) #reads model performance data

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    categories=[c.replace('_',' ') for c in dfperf.columns]
    cat_perc,total_msg=category_distribution(df)
    cat_perc_news,total_news=category_distribution(df[df['genre']=='news'])
    cat_perc_direct,total_direct=category_distribution(df[df['genre']=='direct'])
    cat_perc_social,total_social=category_distribution(df[df['genre']=='social'])

    # create visuals
    #4 graphs are visulaized: 
    #- distribution of messages per category
    #- distribution of social messages per category
    #- distribution of direct messages per category
    #- distribution of news messages per category
    
    graphs = [
                {
                'data': 
                   [
                   Bar 
                      (
                      x=categories,
                      y=cat_perc
                      )
                   ],

                'layout': 
                    {
                    'title': 'Distribution of Message Categories ({} messages)'.format(total_msg),
                    'yaxis': 
                       {
                       'title': "Percentage of messages in a category",
                       'tickformat':".0%"   
                       },
                    'xaxis': 
                       {
                      'title': "",
                      'categoryorder':'total descending'
                       }
                    }
                 },
                 {
                 'data':
                    [
                    Bar
                       (
                       x=categories,
                       y=cat_perc_news
                       )
                    ],

                 'layout': 
                    {
                    'title': 'News Message only ({} messages)'.format(total_news),
                    'yaxis': 
                       {
                       'title': "Percentage of messages in a category",
                       'tickformat':".0%"
                       },
                    'xaxis': 
                       {
                       'title': "",
                       'categoryorder':'total descending'
                       }
                    }
                 },
                 {
                 'data': 
                    [
                    Bar
                       (
                       x=categories,
                       y=cat_perc_direct
                       )
                    ],

                 'layout': 
                    {
                    'title': 'Direct Message only ({} messages)'.format(total_direct),
                    'yaxis': 
                       {
                       'title': "Percentage of messages in a category",
                       'tickformat':".0%"
                       },
                    'xaxis': 
                       {
                       'title': "",
                       'categoryorder':'total descending'
                       }
                    }
                 },
                 {
                 'data': 
                    [
                    Bar
                       (
                       x=categories,
                       y=cat_perc_social
                       )
                    ],
                 'layout': 
                    {
                    'title': 'Social Message only ({} messages)'.format(total_social),
                    'yaxis': 
                       {
                       'title': "Percentage of messages in a category",
                       'tickformat':".0%"
                       },
                    'xaxis': 
                       {
                       'title': "",
                       'categoryorder':'total descending'
                       }
                    }
                 }
              ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

# web page that recaps classifier performance
@app.route('/perf')
def perf():
    
    # extract data needed for visuals
    prec= list(dfperf.iloc[0])
    rec= list(dfperf.iloc[1])
    s1= list(dfperf.iloc[2])
    categories=[c.replace('_',' ') for c in dfperf.columns]
    
    # create visuals
    graphs = [
                {
                'data': 
                   [
                   Bar
                      (
                      x=categories,
                      y=prec,
                      name='Precision'
                      ),
                   Bar
                      (
                      x=categories,
                      y=rec,
                      name='Recall'
                      ),  
                   Bar
                      (
                      x=categories,
                      y=rec,
                      name='S1 Score'
                      )
                   ],

                'layout': 
                   {
                   'title': 'Model performance scores',
                   'yaxis': 
                      {
                      'title': "Score value"
                      },
                   'xaxis': 
                      {
                      'title': ""
                      },
                   'width':1200,
                   'height':450,
                   }
                }
             ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('perf.html', ids=ids, graphJSON=graphJSON)

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
