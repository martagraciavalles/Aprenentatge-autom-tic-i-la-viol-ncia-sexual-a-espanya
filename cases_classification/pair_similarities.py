# %%
import warnings
warnings.filterwarnings('ignore')

from pyemd import emd
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from gensim.models.keyedvectors import KeyedVectors
from nltk.stem.snowball import SpanishStemmer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from datetime import datetime, timedelta
import lxml.etree as ET
import seaborn as sns
import pandas as pd
import numpy as np
import regex as re
import itertools
import unidecode
import html
import os
import networkx as nx
import time
from sklearn.preprocessing import StandardScaler

# %%
''' 
NEWS_PATH
- directory where articles are collected in different folders and following the 
NewsML-G2 format (xml)

OUTPUT_PATH
- path where to store a csv file containing the pairwise similarity metric and probability
of being about the same case for all the news in NEWS_PATH
'''

NEWS_PATH = '../data/news/'
OUTPUT_PATH = '../data/'
FASTTEXT_W2V_PATH = 'utilities/embeddings-l-model.vec'

DEBUG = False

# %%
def listdir_checked(path, unwanted = ['.DS_Store']):
    '''
    Discard unwanted files or directories when listing the elements in a given path
    '''
    return (f for f in os.listdir(path) if f not in unwanted)

def normalize_string(to_normalize, encoded = False):
    '''
    Normalize text given a string
    '''
    text = str(to_normalize).lower()  # lowering text
    if encoded: 
        text = ' '.join([html.unescape(term) for term in text.split()])
    text = unidecode.unidecode(text)

    text = re.sub(r'[^\w\s]', '', text)  # removing all the punctuations
    last_text = text.split()  # tokenize the text

    # remove stopwords
    stopwords_set = set(stopwords.words("spanish"))
    stopwords_set = stopwords_set.union(set(["name"]))
    
    last_text = ' '.join([x for x in last_text if (x not in stopwords_set)])
    return last_text

def normalize_text(array_of_str):
    '''
    Normalize arrays of strings
    '''
    final_array = []
    for text in array_of_str:
        normalized = normalize_string(text)
        if normalized != '': final_array.append(normalized)
    return final_array

def create_articles_dictionary(NEWS_PATH):
    '''
    Import articles information.
    Articles are stored in directories in the NEWS_PATH.
    '''
    data = {}               # keys: media, value: list of dictionaries with info about the news articles of the given media
    unique_urls = []        # list to store unique urls to discard repeated ones
    repeated_data = {}      # store repeated articles following the same format as 'data' dictionary

    for year in listdir_checked(NEWS_PATH):
        for month in listdir_checked(NEWS_PATH + '/' + year):
            for file in listdir_checked(NEWS_PATH + '/' + year + '/' + month):
                try:
                    full_path = NEWS_PATH + '/' + year + '/' + month + '/' + file
                    # Read xml file - info stored following NewsML-G2 format
                    root = ET.parse(full_path).getroot()
                    # Parse news
                    media = file.rsplit('_', 1)[0]
                    # Check repeated urls
                    url = root.findall(".//infoSource")[0].get("uri")
                    str_date = root.findall('.//contentMeta')[0].find('contentCreated').text[:10]

                    info = {
                        'id': file.split(':')[-1].replace('.xml', ''),
                        'media': media,
                        'publication_date': datetime.strptime(str_date, '%Y-%m-%d'),
                        'title': normalize_string(root.findall('.//itemRef')[0].find('title').text, encoded = True),
                        'headline': normalize_string(root.findall(".//itemRef")[0].find('description').text, encoded = True),
                        'article': normalize_string(root.findall('.//itemRef')[1].find('description').text, encoded = True),
                        'url': url
                    }
                    if url not in unique_urls:
                        unique_urls.append(url)
                        try:
                            data[media].append(info)
                        except:
                            data[media] = [info]

                    else:
                        try:
                            repeated_data[media].append(info)
                        except:
                            repeated_data[media] = [info]
                except Exception as e:
                    print('Error processing file', full_path)
                    print(e)


    return data, repeated_data

def load_elements(data):
    '''
    Load auxiliary variables:
    * mapping_keys -> dict with key: tweet id - value: absolute position to manage matrices
    * mapping_tweets -> dict with key: value - tweet_id: absolute position to manage matrices
    '''
    mapping_keys = {}  # key: tweet id -> value: absolute position in all_entities
    mapping_tweets = {}  # key: value -> tweet_id: absolute position in all_entities
    counter = 0

    for media, new in data.items():
        for element in new:
            element_id = element['id'].split('_')[-1]
            
            mapping_keys[element_id] = counter
            mapping_tweets[counter] = element['id']

            counter += 1
            
    return mapping_keys, mapping_tweets

def add_similarity_column(pairs_df, mapping_keys, similarity_matrix, column_name):
    '''
    Arguments:
     * pairs_df
         - pd.DataFrame
         - contains pairs of tweet_ids --> column names: [tweet_id_A, tweet_id_B]
     * mapping_keys
         - dictionary
         - key: tweet_id, value: position
     * similarity_matrix
         - np.matrix
         - symmetrical matrix with the similarity between tweets
     * column_name
         - string
         - name of the new column to be added in pairs_df
    '''
    similarity_pairs = []               # Create list with the same order as pairs_df
    for row in pairs_df.iterrows():
        tid_A = row[1]['tweet_id_A']       # Obtain tweet id
        tid_B = row[1]['tweet_id_B']       # Get the position of each article in the matrix
        pos_A = mapping_keys[str(tid_A)]
        pos_B = mapping_keys[str(tid_B)]
        similarity_pairs.append(similarity_matrix[pos_A, pos_B])            # Order similarity following pairs_df order
    pairs_df.insert(len(pairs_df.columns), column_name, similarity_pairs)   # Add new column
    return pairs_df

# %% [markdown]
# ## 1. LOAD DATA

# %%
data, repeated_data = create_articles_dictionary(NEWS_PATH)

print('LEN DATA:', len(data))

# %%
#CONSTRUCT A DATAFRAME WITH ALL THE ARTICLES AND THE INFORMATION NEEDED
articles_list = []

for media in data.keys():
    for new in data[media]:
        tweet_id = new['id']
        title = normalize_string(new['title'])
        headline = normalize_string(new['headline'])
        url = new['url']
        publication_date = new['publication_date']
        article = normalize_string(new['article'])
        
        articles_list.append({'tweet_id': tweet_id, 
                              'media': media, 
                              'title': title, 
                              'headline': headline, 
                              'url': url, 
                              'article': article,
                              'publication_date': publication_date})

articles_df = pd.DataFrame(articles_list)

articles_df['tweet_id'] = articles_df['tweet_id'].str.split('_').str[-1]
articles_df['media'] = articles_df['media'].str.split('_').str[0]
articles_df['tweet_id'] = articles_df['tweet_id'].astype(int)

print(articles_df.head())

#RENAME SOME MEDIA
articles_df['media'] = articles_df['media'].replace({'La': 'LaSER', 
                                                     'el': 'elPais',
                                                     'noticias': 'noticiascuatro',
                                                     'publico': 'diarioPublico',
                                                     'voz': 'vozpopuli'})

# %%
articles_df['media'].unique()

# %%
# Ordenar el DataFrame por la columna "publication_date"
articles_df = articles_df.sort_values('publication_date')

#articles_df.to_csv('../data/articles_df.csv', index=False)
# %%
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# %%
len(articles_df)

# %% [markdown]
# ----------------------------------------------------

# %% [markdown]
# #### Import the articles checked and classified by marilena

# %%
cases_labeled_marilena = pd.read_csv('../cases_labeled.csv', sep=';')

# %% [markdown]
# #### Merge the two dataframes 'articles_df' and 'cases_labeled' based on the 'tweet_id' column

# %%
'''
Result dataframe containing information about each article using their corresponding tweet_id as an identifier.
Columns = [tweet_id, media, title, headline, article, publication_date ,checked, case_id]
* tweet_id - identifier
* media
* title
* headline
* article 
* publication_date
* checked - whether the tweet_id has been supervised - values: 0/1
* case_id - an integer id attributed to each article, articles about the same case have the same case_id
'''
train_df = pd.merge(articles_df[['tweet_id', 'media', 'title', 'headline', 'article', 'publication_date']], cases_labeled_marilena[['tweet_id','checked', 'case_id']], on='tweet_id')
print('We have', len(train_df), 'articles classified with a case_id defined')

# %% [markdown]
# # Check if there is any error, and correct it 

# %%
# Get tweet ids of tweets not supervised
if 'checked' in train_df.keys():
    checked_tweet_id = train_df[train_df['checked'] == 1]['tweet_id'].values

# Remove repeated articles
repeated_articles_tid = []
for media, news in repeated_data.items():
    for element in news:
        repeated_articles_tid.append(element['id'])
cases_df = train_df[~train_df['tweet_id'].isin(repeated_articles_tid)]

# CREATE ALL POSSIBLE PAIRS OF ARTICLES + ADD TARGET VARIABLE (same_case)
train_pairs_df = pd.DataFrame(columns=['tweet_id_A', 'tweet_id_B', 'same_case'])
rows = []
for pair in list(itertools.combinations(cases_df.tweet_id, 2)):
    case_1 = int(cases_df[cases_df['tweet_id'] == pair[0]]['case_id'])
    case_2 = int(cases_df[cases_df['tweet_id'] == pair[1]]['case_id'])

    if case_1 == case_2:
        same_case = 1
    else:
        same_case = 0

    rows.append({'tweet_id_A': pair[0], 'tweet_id_B': pair[1], 'same_case': same_case})

train_pairs_df = pd.concat([train_pairs_df, pd.DataFrame(rows)], ignore_index=True)

# %%
def jaccard_coefficient_matrix(data, mapping_keys, pairs_df):
    # GET NEWS ARTICLES
    N = len(mapping_keys)
    articles = [None] * N
    for media, new in data.items():
        for element in new:
            try:
                element_id = element['id'].split('_')[-1]
                pos = mapping_keys[element_id]
                
                articles[pos] = normalize_string(element['title'] + ' ' + element['headline'])
            except:
                pass

    # CREATE SIMILARITY MATRIX
    similarity_matrix = np.zeros(shape=(N, N))
    # Iterate for every pair of documents
    for row in pairs_df.iterrows():
        i = mapping_keys[str(row[1]['tweet_id_A'])]
        j = mapping_keys[str(row[1]['tweet_id_B'])]

        similarity_matrix[i, i] = 1
        similarity_matrix[j, j] = 1

        docA = articles[i]
        docB = articles[j]

        # Compute jaccard similarity
        intersection = len(list(set(docA).intersection(docB)))
        union = (len(docA) + len(docB)) - intersection
        try:
            jaccard = float(intersection / union)
        except: 
            jaccard = 0
        # Fill similarity matrix
        similarity_matrix[i, j] = jaccard
        similarity_matrix[j, i] = jaccard

    return similarity_matrix

def cosine_similarity_tfidf(data, mapping_keys, pairs_df):
    # GET NEWS ARTICLES
    N = len(mapping_keys)
    articles = [None] * N
    for media, new in data.items():
        for element in new:
            try:
                element_id = element['id'].split('_')[-1]
                pos = mapping_keys[element_id]
                
                articles[pos] = normalize_string(element['title'] + ' ' + element['headline'] + ' ' + element['article'])
            except:
                pass

    # COMPUTE TF-IDF
    # Stem
    stemmer = SpanishStemmer(ignore_stopwords=False)
    for i, article in enumerate(articles):
        articles[i] = str([stemmer.stem(word) for word in article.split()])
    # Compute tf-idf
    stopwords_spanish = [word.encode().decode('utf-8') for word in stopwords.words('spanish')] # Remove stopwords
    vectorizer = TfidfVectorizer(stop_words=stopwords_spanish)
    X = vectorizer.fit_transform(articles)

    # COMPUTE COSINE SIMILARITY MATRIX
    cosine_sim_matrix = np.zeros(shape=(N, N))
    for row in pairs_df.iterrows():
        i = mapping_keys[str(row[1]['tweet_id_A'])]
        j = mapping_keys[str(row[1]['tweet_id_B'])]

        similarity = cosine_similarity(X[i], X[j])[0][0]

        #cosine_sim_matrix[i,i] = 1
        #cosine_sim_matrix[j,j] = 1
        cosine_sim_matrix[i, j] = similarity
        cosine_sim_matrix[j, i] = similarity

    return cosine_sim_matrix

def wm_distance(data, mapping_keys, pairs_df):
    N = len(mapping_keys)
    articles = [None] * N
    for media, new in data.items():
        for element in new:
            try:
                element_id = element['id'].split('_')[-1]
                pos = mapping_keys[element_id]
                
                articles[pos] = normalize_string(element['title'])
            except:
                pass
    
    fasttext_file = FASTTEXT_W2V_PATH
    word_vectors = KeyedVectors.load_word2vec_format(fasttext_file)
    wmd_matrix = np.zeros(shape=(N,N))
    
    for row in pairs_df.iterrows():
        i = mapping_keys[str(row[1]['tweet_id_A'])]
        j = mapping_keys[str(row[1]['tweet_id_B'])]

        distance = word_vectors.wmdistance(articles[i], articles[j])

        wmd_matrix[i,i] = 1
        wmd_matrix[j,j] = 1
        wmd_matrix[i,j] = distance
        wmd_matrix[j,i] = distance
            
    return wmd_matrix

def cosine_similarity_BERT(data, mapping_keys, pairs_df, model):                          
    N = len(mapping_keys)
    articles = [None] * N
    for media, new in data.items():
        for element in new:
            try:
                element_id = element['id'].split('_')[-1]
                pos = mapping_keys[element_id]
                
                articles[pos] = normalize_string(element['title'] + ' ' + element['headline'] + ' ' + element['article'])
            except:
                pass

    # VECTORIZATION
    print('---STARTING THE VECTORIZATION')
    BERT_model = SentenceTransformer(model)
    encoded = BERT_model.encode(articles)

    # COMPUTE COSINE SIMILARITY MATRIX
    print('---COMPUTING COSINE SIMILARITY MATRIX')
    cosine_sim_matrix = np.zeros(shape=(N, N))
    
    for row in pairs_df.iterrows(): 
        i = mapping_keys[str(row[1]['tweet_id_A'])]
        j = mapping_keys[str(row[1]['tweet_id_B'])]

        similarity = cosine_similarity(encoded[i].reshape(1,-1), encoded[j].reshape(1,-1))[0][0]

        cosine_sim_matrix[i,i] = 1
        cosine_sim_matrix[j,j] = 1
        cosine_sim_matrix[i, j] = similarity
        cosine_sim_matrix[j, i] = similarity

    return cosine_sim_matrix

def train_log_reg(df, id_columns, target_column):
    # split training / test data
    df = df.drop(columns=id_columns)
    training_data, testing_data = train_test_split(df, random_state=2000, test_size=0.1)
    # get labels
    Y_train = training_data[target_column].values
    Y_test = testing_data[target_column].values

    X_train = training_data.drop(columns=[target_column])
    X_test = testing_data.drop(columns=[target_column])

    # logistic regression classifier
    scikit_log_reg = LogisticRegression(verbose=1, solver='liblinear', C=5, penalty='l2', max_iter=1000)

    model = scikit_log_reg.fit(X_train, Y_train.astype(int))
    score = model.score(X_test, Y_test.astype(int))
    print("Testing accuracy of LogReg = ", score)

    coefficients = model.coef_
    feature_names = X_train.columns
    coef_dict = dict(zip(feature_names, coefficients[0]))
    sorted_features = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    for feature, coef in sorted_features:
        print(f"Feature: {feature}\tCoefficient: {coef}")

    return (model)

def create_sim_features(data, mapping_keys, df):
    print(datetime.now())
    print("STARTING WORD MOVER'S DISTANCE")
    # COMPUTE WORD MOVER'S DISTANCE OF FASTTEXT EMBEDDINGS OF TITLES
    wmd_matrix = wm_distance(data, mapping_keys, df)
    df = add_similarity_column(df, mapping_keys, wmd_matrix, 'sim_word_movers')

    print(datetime.now())
    print("STARTING JACCARD COEFFICIENT")
    # COMPUTE JACCARD COEFFICIENT BETWEEN TITLE + HEADLINE OF EACH PAIR OF ARTICLES
    jaccard_matrix = jaccard_coefficient_matrix(data, mapping_keys, df)
    df = add_similarity_column(df, mapping_keys, jaccard_matrix, 'jaccard_coef')

    print(datetime.now())
    print("STARTING TF-IDF")
    # COMPUTE COSINE SIMILARITY OF TF-IDF FOR EACH PAIR OF ARTICLES
    cosine_matrix = cosine_similarity_tfidf(data, mapping_keys, df) 
    df = add_similarity_column(df, mapping_keys, cosine_matrix, 'tf-idf')

    print(datetime.now())
    print("STARTING BERT WORD")
    # COMPUTE COSINE SIMILARITY OF BERT WORD EMBEDDING FOR EACH PAIR OF ARTICLES
    BERT_similarity_uncased = cosine_similarity_BERT(data, mapping_keys, df, 'dccuchile/bert-base-spanish-wwm-uncased')
    df = add_similarity_column(df, mapping_keys, BERT_similarity_uncased, 'sim_BETO_uncased')

    return df

# %%
mapping_keys, mapping_tweets = load_elements(data)

# %%
train_pairs_df = create_sim_features(data, mapping_keys, train_pairs_df)

# %%
#print(max(train_pairs_df['sim_word_movers']))
train_pairs_df = train_pairs_df.replace([float('inf'), float('-inf')], 9999)

# %%
train_pairs_df.to_csv("train_pairs_sim.csv")

# %%
print("TRAIN PAIRS: \n")
train_pairs_df.head()

# %%
# Encuentre el valor máximo
maximo = np.max(train_pairs_df['sim_word_movers'])

# Reemplace el valor máximo por un valor muy pequeño
train_pairs_df['sim_word_movers'][(train_pairs_df['sim_word_movers']) == maximo] = np.finfo(np.float64).tiny

# Encuentre el valor máximo nuevamente
segundo_maximo = np.max(train_pairs_df['sim_word_movers'])
print('Inf value replaced for -->', segundo_maximo)

train_pairs_df = train_pairs_df.replace([9999], segundo_maximo)

train_pairs_df.to_csv("train_pairs_sims.csv")

train_pairs_df = pd.read_csv('train_pairs_sims.csv', sep=',', index_col=0)

# %% [markdown]
# # Train a logistic regression model 
# GOAL: Identify pairs of news articles about the same case 

# %% [markdown]
# ### Create all possible pairs of articles with some restrictions:

# %% [markdown]
# - We determine that two articles can talk about the same case if they were published within +- 30 days

# %%
def combinations_df(df):
    # Filtrar las columnas necesarias
    df = df[['tweet_id', 'media', 'publication_date']]
    
    # Obtener todas las combinaciones posibles de tweet_id
    combinations = list(itertools.combinations(df['tweet_id'], 2))
    
    # Reordenar los valores de las combinaciones de menor a mayor
    combinations = [tuple(sorted(comb)) for comb in combinations]
    
    # Eliminar duplicados
    combinations = list(set(combinations))
    
    # Crear un dataframe con las combinaciones y sus valores correspondientes
    result = pd.DataFrame(columns=['tweet_id_A', 'publication_date_A', 'media_A', 'tweet_id_B', 'publication_date_B', 'media_B', 'time_interval'])
    
    for comb in combinations:
        tweet_id_A, tweet_id_B = comb
        data_A = df[df['tweet_id']==tweet_id_A].iloc[0]
        data_B = df[df['tweet_id']==tweet_id_B].iloc[0]
        
        # Agregar condicional para verificar si la diferencia entre las fechas de publicación es menor o igual a 30 días
        if abs(data_A['publication_date'] - data_B['publication_date']).days <= 7:
            result = result.append({'tweet_id_A': tweet_id_A,
                                    'publication_date_A': data_A['publication_date'],
                                    'media_A': data_A['media'],
                                    'tweet_id_B': tweet_id_B,
                                    'publication_date_B': data_B['publication_date'],
                                    'media_B': data_B['media'], 
                                    'time_interval': abs(data_A['publication_date'] - data_B['publication_date']).days}, ignore_index=True)
    
    return result


# %%
def get_combinations(df):
    # Crear un DataFrame vacío para almacenar las combinaciones
    result = pd.DataFrame(columns=['tweet_id_A', 'publication_date_A', 'media_A', 'tweet_id_B', 'publication_date_B', 'media_B', 'time_interval'])

    # Ordenar el DataFrame por fecha de publicación
    df = df.sort_values('publication_date')

    # Crear un diccionario para almacenar los tweets por fecha de publicación
    tweets_by_date = {}
    for i, tweet in df.iterrows():
        date = tweet['publication_date'].date()
        if date not in tweets_by_date:
            tweets_by_date[date] = []
        tweets_by_date[date].append(tweet)

    # Iterar por cada tweet
    for i, tweet_A in df.iterrows():
        print(f"Procesando tweet {i+1} de {len(df)}...")

        # Seleccionar los tweets que tengan una fecha de publicación dentro del rango [D, D+3]
        date = tweet_A['publication_date'].date()
        dates = [date, date + pd.Timedelta(days=1), date + pd.Timedelta(days=2), date + pd.Timedelta(days=3)]
        tweets_d = [tweet for d in dates if d in tweets_by_date for tweet in tweets_by_date[d] if tweet['tweet_id'] != tweet_A['tweet_id']]

        # Iterar por cada tweet que cumpla con la condición anterior
        for tweet_B in tweets_d:
            # Crear una nueva fila en el DataFrame de resultados
            result = result.append({
                'tweet_id_A': tweet_A['tweet_id'],
                'publication_date_A': tweet_A['publication_date'],
                'media_A': tweet_A['media'],
                'tweet_id_B': tweet_B['tweet_id'],
                'publication_date_B': tweet_B['publication_date'],
                'media_B': tweet_B['media'],
                'time_interval': abs(tweet_A['publication_date'] - tweet_B['publication_date']).days
            }, ignore_index=True)

    return result

# %%
pairs_df = get_combinations(articles_df)

pairs_df.to_csv("pairs_df_combinations.csv")

pairs_df = pd.read_csv('pairs_df_combinations.csv', sep=',', index_col=0)

# %%

# %%
pairs_sim_df = create_sim_features(data, mapping_keys, pairs_df)

pairs_sim_df.to_csv(f'OUTPUT_PATH/pairs_similarities_df.csv', index=False)

pairs_sim_df = pd.read_csv(f'../data/pairs_similarities_df.csv', sep=',')

# Encuentre el valor máximo
maximo = np.max(pairs_sim_df['sim_word_movers'])

# Reemplace el valor máximo por un valor muy pequeño
pairs_sim_df['sim_word_movers'][(pairs_sim_df['sim_word_movers']) == maximo] = np.finfo(np.float64).tiny

# Encuentre el valor máximo nuevamente
segundo_maximo = np.max(pairs_sim_df['sim_word_movers'])


pairs_sim_df = pairs_sim_df.replace(float('inf'), segundo_maximo)



# %%
# TRAIN A LOGISTIC REGRESSION CLASSIFIER
# Filter only checked cases for training
supervised_pairs_df = train_pairs_df[train_pairs_df['same_case'] != 2]  # 2 has been used for ambiguous cases
supervised_pairs_df = supervised_pairs_df[supervised_pairs_df['tweet_id_A'].isin(checked_tweet_id)]
supervised_pairs_df = supervised_pairs_df[supervised_pairs_df['tweet_id_B'].isin(checked_tweet_id)]

# Train the model
log_reg_model = train_log_reg(supervised_pairs_df, id_columns=['tweet_id_A', 'tweet_id_B'], target_column='same_case')

# PREDICT FOR ALL PAIRS OF NEWS
prediction = log_reg_model.predict(pairs_sim_df.drop(columns=['tweet_id_A', 'tweet_id_B', 'publication_date_A', 'publication_date_B', 'media_A', 'media_B', 'time_interval']))
proba_prediction = log_reg_model.predict_proba(pairs_sim_df.drop(columns=['tweet_id_A', 'tweet_id_B', 'publication_date_A', 'publication_date_B', 'media_A', 'media_B', 'time_interval']))[:,1]
pairs_sim_df.insert(len(pairs_sim_df.columns), 'same_case_pred', prediction)
pairs_sim_df.insert(len(pairs_sim_df.columns), 'same_case_pred_proba', proba_prediction)

# %%
pairs_sim_df.to_csv(f'{OUTPUT_PATH}/cases_pariwise_proba.csv', index=False)

#%%
#CONSTRUCT THE GRAPH 
cases_same_df = pairs_sim_df[pairs_sim_df['same_case_pred_proba']>0.85]
cases_same_df.info()

#Create and empty graph 
G = nx.Graph()

#Add a node for each tweet
tweets = set(pairs_sim_df['tweet_id_A']).union(set(pairs_sim_df['tweet_id_B']))
for tweet in tweets:
    G.add_node(tweet)

#Add an edge between two tweets if the prob is more than 0.8
for i, row in cases_same_df.iterrows():
    tweet_A = row['tweet_id_A']
    tweet_B = row['tweet_id_B']
    prob = row['same_case_pred_proba']
    if prob > 0.85:
        G.add_edge(tweet_A, tweet_B)

# Obtain the connected components of the graph
connected_components = list(nx.connected_components(G))

list_comp = []
for i, component in enumerate(connected_components):
    dicc_comp = {"cluster_id": i, "tweet_id": list(component)}
    list_comp.append(dicc_comp)

# Crear el dataframe con la información de los componentes conectados
df_components = pd.DataFrame(list_comp)


df_components.to_csv('../data/cases_conected_df_85.csv', index=False)




# %% [markdown]
# ---------------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt

# create the repeated_df DataFrame from the list of dictionaries
repeated_data_list = [new for media in repeated_data.keys() for new in repeated_data[media]]
repeated_df = pd.DataFrame.from_records(repeated_data_list)

# extract the tweet_id and media from the tweet_id_str and media fields, respectively
repeated_df['tweet_id'] = repeated_df['id'].str.split('_').str[-1]
repeated_df['media'] = repeated_df['media'].str.split('_').str[0]

# drop unnecessary columns
repeated_df.drop(['id', 'tweet_id'], axis=1, inplace=True)

# count the number of times each URL appears in the DataFrame
url_counts = repeated_df['url'].value_counts()

# count the number of URLs that appear x number of times
count_counts = url_counts.value_counts()

# plot the bar chart
ax = count_counts.plot.bar(figsize=(15, 10))
ax.set_title('Count of the number of urls that have been repeated x times')
ax.set_xlabel('Number of times the url is repeated')
ax.set_ylabel('URL count')

# %%
#CONSTRUCT A DATAFRAME WITH ALL THE REPEATED ARTICLES

"""
repeated_df = pd.DataFrame()

for media in repeated_data.keys():
    for new in repeated_data[media]:
        tweet_id = new['id']
        title = normalize_string(new['title'])
        headline = normalize_string(new['headline'])
        url = new['url']
        publication_date = new['publication_date']
        article = normalize_string(new['article'])
        
        repeated_df = repeated_df.append({'tweet_id': tweet_id, 
                                          'media': media, 
                                          'title': title, 
                                          'headline': headline, 
                                          'url': url,
                                          'article': article,
                                          'publication_date': publication_date},
                                           ignore_index=True)
        
repeated_df['tweet_id'] = repeated_df['tweet_id'].str.split('_').str[-1]
repeated_df['media'] = repeated_df['media'].str.split('_').str[0]

# %%
repeated_df.head()

# %%
repeated_df2 = pd.DataFrame(repeated_df.groupby(["url"])["url"].count())
repeated_df2.columns = ['count']

plt.figure(figsize=(15,10))
counts, edges, bars = plt.hist(repeated_df2)
plt.bar_label(bars, padding=5, fontsize=12)
plt.title('Count of the number of urls that have been repeated x times')
plt.xlabel('Number of times the url is repeated')
plt.ylabel('URL count ')
ax = plt.subplot()
ax.set_xticks(repeated_df2['count'])
plt.show()

# %% [markdown]
# - 1590 urls s'han repetit 1 cop
# - 292 urls s'han repetit 2 cops
# - 74 urls s'han repetit 3 cops
# - 19 urls s'han repetit 4 cops
# - 7 urls s'han repetit 5 cops
# - 4 urls s'han repetit 6 cops
# - 1 url s'ha repetit 8 cops
# - 1 url s'ha repetit 9 cops
# - 1 url s'ha repetit 11 cops

# %%
len(articles_df['url'].unique())==len(articles_df)

# %%
articles_df.info()

"""
