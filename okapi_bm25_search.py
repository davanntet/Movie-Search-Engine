import pandas as pd
from nltk.tokenize import word_tokenize
import sqlite3
import pickle
import re
import time

class OkapiBM25Search:
    filename = 'bm25_model.sav'
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.bm25 = pickle.load(open(self.filename, 'rb'))
            conn = sqlite3.connect("datasets/netflix_titles.db")
            self.storys = pd.read_sql("select * from netflix_titles", conn)
        
    def __filter(self,query):
        # select year from query
        # if no year is found, return empty list
        # if year is found, return a list of years

        query = str(query)
        year = re.findall(r'(?:19|20)\d{2}', query)
        year = list(map(int, year))  # convert each element to int
        return year
    
    def search(self,query):
        start_time = time.time()
        tokenized_query = word_tokenize(query.lower())
        doc_scores = self.bm25.get_scores(tokenized_query)
        end_time = time.time()
        self.storys['score'] = doc_scores
        years = self.__filter(query)
        result = self.storys.nlargest(30, 'score')
        if len(years) > 0:
            result = result[result['release_year'].isin(years)]
        return result,(end_time-start_time)*1000
