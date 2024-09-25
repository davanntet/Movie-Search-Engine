# import required libraries
from sentence_transformers import SentenceTransformer, util
import torch
import time
import pandas as pd
import sqlite3
import re

class Search:
    def __init__(self):
        # initialize necessary variables
        # load corpus embeddings from file
        self.corpus_embeddings = torch.load('corpus_embeddings.pt', map_location=torch.device('cpu'))
        # load dataset
        conn = sqlite3.connect("datasets/netflix_titles.db")
        self.storys = storys = pd.read_sql("select * from netflix_titles", conn)
        # load sentence transformer model
        self.model = SentenceTransformer('sentence-transformers/LaBSE')

    
    def __filter(self,query):
        # select year from query
        # if no year is found, return empty list
        # if year is found, return a list of years

        query = str(query)
        year = re.findall(r'(?:19|20)\d{2}', query)
        year = list(map(int, year))  # convert each element to int
        return year
    


    def search(self,inp_question):
        # here is the the search function
        # it will get the most similar 75 storys to the search query and then select storeys that have a score higher than the mean score
        # if search query contains a year, it will filter the results based on the year
        # results are sorted by ranking
        # the search function returns a pandas dataframe

        start_time = time.time()
        question_embedding = self.model.encode(inp_question, convert_to_tensor=True)
        hits = util.semantic_search(question_embedding, self.corpus_embeddings, top_k=75, score_function=util.dot_score)
        end_time = time.time()
        mean =  sum(list([float(x['score']) for x in hits[0]]))/len(hits[0])
        idx, score = zip(*((x['corpus_id'], x['score']) for x in hits[0] if x['score'] > mean))
        idx = list(idx)
        time_search = (end_time-start_time)*1000 # convert to milliseconds
        years = self.__filter(inp_question)
        results = self.storys.iloc[idx].copy()
        results['score'] = score
        if len(years) == 0:
            return results, time_search
        else:
            return results[results['release_year'].isin(filter(inp_question))],time_search
    
    def __str__(self) -> str:
        return "I am a search engine for searching netflix movies and tv shows"
    
class SingletonSearch:
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state
        if not hasattr(self, 'initialized'):
            self.initialized = True
            # initialize necessary variables
            # load corpus embeddings from file
            self.corpus_embeddings = torch.load('corpus_embeddings.pt', map_location=torch.device('cpu'))
            # load dataset
            conn = sqlite3.connect("datasets/netflix_titles.db")
            self.storys = pd.read_sql("select * from netflix_titles", conn)
            # load sentence transformer model
            self.model = SentenceTransformer('sentence-transformers/LaBSE')

    def __filter(self,query):
        query = str(query)
        year = re.findall(r'(?:19|20)\d{2}', query)
        year = list(map(int, year))  # convert each element to int
        return year

    def search(self,inp_question):
        start_time = time.time()
        question_embedding = self.model.encode(inp_question, convert_to_tensor=True)
        hits = util.semantic_search(question_embedding, self.corpus_embeddings, top_k=75, score_function=util.dot_score)
        end_time = time.time()
        mean =  sum(list([float(x['score']) for x in hits[0]]))/len(hits[0])
        idx, score = zip(*((x['corpus_id'], x['score']) for x in hits[0] if x['score'] > mean))
        idx = list(idx)
        time_search = (end_time-start_time)*1000 # convert to milliseconds
        years = self.__filter(inp_question)
        results = self.storys.iloc[idx].copy()
        results['score'] = score
        if len(years) == 0:
            return results, time_search
        else:
            return results[results['release_year'].isin(self.__filter(inp_question))],time_search

    def __str__(self) -> str:
        return "I am a search engine for searching netflix movies and tv shows"