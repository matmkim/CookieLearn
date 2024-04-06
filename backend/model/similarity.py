import warnings
warnings.filterwarnings('ignore')
from functools import reduce
from itertools import product

import importlib, sys, pandas as pd, numpy as np, spacy, re, itertools, pickle, string
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import models, util, datasets, evaluation, losses

from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import SpectralCoclustering

from Levenshtein import distance as l_dist
from fuzzywuzzy import fuzz
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import sknetwork as skn
import networkx as nx

import konlpy
from ekonlpy.sentiment import KSA
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

from tqdm import tqdm

from utils import WeightedNetworkDissimilarity

class Text_Similarity:
    def __init__(self, semantic_threshold=0.7, sentiment_threshold=0.1,
                 method='ordered'):
        self.semantic_threshold = semantic_threshold
        self.sentiment_threshold = sentiment_threshold
        self.method = method
        self.memory=dict()
        self.tokenizer = AutoTokenizer.from_pretrained("BM-K/KoSimCSE-roberta")
        self.model = AutoModel.from_pretrained("BM-K/KoSimCSE-roberta")
        self.ksa=KSA()

    def get_text_similarity(self, article_1, article_2):
        # Get the embeddings and sentiments of the two articles
        embeddings_1, sentiments_1 = self._get_text_embeddings(article_1)
        embeddings_2, sentiments_2 = self._get_text_embeddings(article_2)

        matches = np.argwhere((cosine_similarity(embeddings_1, embeddings_2) > self.semantic_threshold) &
                             (manhattan_distances(sentiments_1, sentiments_2) < self.sentiment_threshold))
        if self.method == "ordered":
            sim = self._get_edit_similarity(embeddings_1, embeddings_2, matches)
        elif self.method == "unordered":
            sim = self._get_overlap_similarity(embeddings_1, embeddings_2, matches)
        else:
            raise("method must be 'ordered' or 'unordered'")
        return sim

    def _get_text_embeddings(self,article):
      if tuple(article) in self.memory:
        return self.memory[tuple(article)]

      inputs = self.tokenizer(article, padding=True, truncation=True, return_tensors="pt")
      with torch.no_grad():
        embeddings = self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

      sentiments=[[list(self.ksa.get_score(self.ksa.tokenize(_)).values())[2]] for _ in article]

      self.memory[tuple(article)]=(embeddings.detach().numpy(), np.array(sentiments))

      return embeddings.detach().numpy(), np.array(sentiments)

    def _get_edit_similarity(self, embeddings_1, embeddings_2, matches):
        article_1_dict = {}
        article_2_dict = {}

        alphabet = list(string.ascii_lowercase + string.ascii_uppercase  + string.digits + "!@#$%^&*()-+=<>?:;"+ "".join([chr(i) for i in range(161,10000)]))

        for i in range(len(matches)):
            symbol = alphabet.pop(0)
            article_1_dict[matches[i][0]] = symbol
            article_2_dict[matches[i][1]] = symbol

        article_1_string = "".join([article_1_dict[i] if i in article_1_dict else alphabet.pop(0) for i in range(len(embeddings_1))])
        article_2_string = "".join([article_2_dict[i] if i in article_2_dict else alphabet.pop(0) for i in range(len(embeddings_2))])

        sim = fuzz.partial_ratio(article_1_string, article_2_string)/100
        return sim

    def _get_overlap_similarity(self, embeddings_1, embeddings_2, matches):
        return len(matches)/ min(len(embeddings_1), len(embeddings_2))
    
class Network_Similarity:
  def __init__(self,dim=3,weight=[0.45,0.45,0.1],method="WD"):
    self.memory=dict()
    self.spectral = skn.embedding.Spectral(n_components=dim)
    self.weight=weight
    self.dim=dim
    self.method=method

  def _get_network_embedding(self,adjacency):
    if str(adjacency) in self.memory:
      return self.memory[str(adjacency)]
    self.memory[str(adjacency)]=self.spectral.fit_transform(adjacency)
    return self.spectral.fit_transform(adjacency)

  def get_network_similarity(self,g1,g2):
    # expected networkx graph object
    if self.method=="spectral":
      embedding1=self._get_network_embedding(g1.to_numpy_array())
      embedding2=self._get_network_embedding(g2.to_numpy_array())
      return 1/np.sqrt(np.mean(cosine_similarity(embedding1, embedding2)**2))

    elif self.method=="WD":
      wnd=WeightedNetworkDissimilarity(g1,g2,self.weight)
      dissimilarity=wnd.compute_WD_metric()
      return 1-dissimilarity # return dissimilarity

    else:
      raise AttributeError