import warnings
warnings.filterwarnings('ignore')
from functools import reduce
from itertools import product
import json

import importlib, sys, pandas as pd, numpy as np, spacy, re, itertools, pickle, string
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import SpectralCoclustering

from Levenshtein import distance as l_dist

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import sknetwork as skn
import networkx as nx

from tqdm import tqdm

from similarity import Text_Similarity, Network_Similarity

class ArticleNetwork:
  def __init__(self,keyword,df_crawling):
    self.similarities=[]
    self.nx_graph=nx.Graph()
    self.df=df_crawling
    self.keyword=keyword
    self.articles=list(df_crawling['content_url'])

  def fit(self,semantic_threshold=0.7, sentiment_threshold=0.1,method="ordered"):
    # compute similarites between each article
    self.similarities=[]
    text_sim = Text_Similarity(semantic_threshold, sentiment_threshold,method)
    for i in tqdm(range(len(self.df))):
      for j in tqdm(range(i+1,len(self.df))):
          self.similarities.append((self.df['content_url'][i], self.df['content_url'][j], text_sim.get_text_similarity(self.df['processed_text'][i], self.df['processed_text'][j])))

  def build(self):
    # construct weighted graph with edge list, both networkx object and skn object
    self.nx_graph.add_weighted_edges_from(self.similarities)
    self.skn_graph=skn.data.from_edge_list(self.similarities, directed=False, weighted=True)

  def get_network(self,graph_type="nx"):
    if graph_type=="nx":
      return self.nx_graph
    elif graph_type=="skn":
      return self.skn_graph
    else:
      raise AttributeError

  def clustering(self):
    louvain = skn.clustering.Louvain()
    self.clusters = louvain.fit_predict(self.skn_graph.adjacency)
    df['article_cluster'] = self.clusters

  def get_cluster(self):
    return dict(zip(self.articles,self.clusters))

  def plot_network(self,width=550,height=550,PageRank=True):
    # network visulalization, PageRank score used to make distance valid
    if PageRank:
      pagerank = skn.ranking.PageRank()
      scores = pagerank.fit_predict(self.skn_graph.adjacency)
      image = skn.visualization.svg_graph(self.skn_graph.adjacency,scores=scores,labels=self.clusters,width=width, height=height)
      return image
    else:
      image = skn.visualization.svg_graph(self.skn_graph.adjacency,labels=self.clusters,width=width, height=height)
      return image

  def embedding(self,dim=2,normalized=False):
    spectral = skn.embedding.Spectral(dim, normalized=normalized)
    embedding = spectral.fit_transform(self.skn_graph.adjacency)
    self.embeds=embedding

  def get_embedding(self):
    return dict(zip(self.articles,self.embeds))

  def plot_embedding(self,width=550,height=550,node_size=5):
    image = skn.visualization.svg_graph(position=self.embeds, labels=self.clusters, node_size=node_size, width=width, height=height)
    return image

  def export(self,file_path=""):
    self.df.to_csv(file_path+"/"+f'article_dataframe_{self.keyword}.csv')
    
    with open(file_path+"/"+f'article_similarity_{self.keyword}.json', 'w') as json_file:
      json.dump(self.similarities, json_file)

    pd.DataFrame(self.skn_graph.adjacency.todense(), index=self.articles, columns=self.articles).to_csv(file_path+"/"+f"article_adjacency_{self.keyword}.csv")

class DomainNetwork:
  def __init__(self,article_network:ArticleNetwork):
    self.article_network=article_network
    self.df=article_network.df
    self.keyword=article_network.keyword
    self.nx_graph=nx.Graph()
    self.similarities=[]

  def build(self):
    self.similarities=[]
    domain_to_article_graph = skn.data.from_edge_list(list(self.df[['text_company', 'content_url']].itertuples(index=False, name=None)), bipartite=True)
    domain_to_domain_graph = np.sqrt(domain_to_article_graph.biadjacency) * self.article_network.skn_graph.adjacency * np.sqrt(domain_to_article_graph.biadjacency).T

    self.domain_to_article_graph=domain_to_article_graph
    self.domains=domain_to_article_graph.names_row
    self.adjacency=domain_to_domain_graph

    for node_name in self.domains:
        self.nx_graph.add_node(node_name)

    for i, node_i in enumerate(self.domains):
        for j, node_j in enumerate(self.domains):
            if self.adjacency[i, j] != 0:
                self.nx_graph.add_edge(node_i, node_j, weight=self.adjacency[i, j])
            self.similarities.append((node_i, node_j, self.adjacency[i, j]))
    
    self.skn_graph=skn.data.from_edge_list(self.similarities, directed=False, weighted=True)

  def get_network(self,graph_type="nx"):
    if graph_type=="nx":
      return self.nx_graph
    elif graph_type=="skn":
      return self.skn_graph
    else:
      raise AttributeError

  def clustering(self):
    louvain = skn.clustering.Louvain()
    domain_clusters = louvain.fit_predict(self.adjacency)
    self.clusters=domain_clusters

  def get_cluster(self):
    return dict(zip(self.domains,self.clusters))

  def plot_network(self,width=550,height=550,PageRank=True):
    if PageRank:
      pagerank = skn.ranking.PageRank()
      scores = pagerank.fit_predict(self.adjacency)
      image = skn.visualization.svg_graph(self.adjacency,scores=scores,names=self.domains,labels=self.clusters,width=width, height=height)
      return image
    else:
      image = skn.visualization.svg_graph(self.skn_graph.adjacency,names=self.domains,labels=self.clusters,width=width, height=height)
      return image

  def embedding(self,dim=2,normalized=False):
    spectral = skn.embedding.Spectral(dim, normalized=normalized)
    embedding = spectral.fit_transform(self.adjacency)
    self.embeds=embedding

  def get_embedding(self):
    return dict(zip(self.domains,self.embeds))

  def plot_embedding(self,width=550,height=550,node_size=5):
    image = skn.visualization.svg_graph(position=self.embeds, names=self.domains, labels=self.clusters, node_size=node_size, width=width, height=height)
    return image

  def export(self,file_path=""):
    pd.DataFrame({"domain_name":self.domains,"domain_cluster":self.clusters}).to_csv(file_path+"/"+f'domain_dataframe_{self.keyword}.csv')
    
    with open(file_path+"/"+f'domaint_similarity_{self.keyword}.json', 'w') as json_file:
      json.dump(self.similarities, json_file)

    pd.DataFrame(self.adjacency.todense(), index=self.domains, columns=self.domains).to_csv(file_path+"/"+f"domain_adjacency_{self.keyword}.csv")

class EventNetwork:
  # df_event=pd.DataFrame({'event_name':str,'graph':nx.Graph})
  def __init__(self,category,df_event):
    self.similarities=[]
    self.nx_graph=nx.Graph()
    self.df=df_event
    self.events=list(df_event['event_name'])
    self.category=category

  def fit(self,dim=3,weight=[0.45,0.45,0.1],method="WD"):
    self.similarities=[]
    net_sim=Network_Similarity(dim=3,weight=[0.45,0.45,0.1],method="WD")

    for i in tqdm(range(len(self.events))):
        for j in tqdm(range(i+1,len(self.events))):
            self.similarities.append((self.events[i], self.events[j], net_sim.get_network_similarity(self.df['graph'][i], self.df['graph'][j])))

  def build(self):
    # construct weighted graph with edge list, both networkx object and skn object
    self.nx_graph.add_weighted_edges_from(self.similarities)
    self.skn_graph=skn.data.from_edge_list(self.similarities, directed=False, weighted=True)

  def get_network(self,graph_type="nx"):
    if graph_type=="nx":
      return self.nx_graph
    elif graph_type=="skn":
      return self.skn_graph
    else:
      raise AttributeError

  def clustering(self):
    louvain = skn.clustering.Louvain()
    self.clusters = louvain.fit_predict(self.skn_graph.adjacency)
    self.df['event_cluster'] = self.clusters

  def get_cluster(self):
    return dict(zip(self.events,self.clusters))

  def plot_network(self,width=550,height=550,PageRank=True):
    if PageRank:
      pagerank = skn.ranking.PageRank()
      scores = pagerank.fit_predict(self.skn_graph.adjacency)
      image = skn.visualization.svg_graph(self.skn_graph.adjacency,names=self.events,scores=scores,labels=self.clusters,width=width, height=height)
      return image
    else:
      image = skn.visualization.svg_graph(self.skn_graph.adjacency,names=self.events,labels=self.clusters,width=width, height=height)
      return image

  def embedding(self,dim=2,normalized=False):
    spectral = skn.embedding.Spectral(dim, normalized=normalized)
    embedding = spectral.fit_transform(self.skn_graph.adjacency)
    self.embeds=embedding

  def get_embedding(self):
    return dict(zip(self.events,self.embeds))

  def plot_embedding(self,width=550,height=550,node_size=5):
    image = skn.visualization.svg_graph(position=self.embeds,names=self.events, labels=self.clusters, node_size=node_size, width=width, height=height)
    return image

  def export(self,file_path=""):
    self.df.drop(columns=["graph"]).to_csv(file_path+"/"+f'event_dataframe_{self.category}.csv')
    
    with open(file_path+"/"+f'event_similarity_{self.category}.json', 'w') as json_file:
      json.dump(self.similarities, json_file)

    pd.DataFrame(self.skn_graph.adjacency.todense(), index=self.events, columns=self.events).to_csv(file_path+"/"+f"event_adjacency_{self.category}.csv")
