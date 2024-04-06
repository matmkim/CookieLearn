from utils import process_text, WeightedNetworkDissimilarity
from similarity import Text_Similarity, Network_Similarity
from network import ArticleNetwork, DomainNetwork, EventNetwork
import pickle, json
import pandas as pd
import networkx as nx

topics=["한동훈_20240320_20240325","조국_20240320_20240325","이재명_20240320_20240325","의료_20240301_20240331","양문석_20240315_20240404"]#,"이종섭_20240320_20240325"]
graphs=[]

for _ in topics:
  with open("./result/models/"+f'DoNet_{_}.p', 'rb') as f:
    DoNet = pickle.load(f)
  mapping={_:_.strip() for _ in list(DoNet.nx_graph.nodes())}
  nx.relabel_nodes(DoNet.nx_graph, mapping, copy=False)
  graphs.append(DoNet.nx_graph)

common_nodes = set(list(graphs[1].nodes()))
for graph in graphs[1:]:
    common_nodes.intersection_update(graph.nodes())

subgraphs = [graph.subgraph(common_nodes).copy() for graph in graphs]

df_event=pd.DataFrame({"event_name":topics,"graph":subgraphs})

event_network=EventNetwork("22대총선_주요인물_의료",df_event)

event_network.fit()
event_network.build()

event_network.clustering()
event_network.embedding(dim=2)
with open(f"./result/images/event_graph_{event_network.category}_1.svg", 'wt') as file:
    svg_content=event_network.plot_network()
    file.write(svg_content)
with open(f"./result/images/event_graph_{event_network.category}_2.svg", 'wt') as file:
    svg_content=event_network.plot_embedding()
    file.write(svg_content)
event_network.export(file_path="./result")
