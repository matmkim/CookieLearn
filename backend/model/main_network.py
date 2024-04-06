from utils import process_text
from similarity import Text_Similarity, Network_Similarity
from network import ArticleNetwork, DomainNetwork, EventNetwork
import pandas as pd
import pickle

# keyword가 network 다루는 주제이름: (검색어)_(시작일)_(종료일)
keyword="의료_20240301_20240331"

df=pd.read_csv(f"./output/Article_{keyword}.csv")
df["processed_text"] = df.apply(lambda x: process_text(x['text_headline'] +". "+ x['text_sentence']), axis=1)
article_network=ArticleNetwork(f"{keyword}",df)
article_network.fit()
article_network.build()
article_network.clustering()
article_network.embedding()
with open(f"./result/images/article_graph_{article_network.keyword}.svg", 'wt') as file:
  svg_content=article_network.plot_network()
  file.write(svg_content)
article_network.export()
with open("./result/models/"+f'ArNet_{article_network.keyword}.p', 'wb') as f:
  pickle.dump(article_network, f)

domain_network=DomainNetwork(article_network)
domain_network.build()
domain_network.clustering()
domain_network.embedding()
with open(f"./result/images/domain_graph_{domain_network.keyword}.svg", 'wt') as file:
  svg_content=domain_network.plot_network()
  file.write(svg_content)
domain_network.export()
with open("./result/models/"+f'DoNet_{domain_network.keyword}.p', 'wb') as f:
  pickle.dump(domain_network, f)
