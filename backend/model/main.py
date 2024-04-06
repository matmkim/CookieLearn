from korea_news_crawler.articlecrawler import ArticleCrawler
from bias_evaluation import BiasEvaluation
import os

if __name__ == '__main__':
	keyword = input('키워드를 입력하시오: ')
	Crawler = ArticleCrawler()
	Crawler.set_category('정치')
	Crawler.set_date_range('2024-03-01', '2024-03-31')
	Crawler.start(keyword)

	main_dir = './'
	model_dir = './model_save/model_20240404/'
	csv_dir = './output/'
	out_dir = '../app/analysis_result'

	bias = BiasEvaluation(main_dir, model_dir)
	for csv_path in os.listdir(csv_dir):
			bias.evaluate_csv(csv_dir + csv_path, out_dir, os.path.splitext(os.path.basename(csv_path))[0], encoding = 'utf-8', verbose = True)
	
