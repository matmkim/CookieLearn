### import packages (pretty messy) ###
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, SequentialSampler
import tensorflow as tf
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import numpy as np
from tqdm.notebook import tqdm
import pandas as pd
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
import csv
import kss # Korean Sentence Splitter: https://github.com/likejazz/korean-sentence-splitter

###########################################################
### Class explanation
### main_dir: main directory of the file
### model_dir: directory of the model
### batch_size: batch size (default: 32)
### max_len: max length of BERT model (default: 256)(do not change)
###
### Use example: 
### bias = BiasEvaluation('C:/Users/nsk01/OneDrive/바탕 화면/Hackathon/', 'C:/Users/nsk01/OneDrive/바탕 화면/Hackathon/model_save/model_20240404/')
### csv_path = bias.main_dir + 'articles/election.csv'
### out_dir = bias.main_dir + 'output/'
### bias.evaluate_csv(csv_path, out_dir, 'election'):
###########################################################
### Method explanation: evaluate
### Evaluates the political bias of list of articles, and returns the bias
###
### input:
### articles: list of strings, containing each articles
### verbose: if set to True, prints out current status (default: False)
###
### return: list of float between 0 and 1, containing political bias of each article
###########################################################
### Method explanation: evaluate
### Evaluates the political bias of articles in csv file, and saves new csv file that contains the bias
###
### input:
### csv_path: path of csv file to read
### out_dir: directory to save the output csv files
### out_name: name of output csv files
### encoding: encoding of the csv file (default: 'UTF-8')v
### verbose: if set to True, prints out current status (default: False)
###
### return: None (output is saved as csv file inside out_dir directory)
###########################################################
class BiasEvaluation():
    def __init__(self, main_dir, model_dir, batch_size = 32, max_len = 256):
        print("Initializing Bias Evaluation model")
        self.main_dir, self.model_dir = main_dir, model_dir
        print("Main directory is set to : " + self.main_dir)
        print("Model directory is set to : " + self.model_dir)
        
        # Set torch device
        self.update_device()
        
        # Load a trained model
        self.load_model()
        
        self.batch_size, self.max_len = batch_size, max_len
        
        print("Initialization complete: max_len %s, batch size %s" % (self.max_len, self.batch_size))
    
    def update_device(self, device_name = None):
        if device_name: self.device = torch.device(device_name)
        else: self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device set to : %s" % torch.cuda.get_device_name(device))
        
    def load_model(self):
        print("Loading pre-trained model from : " + self.model_dir)
        self.model = BertForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model.to(device)
        self.model.eval()
        
    def evaluate(self, articles, verbose = False): # evaluates political bias of list of articles
        result = []
        if verbose: print("Processing %s articles..." % len(articles))
        for i in range(len(articles)):
            article = articles[i]
            if verbose: print("Processing article %s / %s ..." % ((i+1), len(articles)))
            # Tokenize all of the sentences and map the tokens to thier word IDs.
            input_ids = []
            attention_masks = []
            sentences = kss.split_sentences(article) # split article into multiple sentences
            if verbose: print("Article #%s has %s sentences" % ((i+1), len(sentences)))
            if verbose: print("Tokenizing the sentences...")
            for sent in sentences:
                encoded_dict = self.tokenizer.encode_plus(
              sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = self.max_len,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                            truncation = True
                )
                input_ids.append(encoded_dict['input_ids'])
                attention_masks.append(encoded_dict['attention_mask'])
    
            input_ids = torch.cat(input_ids, dim=0)
            attention_masks = torch.cat(attention_masks, dim=0)
    
            prediction_data = TensorDataset(input_ids, attention_masks)
            prediction_sampler = SequentialSampler(prediction_data)
            prediction_loader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=self.batch_size)
            
            # Predict
            if verbose: print("Evaluating the sentences...")
            evaluation = 0
            for batch in prediction_loader:
                batch = tuple(t.to(device) for t in batch)
    
                b_input_ids, b_input_mask = batch
    
                with torch.no_grad():
                    outputs = self.model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)
                logits = outputs[0]
    
                logits = logits.detach().cpu().numpy()

                sigmoid = lambda x:1 / (1 + np.exp(-x))
                evaluation += sum([sigmoid(val[1] - val[0]) for val in list(logits)])
            
            if verbose: print("Bias of article #%s is %f" % ((i+1), evaluation / len(sentences)))
            result.append(evaluation / len(sentences))
    
        return result
    
    
    def evaluate_csv(self, csv_path, out_dir, out_name, encoding = 'UTF-8', verbose = False):
        dataset = pd.read_csv(csv_path, names=['time', 'category_name', 'text_company', 'text_headline', 'text_sentence', 'content_url'], encoding = encoding).drop(0)
        if max_len: dataset = dataset.sample(max_len)
        titles = dataset.text_headline.values
        texts = dataset.text_sentence.values
        if verbose: print("Processing %s articles from CSV file..." % len(titles))
    
        eval = self.evaluate(texts, verbose)
    
        list = dataset.values.tolist()
        pair_data, notext_data = [], []
        for i in range(len(eval)):
            pair_data.append([list[i][3], list[i][4], eval[i].item()])
            notext_data.append(list[i][:4]+list[i][5:]+[eval[i].item()])
    
        pair_data.sort(key = lambda x: x[2]) # sort the data by the political bias
        pair_data = [['text_headline', 'text_sentence', 'bias_label']] + pair_data
        notext_data = [['time',	'category_name', 'text_company', 'text_headline', 'content_url', "bias_label"]] + notext_data
    
        csv_path_1 = out_dir + out_name + "_pair_sorted.csv"
        csv_path_2 = out_dir + out_name + "_notext.csv"
        
        with open(csv_path_1, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerows(pair_data)
            csvfile.close()
        
        with open(csv_path_2, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerows(notext_data)
            csvfile.close()
    
        if verbose: print("Save pair data at : "+csv_path_1)
        print("Saved notext data at : "+csv_path_2)
 