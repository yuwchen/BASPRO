import os
import json
import numpy as np
import pandas as pd
import torch
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM
import logging
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)

class Perplexity_Checker(object):
    def __init__(self, MODEL_PATH, DEVICE):
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        self.model = BertForMaskedLM.from_pretrained(MODEL_PATH)
        self.model.eval()
        self.model.to(DEVICE)
        self.DEVICE = DEVICE

    def sentence_preprocese(self, text):
        # Tokenize input
        tokenized_text = np.array(self.tokenizer.tokenize(text))
        find_sep = np.argwhere(tokenized_text == '[SEP]')
        segments_ids = np.zeros(tokenized_text.shape, dtype=int)
        if find_sep.size == 1:
            start_point = 1
        else:
            start_point = find_sep[0, 0] + 1
            segments_ids[start_point:] = 1

        end_point = tokenized_text.size - 1

        # Mask a token that we will try to predict back with `BertForMaskedLM`
        tokenized_text = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        masked_texts = []
        for masked_index in range(start_point, end_point):
            new_tokenized_text = np.array(tokenized_text, dtype=int)
            new_tokenized_text[masked_index] = self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
            masked_texts.append(new_tokenized_text)

        segments_ids = np.tile(segments_ids, (end_point - start_point, 1))

        return masked_texts, segments_ids, start_point, end_point, tokenized_text[start_point:end_point]

    def perplexity(self, text):
        indexed_tokens, segments_ids, start_point, end_point, real_indexs = self.sentence_preprocese(text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor(indexed_tokens)
        segments_tensors = torch.tensor(segments_ids)

        # If you have a GPU, put everything on cuda
        tokens_tensor = tokens_tensor.to(self.DEVICE)
        segments_tensors = segments_tensors.to(self.DEVICE)

        # Predict all tokens
        with torch.no_grad():
            outputs = self.model(tokens_tensor, token_type_ids=segments_tensors)
            predictions = torch.softmax(outputs[0], -1)

        total_perplexity = 0
        for i, step in enumerate(range(start_point, end_point)):
            # predicted_index = torch.argmax(predictions[i, step]).item()
            # predicted_token = self.tokenizer.convert_ids_to_tokens([predicted_index])[0]
            # print(predicted_token)
            total_perplexity += np.log(predictions[i, step, real_indexs[i]].item())

        # total_perplexity = np.exp(-total_perplexity / (end_point - start_point))
        total_perplexity = -total_perplexity / (end_point - start_point)
        return total_perplexity


def calculate_perplexity(input_path, DEVICE): 
    MODEL_PATH = 'bert-base-chinese'
    text_formatter = lambda x: "[CLS]{} [SEP]".format(x)
    pchecker = Perplexity_Checker(MODEL_PATH, DEVICE)
    sen_list = open(input_path,'r').read().splitlines()
    f_out = open(input_path.replace(".txt","_per.txt"),'w')
    for line in tqdm(sen_list):
        sen = line.split("#")[1]
        score = pchecker.perplexity(text_formatter(sen))
        f_out.write(line+"#"+str(score)+"\n")


