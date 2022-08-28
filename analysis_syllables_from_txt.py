

import numpy as np
import pickle
from pypinyin import pinyin, lazy_pinyin, Style


with open('./gt_syllable_with_tone.pickle', 'rb') as fp:
    syllable = pickle.load(fp)

syllable.pop('々', None)

syllable_key = {}
for idx, key in enumerate(list(syllable.keys())):
    syllable_key[key] = idx


truth_syllable = np.zeros((len(syllable_key), ))
for key, value in syllable.items():
    truth_syllable[syllable_key[key]]=value

def map_syllable(syllable_sen, syllable_key):
    syllable_vector = np.zeros((len(syllable_key), ))
    for key in syllable_sen:
        syllable_vector[syllable_key[key]]+=1
    return syllable_vector

def cosine_similarity(input):

    cos_sim = np.dot(input, truth_syllable)/(np.linalg.norm(input)*np.linalg.norm(truth_syllable))
    return cos_sim

def calculate_score(result_syllable):
    set_syllable = np.sum(result_syllable, axis=1)
    chro_syllable = np.sum(set_syllable, axis=0)
    distributation_score = cosine_similarity(chro_syllable)
    converage_score = np.count_nonzero(set_syllable)/result_syllable.shape[0]
    converage_score_corpus = np.count_nonzero(chro_syllable)
    distribution_score_set = [cosine_similarity(the_set_syllable) for the_set_syllable in set_syllable]
    set_distribution_mean = np.mean(distribution_score_set)
    set_distribution_std = np.std(distribution_score_set)

    return distributation_score, converage_score, converage_score_corpus, set_distribution_mean, set_distribution_std

def get_corpus_syllable(corpus):
    import opencc
    converter = opencc.OpenCC('s2t.json')

    corpus_syllable = []
    corpus_base_syllable = []
    for corpus_set in corpus:
        set_syllable = []
        for corpus_sen in corpus_set:
                
            corpus_sen = converter.convert(corpus_sen)
            base_syllable = lazy_pinyin(corpus_sen)
            corpus_base_syllable.extend(base_syllable)
            the_syllable = pinyin(corpus_sen, style=Style.TONE3, heteronym=False)
            the_syllable = [x[-1] for x in the_syllable]
            #print(the_syllable, corpus_sen)
            syllable_vector = map_syllable(the_syllable,syllable_key)
            set_syllable.append(syllable_vector)
        corpus_syllable.append(set_syllable)
    corpus_syllable = np.asarray(corpus_syllable)
    print('base syllable:', len(set(corpus_base_syllable)))
    print(corpus_syllable.shape)
    return corpus_syllable

def calculate_scores(path):
    f = open(path,'r').read().splitlines()
    corpus = []
    for line in f[1:]:
        sen = line.split(":")[-1]
        corpus.append(sen)
    corpus = np.asarray(corpus)
    corpus = np.reshape(corpus,(5,5))
    corpus_syllable = get_corpus_syllable(corpus)
    distributation_score, converage_score, converage_score_corpus, set_distribution_mean, set_distribution_std = calculate_score(corpus_syllable)
    print("distributation_score",distributation_score)
    print("converage_score",converage_score)
    print("converage_score_corpus",converage_score_corpus)
    print("set_distribution_mean",set_distribution_mean)
    print("set_distribution_std",set_distribution_std)

calculate_scores("/Users/yuwen/Desktop/speech_corpus/code_at_github/BASPRO-main/output/corpus.txt")