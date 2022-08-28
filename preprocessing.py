import re
import os
import numpy as np
import pickle
from tqdm import tqdm


def general_filter(input_path):
    """
    The general filter filter out sentences that are not in Chinese 
    and only remain sentences have exactly 10 characters
    input (str): path to input txt files (Txt format see raw_data.txt for example)
    output (.txt): the remaining sentences with index
    """

    f_out = open('result_s1.txt','w') #output file name
    
    idx=1
    for sen in open(input_path,'r').read().splitlines():

        #skip sentecnes contain non-Chinese characters 
        if re.search('[a-zA-Z]', sen) or (not sen.isalpha()):
            continue
        #skip sentences have spaces
        if ' ' in sen:
            continue
        #skip sentances that length is not 10
        if len(sen) != 10:
            continue
        f_out.write('Idx_'+str(idx)+"#"+sen+"\n")
        idx+=1

            
def jieba_seg(input_path):
    """
    jieba segmentation
    input (str): path to input txt file
    output (.txt): segmentation results. The output txt will be save in the same dir as the input file.
    """

    import jieba

    sen_list = open(input_path,'r').read().splitlines()
    f_out = open(input_path.replace(".txt", "_jieba.txt"),'w') 

    for line in tqdm(sen_list):
        sen = line.split("#")[-1]
        seg_list = jieba.cut(sen)
        f_out.write(line+"#"+' '.join(seg_list)+'\n') 

def ckip_seg(input_path):
    """
    ckip segmentation
    input (str): path to input txt file
    output (.txt): segmentation results. The output txt will be save in the same dir as the input file.
    """
    from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
    ckip_model_path ='./ckip_data' #path to the downloaded ckip model

    ws = WS(ckip_model_path)
    pos = POS(ckip_model_path)

    sen_list = open(input_path,'r').read().splitlines()
    f_out = open(input_path.replace(".txt", "_ckip.txt"),'w')
    for line in tqdm(sen_list):
        sen = line.split("#")[-1]
        word_sentence_list = ws([sen])
        pos_sentence_list = pos(word_sentence_list)
        f_out.write(line+"#"+' '.join(word_sentence_list[0])+"#"+' '.join(pos_sentence_list[0])+'\n')  


def ddparser_seg(input_path):
    """
    ddparser segmentation
    input (str): path to input txt file
    output (.txt): segmentation results. The output txt will be save in the same dir as the input file.
    """
    import opencc
    from ddparser import DDParser

    ddp = DDParser(use_pos=True)
    t2s = opencc.OpenCC('t2s.json') 
    s2t = opencc.OpenCC('s2t.json')

    sen_list = open(input_path,'r').read().splitlines()
    f_out = open(input_path.replace(".txt", "_ddparser.txt"),'w')

    for line in tqdm(sen_list):
        # Convert the traditional Chinese to simplies Chinese because ddpaser is designed for simplified Chinese
        sen = line.split("#")[-1]
        sen_t = t2s.convert(sen)  
        result = ddp.parse(sen_t)
        word = ' '.join(result[0]['word'])
        word = s2t.convert(word)
        pos = result[0]['postag']
        f_out.write(line+"#"+word+"#"+' '.join(pos)+'\n')  
  

def sensitive_filter(input_path, sensitive_word_path, save_rm=True):
    """
    Remove sentences contain sensitive words
    For Chinese, the sentences need to be segmentated first
    input (str,str,bool): path to sentences list, path to sensitive word list, whether to save the removed results
    output (.txt): results of removing sensitive words
    """
    sensitive_word_list = set(open(sensitive_word_path,"r").read().splitlines())
    sen_list = open(input_path,'r').read().splitlines()

    output_path = input_path.replace(".txt", "_s3.txt")
    if save_rm:
        f_rm = open(input_path.replace(".txt", "_s3_rm.txt"),'w')
    f_out = open(output_path,'w')

    for line in sen_list:
        word_split = line.split("#")[2].split(" ")
        the_sensitive_word = list(sensitive_word_list & set(word_split))
        if len(the_sensitive_word)>=1:
            if save_rm:
                f_rm.write(line+"#"+' '.join(the_sensitive_word)+"\n")
        else:
            f_out.write(line+"\n")


def ckip_filter(input_path, save_rm):
    """
    POS filter using POS tags and segmentation results from ckip
    input (str, bool): path to ckip POS tagging & segmentation results, whether to save the removed sentences
    output (.txt): results of POS filter
    """

    wordlist_first = ['二','三','四','五','六','七','八','九','十',
                        '也','於','為','還','都','連','並','而','但','再']
    wordlist_firsttwo = ['而且','於是','就是','無論','因為','其中','為了','儘管','所以',
                         '甚至','還是','在於','如果','因此','可能','例如']
    wordlist_middle = ['吧','嗎','呢','啊','啥']
    
    ckip_include = set(['Nb','Nc','FW'])
    ckip_end = ['Caa','Cab','Cba','Cbb','P','T']
    ckip_start = ['DE','SHI','T']

    output_path = input_path.replace(".txt", "_s4.txt")
    f_out = open(output_path,'w')
    if save_rm:
        f_rm = open(input_path.replace(".txt", "_s4_rm.txt"),'w')

    sen_list = open(input_path,'r').read().splitlines()

    for line in sen_list:
        rm_flag = False
        word_split = line.split("#")[2].split(" ")
        sen = line.split("#")[1]

        ## check the number of maximun word characters
        for word in word_split:
            if len(word)>=5:
                rm_flag = True

        ## check whether the sentence contain duplicate words
        if len(set(word_split))<len(word_split):
            rm_flag = True
        
        first_word = sen[0]
        if first_word in wordlist_first:
            rm_flag = True

        firsttwo = sen[:2]
        if firsttwo in wordlist_firsttwo:
            rm_flag = True

        for middle in wordlist_middle:
            if middle in sen[:-1]:
                rm_flag = True

        tag = line.split("#")[3].split(" ")
        if len(ckip_include & set(tag) )>=1:
            rm_flag = True
        if tag[-1] in ckip_end:
            rm_flag = True
        if tag[0] in ckip_start:
            rm_flag = True

        if not rm_flag:
            f_out.write(line+'\n')
        else:
            if save_rm:
                f_rm.write(line+'\n')

    return output_path

def ddparser_filter(input_path, save_rm):
    """
    POS filter using POS tags and segmentation results from ddparser
    input (str, bool): path to ddparser POS tagging & segmentation results, whether to save the removed sentences
    output (.txt): results of POS filter
    """

    wordlist_first = ['二','三','四','五','六','七','八','九','十',
                        '也','於','為','還','都','連','並','而','但','再']
    wordlist_firsttwo = ['而且','於是','就是','無論','因為','其中','為了','儘管','所以',
                         '甚至','還是','在於','如果','因此','可能','例如']
    wordlist_middle = ['吧','嗎','呢','啊','啥']

    ddparser_include = set(['LOC','ORG','TIME','PER','w','nz'])
    ddparser_end = ['p','u','c']
    ddparser_start = ['xc','u']

    output_path = input_path.replace(".txt", "_s4.txt")
    f_out = open(output_path,'w')
    if save_rm:
        f_rm = open(input_path.replace(".txt", "_s4_rm.txt"),'w')
    sen_list = open(input_path,'r').read().splitlines()

    for line in sen_list:
        rm_flag = False
        word_split = line.split("#")[2].split(" ")
        sen = line.split("#")[1]

        ## check the number of maximun word characters
        for word in word_split:
            if len(word)>=5:
                rm_flag = True

        ## check whether the sentence contain duplicate words
        if len(set(word_split))<len(word_split):
            rm_flag = True
        
        first_word = sen[0]
        if first_word in wordlist_first:
            rm_flag = True

        firsttwo = sen[:2]
        if firsttwo in wordlist_firsttwo:
            rm_flag = True

        for middle in wordlist_middle:
            if middle in sen[:-1]:
                rm_flag = True

        tag = line.split("#")[3].split(" ")  
        if len(ddparser_include & set(tag) )>=1:
            rm_flag = True
        if tag[-1] in ddparser_end:
            rm_flag = True
        if tag[0] in ddparser_start:
            rm_flag = True

        if not rm_flag:
            f_out.write(line+'\n')
        else:
            if save_rm:
                f_rm.write(line+'\n')

    return output_path



def pos_seg_filter(input_path_ckip=None, input_path_ddparser=None, save_rm=True):
    """
    Call ckip_filter or ddparser_filter or both
    input (str,str,bool): path to ckip results, path to ddparser results, whether to save the removed sentences
    output (.txt): results of POS and segmentation filter
    """
    if input_path_ckip and not(input_path_ddparser):
        ckip_filter(input_path_ckip, save_rm)
    elif not input_path_ckip and (input_path_ddparser):
        ddparser_filter(input_path_ddparser, save_rm)
    else:
        ckip_output = ckip_filter(input_path_ckip, save_rm)
        ddpaser_output = ddparser_filter(input_path_ddparser, save_rm)
        f_ckip = open(ckip_output,'r').read().splitlines()
        f_ddp = open(ddpaser_output,'r').read().splitlines()
        ddp_idx = []
        for line in f_ddp:
            idx = line.split("#")[0]
            ddp_idx.append(idx)

        f_out = open(input_path_ckip.replace(".txt", "_ckipddp_s4.txt"),'w')
        if save_rm:
            f_rm = open(input_path_ckip.replace(".txt", "_ckipddp_s4_rm.txt"),'w')
        
        for line in f_ckip:
            idx = line.split("#")[0]
            if idx in ddp_idx:
                f_out.write(line+'\n')
            else:
                f_rm.write(line+'\n')
        

def get_perplexity(input_path, DEVICE='cpu'):
    """
    Get perplexity scores of sentences.
    Change DEVICE to 'cuda:0' if you use gpu
    input (str): path to input sentences list
    output (.txt): results of corresponding perplexity scores
    """
    import perplexity
    perplexity.calculate_perplexity(input_path, DEVICE)


def perplexity_filter(input_path, save_rm=True, th=4.0):
    """
    Filter out sentences with perplexity higher than the threshold
    input (str, bool, float): path to input sentences, whether to save the removed results, threshold
    output (.txt): results of perplexity filtering
    """
    sen_list = open(input_path,'r').read().splitlines()
    f_out = open(input_path.replace(".txt", "_s5.txt"),'w')
    if save_rm:
        f_rm = open(input_path.replace(".txt", "_s5_rm.txt"),'w')

    for line in sen_list:
        per_score = float(line.split("#")[-1])
        if per_score>=th:
            if save_rm:
                f_rm.write(line+"\n")
        else:
            f_out.write(line+"\n")

def calculate_asr_and_intell(input_path, wav_dir_path):
    """
    Calculate the ASR of the utterances and the intelligibility score
    input (str, str): path to input sentences,  path to utterance directory
    output (.txt): input sentences with ASR prediction and intelligibility scores
    """
    import speech_recognition as sr
    from scipy.io import wavfile
    from Levenshtein import distance, ratio

    r = sr.Recognizer()

    output_name = input_path.replace(".txt", "_asr.txt")
    if os.path.exists(output_name):
        f_out = open(output_name,"r").read().splitlines()
        exist_index = [line.split("#")[1] for line in f_out]
        f_out = open(output_name,"a")

    else:
        f_out = open(output_name,"w")
        exist_index = []

    sen_list = open(input_path,"r").read().splitlines()

    for line in tqdm(sen_list):
        index = line.split("#")[0]
        sen = line.split("#")[1]
        if sen in exist_index:
            continue
        else:
            wav_path = os.path.join(wav_dir_path,index+'.wav')
            try:
                rate, data = wavfile.read(wav_path)
                y = (np.iinfo(np.int32).max * (data/np.abs(data).max())).astype(np.int32)
                output_tmp_file = 'asr_tmp' #create tmp file
                wavfile.write(output_tmp_file, rate, y)   
                with sr.AudioFile(output_tmp_file) as source:
                    audio = r.record(source)
                    asr_result = r.recognize_google(audio, language='zh-TW')
                intell_score = ratio(sen, asr_result)
                f_out.write(line+"#"+asr_result+"#"+str(intell_score)+"\n")
            except Exception as e:
                print(e)

def intelligibility_filter(input_path, save_rm=True, th=1.):
    """
    Filter out sentences with intelligiblity lower than the threshold
    input (str, bool, float): path to input sentences, whether to save the removed results, threshold
    output (.txt): results of intelligibility filtering
    """
    sen_list = open(input_path,'r').read().splitlines()
    f_out = open("candidate_sentences.txt",'w')
    if save_rm:
        f_rm = open(input_path.replace(".txt", "_s6_rm.txt"),'w')

    for line in sen_list:
        intell_score = float(line.split("#")[-1])
        if intell_score<th:
            if save_rm:
                f_rm.write(line+"\n")
        else:
            f_out.write(line+"\n")

def calculate_statistics(input_path):
    """
    Calculate the ground-truth syllable distribution. 
    input (str): path to the crawled articles
    output (dict):(1)gt_syllable.pickle
                  (2)gt_syllable_with_tone.pickle
                  (3)gt_initial.pickle
                  (4)gt_final.pickle
    """
    from pypinyin import pinyin, lazy_pinyin, Style
    from collections import Counter

    f = open(input_path,'r').read().splitlines()
    syllable = []
    wtone_syllable = []
    initial = []
    final = []

    for sen in tqdm(f):      
        if re.search('[a-zA-Z]', sen) or (not sen.isalpha()):
            pass
        else:
            syllable.extend(lazy_pinyin(sen))
            the_syllable = pinyin(sen, style=Style.TONE3, heteronym=False)
            wtone_syllable.extend([x[-1] for x in the_syllable])
            the_initial = pinyin(sen, style=Style.INITIALS, heteronym=False)
            the_final = pinyin(sen, style=Style.FINALS, heteronym=False)
            initial.extend([x[-1] for x in the_initial])
            final.extend([x[-1] for x in the_final])

    wtone_syllable = dict(Counter(wtone_syllable))
    print("syllables with tone:", wtone_syllable)
    print('nums of syllable with tone',len(wtone_syllable))

    syllable = dict(Counter(syllable))
    print("base syllable:", syllable)
    print('nums of base syllable (without consider the tone): ',len(syllable))

    initial = dict(Counter(initial))
    print("initial:", initial)
    print('nums of initial',len(initial))

    final = dict(Counter(final))
    print("final:", final)
    print('nums of final',len(final))

    with open('gt_syllable_with_tone.pickle', 'wb') as handle:
        pickle.dump(wtone_syllable, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('gt_syllable.pickle', 'wb') as handle:
        pickle.dump(syllable, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('gt_final.pickle', 'wb') as handle:
        pickle.dump(final, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('gt_initial.pickle', 'wb') as handle:
        pickle.dump(initial, handle, protocol=pickle.HIGHEST_PROTOCOL)


def map_syllable(syllable_sen, syllable_key):
    syllable_vector = np.zeros((len(syllable_key), ))
    for key in syllable_sen:
        syllable_vector[syllable_key[key]]+=1
    return syllable_vector

def prepare_data_for_sampling(gt_syllable_path, sen_list_path, wtone=True):
    """
    input (dict, str, bool): path to ground truth syllable dict, path to sentences list, whether the gt_syllable is syllable with tone
    output:
            (1) gt_syllable_distribution.npy #real-world syllable distrubution. dimension: (numbers_of_syllables, 1)  
            (2) gt_syllables_key.pickle    # record the mapping of syllables  
            (3) idx_syllables.npy    # record the mapping of sentences and syllables  
            (4) idx_content.npy  # record the content   
            (5) idx_oriidx.npy      # record the mapping of original index and new index  
            (1), (3), (4) are inputs for sampling  
    """
    from pypinyin import pinyin, lazy_pinyin, Style

    with open(gt_syllable_path, 'rb') as fp:
        gt_syllable = pickle.load(fp)

    print("numbers of syllable:",len(gt_syllable))
    #print(gt_syllable)
    syllable_key = {}
    for idx, key in enumerate(list(gt_syllable.keys())):
        syllable_key[key] = idx

    truth_syllable = np.zeros((len(syllable_key), ))
    for key, value in gt_syllable.items():
        truth_syllable[syllable_key[key]]=value

    f = open(sen_list_path,'r').read().splitlines()
    idx_syllable = []
    idx_content = []
    idx_oriidx = []
    for line in tqdm(f):
        sen =  line.split("#")[1]
        oriidx = line.split("#")[0]    
        try:
            if wtone:
                the_syllable = pinyin(sen, style=Style.TONE3, heteronym=False)
                the_syllable = [x[-1] for x in the_syllable]
            else:
                the_syllable = lazy_pinyin(sen)
            syllable_vector = map_syllable(the_syllable,syllable_key)
            idx_syllable.append(syllable_vector)
            idx_content.append(sen)
            idx_oriidx.append(oriidx)
        except Exception as e:
            print(e)

    idx_content = np.asarray(idx_content)
    idx_syllable = np.asarray(idx_syllable)
    idx_oriidx = np.asarray(idx_oriidx)

    with open('gt_syllable_key.pickle', 'wb') as handle:
        pickle.dump(syllable_key, handle, protocol=pickle.HIGHEST_PROTOCOL)

    np.save('gt_syllable_distribution', truth_syllable)
    np.save('idx_content', idx_content)
    np.save('idx_syllables',idx_syllable)
    np.save('idx_oriidx',idx_oriidx)