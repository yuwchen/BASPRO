# BASPRO: A balanced script producer for speech corpus based on the genetic algorithm

<img src="https://github.com/yuwchen/BASPRO/blob/main/images/BASPRO.png" alt="main"  width=50% height=50% />


***
## Data collection

See raw_data.txt for an example of a original file.

raw_data.txt example:
```
第一句話
換行後是下一句
每句的長度有可能不同
也有可能包含非中文字
```

## Data Processing

### Step 1 General Filtering
* Leave only sentences with ten characters in Chinese
* Automatically add indexes to all sentences

```
import preprocessing
preprocessing.general_filter(input_file_path)

e.g.
preprocessing.general_filter("/path/to/raw_data.txt") 

```
* The processed file will be saved as result_s1.txt

*input format: sentence*  
*output format: Idx#sentence*

***

### Step 2 Segmentation and POS tagging

Segmentation:

* [jieba](https://github.com/fxsjy/jieba)  
```
preprocessing.jieba_seg("/path/to/result_s1.txt")
```
. output files will be saved as {input_file_name}_jieba.txt  
*input format: Idx#sentence*   
*output format: Idx#sentence#word segmentation*  

Segmentation & POS tagging:

(1) [ckiptagger](https://github.com/ckiplab/ckiptagger)  

```
preprocessing.ckip_seg("/path/to/result_s1.txt")
``` 
. output files will be saved as {input_file_name}_ckip.txt  
*input format: Idx#sentence  
*output format: Idx#sentence#word segmentation#pos tags*  

(2)[ddparser](https://github.com/baidu/DDParser)

Additional requirement (Conversions between Traditional Chinese, Simplified Chinese): [opencc](https://github.com/BYVoid/OpenCC)  
Install:
```
pip install LAC
pip install ddparser
pip install opencc
```
Run segmentation:
```
preprocessing.ddparser_seg("/path/to/result_s1.txt")
```
. output files will be saved as {input_file_name}_ddparser.txt  
*input format: Idx#sentence  
*output format: Idx#sentence#word segmentation#pos tags*  

(Note: the ddparser cannot run on Mac M1)  


### Step 3 Sensitive Word Filtering

* Remove sentences contain sensitive words (See sensitive_word_list.txt for details)
  
```
preprocessing.sensitive_filter("/path/to/inputfile","/path/to/sensitive/word/list", save_rm=True)
e.g.
preprocessing.sensitive_filter("./result_s1_ckip.txt","./sensitive_word_list.txt", save_rm=True)
# output file will be saved as result_s1_ckip_s3.txt

```
*input & output format: Idx#sentence#word segmentation{#pos tags}*  

### Step 4 POS Filtering

(1) Remove sentences contain words longer than 5 characters (Most Chinese words are less than five characters)  
(2) Remove sentences contain duplicate words  
(3) Remove sentences end or contain certain words  

Removal criteria 
|                      |                                                                                                                 |
|----------------------|-----------------------------------------------------------------------------------------------------------------|
| first character      | '二','三','四','五','六','七','八','九','十','也','於','為','還','都','連','並','而','但','再'                        |
| first two characters | '而且','於是','就是','無論','因為','其中','為了','儘管','所以','甚至','還是','在於','如果','因此','可能','例如'            |
| in the middle        | '吧','嗎','呢','啊','啥'                                                                                          |

(4) Select sentences candidate based on POS tags

POS removal criteria
| toolkit | include                           | start          | end                             |
|---------|-----------------------------------|----------------|---------------------------------|
| ckip    | 'Nb','Nc','FW'                    | 'DE','SHI','T' | 'Caa','Cab','Cba','Cbb','P','T' |
| ddparser   | 'LOC','ORG','TIME','PER','w','nz' | 'p','u','c'    | 'xc','u'                        |

ckip POS tag: https://github.com/ckiplab/ckiptagger/wiki/POS-Tags  
ddparser POS tag: https://github.com/baidu/lac    


```
# use ckip results only
preprocessing.pos_seg_filter(input_path_ckip="./result_s1_ckip_s3.txt", save_rm=True)
# or use ddparser results only
preprocessing.pos_seg_filter(input_path_ddparser="./result_s1_ddparser_s3.txt", save_rm=True)
# or use both ckip and ddparser results
preprocessing.pos_seg_filter(input_path_ckip="./result_s1_ckip_s3.txt", input_path_ddparser="./result_s1_ddparser.txt", save_rm=True)

```
. output files will be saved as {input_file_name}_{pos tag}_s4.txt  
. if save_rm=True, the removed sentences will be recorded in {input_file_name}_s4_rm.txt  
. edit **ckip_filter** or **ddparser_filter** function in **preprocessing.py** if you want to use other criteria  
*input & output format: Idx#sentence#word segmentation#pos tags*  


***
### Step 5 Perplexity Filtering

* Step 5-0: Install pytorch transformer
```
pip install pytorch-transformers
```

* Step 5-1: Calculate perplexity scores
```
preprocessing.get_perplexity("/path/to/inputfile")
e.g.
preprocessing.get_perplexity("./result_s1_ckip_s3_ckipddp_s4.txt") # if you use cpu
#or
preprocessing.get_perplexity("./result_s1_ckip_s3_ckipddp_s4.txt", 'cuda:0') #if you use gpu

```
Note: recommended to use gpu  
. output files will be saved as {file_name}_per.txt   
*input format: Idx#sentence{#word segmentation#pos tags}*  
*output format: Idx#sentence{#word segmentation#pos tags}#perplexity_score*  

* Step 5-2: Remove sentences have high perplexity
```
preprocessing.perplexity_filter(input_file_path, save_rm=True, th=4.0)
```
. th: threshold of perplexity filtering, default value is 4.0  
. output files will be saved as {input_file_name}_s5.txt  
*input format: Idx#sentence{#word segmentation#pos tags}#perplexity_score*   
*output format: Idx#sentence{#word segmentation#pos tags}#perplexity_score*  

***
### Step 6 Intelligibility Filtering

Select candidate sentences based on the intelligibility scores. The following figure shows how the intelligibility score is calculated.

<img src="https://github.com/yuwchen/BASPRO/blob/main/images/intell_filter.png" alt="main"  width=40% height=40% />


#### Step 6-1: Text to Speech

| Toolkit                                                      | Quality    | Processing time for 10 samples| Support Taiwanese Accent | Speaker Gender |
|--------------------------------------------------------------|------------|-------------------------------|--------------------------|----------------|
| [Gtts](https://github.com/pndurette/gTTS)                    | Excellent  | ~20s                          | V                        |Female          |
| [Paddle Speech](https://github.com/PaddlePaddle/PaddleSpeech)| Good       | ~92s                          | X                        |Male & Female   | 
| [ttskit](https://github.com/kuangdd/ttskit)                  | OK         | ~22s                          | X                        |Male & Female   |
| [zhtts](https://github.com/Jackiexiao/zhtts)                 | OK?        | ~15s                          | X                        |Female          |


```
text2speech.tts_gtts(input_file_path, save_info=True, convert_format=True) #Set convert_format=False if you want to use the original format
text2speech.tts_paddle(input_file_path, save_info=True)
text2speech.tts_ttskit(input_file_path, save_info=True)
text2speech.tts_zhtts(input_file_path, save_info=True)
```
PaddleSpeech & TTSkit & Zhtts: cannot run on  Mac M1

Gtts:  
. The original Gtts output format might have some problem when loading with python.  
. Load the Gtts output waveform using **librosa.load(/wav/file/path)**
. Install ffmpeg if encounter **audioread.exceptions.NoBackendError** when using librosa  
```
conda install -c conda-forge ffmpeg
```

. save_info=True will save the mapping between wavefile index and content in ttx_info_{toolkit}.txt  
. output waveform will be save in {file_name}_{toolkit} directory  
. this step might take a long time to fininsh  

*input format: Idx#sentence{#word segmentation#pos tags#perplexity_score}*  


#### Step 6-2: Calculate the intelligibility scores based on ASR results 
```
preprocessing.calculate_asr_and_intell(input_file_path, wav_dir_path)

```
. this step will take a long time to fininsh. 
. the index in the input_file_path should match the file name in the wave file directory  
e.g.  
```
* input_file.txt
Idx1#第一句話的範例有十字#...#...#...
Idx2#有沒有包含斷詞不影響#...#...#...

* wav_dir
├── Idx1.wav (TTS results of the sentence "第一句話的範例有十字")
├── Idx2.wav (TTS results of the sentence "有沒有包含斷詞不影響")
│   ...
└── IdxN.wav
```
. output file will be save as {input_file_name}_asr.txt

#### Step 6-3: Select sentences base based on the intelligibility scores 
```
preprocessing.intelligibility_filter(input_file_path, save_rm=True, th=1.0)
```
th: threhold of intelligibility filtering, default value is 1.0
output files will be saved as candidate_sentences.txt
*input format: Idx#sentence{#word segmentation#pos tags#perplexity_score#asr_prediction_result}#intelligibility_score*   
*output format: Idx#sentence{#word segmentation#pos tags#perplexity_score#asr_prediction_result}#intelligibility_score*  


## Script composing

### Step 1: Calculate statistics for text corpus

#### Step 1-0: install [pypinyin](https://pypi.org/project/pypinyin/)
```
pip install pypinyin
```

#### Step 1-1: calculate the statistics
```
preprocessing.calculate_statistics("/path/to/raw_data.txt")
```
input format: sentence
output: dict
(1)gt_syllable.pickle  
(2)gt_syllable_with_tone.pickle  
(3)gt_initial.pickle  
(4)gt_final.pickle  

### Step 2: Prepare the data for sampling 

```
preprocessing.prepare_data("gt_syllable_with_tone.pickle", "result_s5.txt")  
 
```
input:  
(1) gt_syllable.pickle or gt_syllable_with_tone.pickle  
(2) input_sen_list.txt with format: Idx#sentence#word segmentation{#...}  

output:  
(1) gt_syllable_distribution.npy #real-world syllable distrubution. dimension: (numbers_of_syllables, 1)  
(2) gt_syllables_key.pickle    # record the mapping of syllables  
(3) idx_syllables.npy    # record the mapping of sentences and syllables  
(4) idx_content.npy  # record the content   
(5) idx_oriidx.npy      # record the mapping of original index and new index  

(1), (3), (4) are inputs for sampling  

. Example:  

If a language only contain three words, AA, BB, CC, and their corresponding phonemes are a, b, c. 
Asssuming the crawled articles are:   
```
AA AA BB CC CC CC CC 
CC AA BB AA AA CC CC CC CC CC
```
Then, the phonemes of the articles are 
```
a a b c c c c
c a b a a c c c c c
```
The statistics of the real-world (ground-truth) condition: 
```
gt_syllables_key = {"a":5,"b":2,"c":10} 
gt_syllables_key.keys() = ["a","b","c"]  
gt_syllables_distribution = [5, 2, 10]  
```

Assuming the candidate sentences are:  
```
idx_3:AA BB BB
idx_5:BB CC CC  
```
Then, 
```
idx_syllables = [[1,2,0],  #syllable distribution of the 1st sentence in the candidate sentences file
                 [0,1,2]]  #syllable distribution of the 2nd sentence in the candidate sentences file

idx_content = [AA BB BB,  
               BB CC CC]  
```

## GA-based Sampling

### Adjust the hyperparameters

```
#in sampling.py file

num_of_set = 20  #numbers of the set in the corpus
num_of_sen_in_set = 20 #numbers of the sentences in a set
population_size = 10000 #initial population size of the GA
iteration = 500 # numbers of interation for GA

truth_syllable = np.load('gt_syllable_distribution.npy') #load the results of Data preparation Step2
idx_syllable = np.load("idx_syllable.npy"). #load the results of Data preparation Step2


```
### Generate a new corpus

```
python sampling.py --outputdir output
```
output:
(1) best_chro.npy  # the best chromosome (the sampled sentences index list). 
(2) corpus.txt     # the content of best_chro. 
(3) f_max.npy      # the maximun fitness during the training. 
(4) f_mean.npy     # the mean fitenss during the training.    
(5) final_chro.npy # best chromosome in the end of sampleing, usually the same as best_chro.npy. 



## Post-processing (Optional)

Post-processing step provide GA-based and Greedy-based method to replace some sentences in the corpus. 
Greedy-based method is better for the condition that only a few sentences are required to be replace. 
On the other hand, GA-based method is more suitable for the condiation that many sentences in the corpus need the replacement.

### Step 1: record the unwanted sentences 

Write the "index_in_sentence_candidates" in the excluded_idx.txt files.

For example, 
after reading the sentences in corpus.txt, you want to replace "一起搭多元計程車回家" with another sentences. 
Create a excluded_idx.txt file and write ${index_in_sentence_candidates} in the excluded_idx.txt.

```
# corpus.txt

set_idx:sentence_idx:index_in_sentence_candidates:content:
0:0:4993:一起搭多元計程車回家
0:1:4290:比其他種類的草莓還甜

# excluded_idx.txt
4993
```

### Step 2: GA-based replacement

Run the sampling.py again. The sentence in excluded_idx.txt will be replaced by other sentences.

```
python sampling.py --initial_dir output --excluded excluded_idx.txt
```

### Step 2: Greedy-based replacement

The sentence in excluded_idx.txt will be replaced by other sentences.

```
python greedy.py --initial_dir output --excluded excluded_idx.txt
```
