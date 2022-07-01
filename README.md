# Speech Corpus

## Data collection

See raw_data.txt for an example of a original file
raw_data.txt example:
```
第一句話
換行後是下一句
每句的長度有可能不同
也有可能包含非中文字
```

## Preprocessing

### Step 1 General Filtering
* Leave only sentences with ten words in Chinese
```
import preprocessing
preprocessing.general_filter(input_file_path)

e.g.
preprocessing.general_filter("/path/to/raw_data.txt") 

```
* The processed file will be saved as result_s1.txt

*input format: sentence*  
*output format: sentence*

***
### Step 2 POS Filtering

#### Word segmentation & Get POS tags 

* Automatically add indexes to all sentences

Segmentation:

* [jieba](https://github.com/fxsjy/jieba)
```
preprocessing.jieba_seg("/path/to/result_s1.txt")
```
. output files will be saved as {input_file_name}_jieba.txt
*input format: sentence*
*output format: Idx#sentence#word segmentation*

Segmentation & POS tagging:

(1) [ckiptagger](https://github.com/ckiplab/ckiptagger)

```
preprocessing.ckip_seg("/path/to/result_s1.txt")
```
. output files will be saved as {input_file_name}_ckip.txt
*input format: sentence
*output format: Idx#sentence#word segmentation#pos tags*

(2)[ddparser](https://github.com/baidu/DDParser)

```
preprocessing.baidu_seg("/path/to/result_s1.txt")
```
. output files will be saved as {input_file_name}_baidu.txt  
*input format: sentence
*output format: Idx#sentence#word segmentation#pos tags*  

Additional requirement (Conversions between Traditional Chinese, Simplified Chinese):
* [opencc](https://github.com/BYVoid/OpenCC) 

(Require further testing, cannot run ddparser on Mac M1)


### Filtering

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
| baidu   | 'LOC','ORG','TIME','PER','w','nz' | 'p','u','c'    | 'xc','u'                        |

ckip POS tag: https://github.com/ckiplab/ckiptagger/wiki/POS-Tags  
baidu POS tag: https://github.com/baidu/lac    


```
preprocessing.pos_seg_filter("/path/to/result_s1_jieba.txt", save_rm=True)
#only use segmentation for filtering

preprocessing.pos_seg_filter("/path/to/result_s1_ckip.txt", pos="ckip", save_rm=True)
preprocessing.pos_seg_filter("/path/to/result_s1_baidu.txt", pos="baidu", save_rm=True)
#use both segmentation and POS tagging for filtering

```
. output files will be saved as {input_file_name}_{pos}_s2.txt  
. if save_rm=True, the removed sentences will be recorded in {input_file_name}_s2_rm.txt  
. edit **pos_seg_filter** function in **preprocessing.py** if you want to use other criteria  
*input & output format: Idx#sentence#word segmentation#pos tags*  


***
### Step 3 Sensitive Word Filtering
* Remove sentences contain sensitive words (See sensitive_word_list.txt for details)
  
```
preprocessing.sensitive_filter("/path/to/inputfile", save_rm=True)
e.g.
preprocessing.sensitive_filter("./result_s1_ckip_s2.txt", save_rm=True)
```

. output file will be saved as result_s3.txt
*input & output format: Idx#sentence#word segmentation{#pos tags}*  

***
### Step 4 Perplexity Filtering

* Calculate perplexity scores
```
preprocessing.get_perplexity("/path/to/inputfile")
e.g.
preprocessing.get_perplexity("./result_s3.txt")
```

. output files will be saved as {file_name}_per.txt  
*input format: Idx#sentence{#word segmentation#pos tags}*  
*output format: Idx#sentence{#word segmentation#pos tags}#perplexity_score*  

* Remove sentences have high perplexity
```
preprocessing.perplexity_filter(input_file_path, save_rm=True, th=4.0)
```
. th: threhold of perplexity filtering, default value is 4.0  
. output files will be saved as result_s4.txt  
*input format: Idx#sentence{#word segmentation#pos tags}#perplexity_score*   
*output format: Idx#sentence{#word segmentation#pos tags}#perplexity_score*  

***
### Step 5 ASR Filtering

* Select candidate sentences based on the predictions of ASR systems
#### Step 5-1: Text to Speech

| Toolkit                                                      | Quality | Speed                         | Support Taiwanese Accent | Speaker Gender |
|--------------------------------------------------------------|---------|-------------------------------|--------------------------|----------------|
| [Gtts](https://github.com/pndurette/gTTS)                    | High    | Fast<br>(but has limited access, ~2s/sample) | V         |Female        |
| [Paddle Speech](https://github.com/PaddlePaddle/PaddleSpeech)|         |                               | X                        |Male & Female   | 
| [ttskit](https://github.com/kuangdd/ttskit)                  |         |                               |                          |Male & Female   |
| [zhtts](https://github.com/Jackiexiao/zhtts)                 | Low     |                               | V                        |Female          |


```
text2speech.tts_gtts(input_file_path, save_info=True, convert_format=True)
text2speech.tts_paddle(input_file_path, save_info=True)
text2speech.tts_ttskit(input_file_path, save_info=True)
text2speech.tts_zhtts(input_file_path, save_info=True)
```
PaddleSpeech & TTSkit & Zhtts:
(Require further testing, cannot run on  Mac M1)

Gtts:
. The original Gtts output format might have some problem when loading with python. 
. Load the Gtts output waveform using librosa.load(/wav/file/path)
. Install ffmpeg if encounter audioread.exceptions.NoBackendError when using librosa
. Set convert_format=True to convert the gtts output wavform
```
conda install -c conda-forge ffmpeg
```

. save_info=True will save the mapping between wavefile index and content in ttx_info_{toolkit}.txt  
. output waveform will be save in {file_name}_{toolkit} directory  
. this step might take a long time to fininsh  
*input format: Idx#sentence{#word segmentation#pos tags#perplexity_score}*  


#### Step 5-2: Calculate the intelligibility scores based on ASR results 
```
preprocessing.calculate_asr(input_file_path, wav_dir_path)

```
. this step might take a long time to fininsh  
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
. output file will be save as result_asr.txt

#### Step 5-3: Select sentences base based on the intelligibility scores 
```
preprocessing.intelligibility_filter(input_file_path, save_rm=True, th=1.0)
```
th: threhold of intelligibility filtering, default value is 1.0
output files will be saved as result_s5.txt
*input format: Idx#sentence{#word segmentation#pos tags#perplexity_score]#intelligibility_score*   
*output format: Idx#sentence{#word segmentation#pos tags#perplexity_score}#intelligibility_score*  

## Data preparation

### Step 1: Calculate statistics for text corpus
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
(1) gt_syllabus_distribution.npy 
. real-wold syllabus distrubution. dimension: (numbers_of_syllabus, 1)
. example:
if the syllabus of the text corpus are "ABCCBC", then
syllabus_key = {"A":1,"B":2,"C":3}, syllabus_key.keys() = ["A","B","C"]
gt_syllabus_distribution = [1, 2, 3] 

(2) gt_syllabus_key.pickle    # record the mapping of syllabus
(3) idx_syllabus.npy    # record the mapping of sentences and syllabus
. example:
input corpus.txt:
idx_3:AAAB    # 3A1B
idx_5:BBC     # 2B1C
sen_syllabus = [[3,1,0],
                [0,2,1]]
(4) idx_content.npy  # record the content
(5) idx_oriidx.npy      # record the mapping of original index and new index


(1) and (3) are inputs for sampling

## Sampling

### Adjust the hyperparameters

```
#in sampling.py file

num_of_set = 20  #numbers of the set in the corpus
num_of_sen_in_set = 20 #numbers of the sentences in a set
population_size = 10000 #initial population size of the GA
iteration = 500 # numbers of interation for GA

truth_syllable = np.load('gt_syllabus_distribution.npy') #load the results of Data preparation Step2
idx_syllable = np.load("idx_syllabus.npy"). #load the results of Data preparation Step2


```
### sampling from scratch

```
python sampling.py --outputdir output
```
output:
(1) best_chro.npy  # the best chromosome (the sampled sentences index list). 
(2) corpus.txt     # the content of best_chro. 
(3) f_max.npy      # the maximun fitness during the training. 
(4) f_mean.npy     # the mean fitenss during the training.    
(5) final_chro.npy # best chromosome in the end of sampleing, usually the same as best_chro.npy. 

### sampling from previous result (GA)

If you want to replace some sentences in the corpus. Record the "index_in_sentence_candidates" in excluded_idx.txt files.

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

Run the sampling.py again. The sentence in excluded_idx.txt will be replaced by other sentences.

```
python sampling.py --initial_dir output --excluded excluded_idx.txt
```

### sampling from previous result (Greedy)

If you only want to replace a few sentences in the corpus, using greedy algorithm is more effectively.

```
python greedy.py --initial_dir output --excluded excluded_idx.txt
```
