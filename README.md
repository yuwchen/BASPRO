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
* Automatically add indexes to all sentences

*input format: sentence*  
*output format: Idx#sentence*

***
### Step 2 POS Filtering

#### Word segmentation & Get POS tags 

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
*input format: Idx#sentence#word segmentation*
*output format: Idx#sentence#word segmentation#pos tags*

(2)[ddparser](https://github.com/baidu/DDParser)

```
preprocessing.baidu_seg("/path/to/result_s1.txt")
```
. output files will be saved as {input_file_name}_baidu.txt  
*input format: Idx#sentence#word segmentation*  
*output format: Idx#sentence#word segmentation#pos tags*  

Additional requirement (Conversions between Traditional Chinese, Simplified Chinese):
* [opencc](https://github.com/BYVoid/OpenCC) 

(Require further testing, cannot run ddparser on Mac M1)


#### Filtering

* Remove sentences contain words longer than 5 characters 
( Most Chinese words are less than five characters) 
* Remove sentences contain duplicate words 
* Select sentences candidate based on POS tags

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
* if save_rm=True, the removed sentences will be recorded in {input_file_name}_s2_rm.txt  
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
#### Step 4-1: Text to Speech
option(a): [Gtts](https://github.com/pndurette/gTTS)
option(b): [Paddle Speech](https://github.com/PaddlePaddle/PaddleSpeech)
option(c): [ttskit](https://github.com/kuangdd/ttskit)
option(d): [zhtts](https://github.com/Jackiexiao/zhtts)

| Toolkit       | Quality | Speed                         | Support Taiwanese Accent | Speaker Gender |
|---------------|---------|-------------------------------|--------------------------|----------------|
| gtts          | High    | Fast (but has limited access) | V                        |Female          |
| paddle speech |         |                               | X                        |Female          | 
| ttskit        |         |                               |                          |Male & Female   |
| zhtts         |         |                               | V                        |Female          |

```
text2speech.tts_gtts(input_file_path, save_info=True)
text2speech.tts_paddle(input_file_path, save_info=True)
text2speech.tts_ttskit(input_file_path, save_info=True)
text2speech.tts_zhtts(input_file_path, save_info=True)
```

. save_info=True will save the mapping between wavefile index and content in ttx_info_{toolkit}.txt  
. output waveform will be save in {file_name}_{toolkit} directory  
. this step might take a long time to fininsh  
*input format: Idx#sentence{#word segmentation#pos tags#perplexity_score}*  


#### Step 4-2: Calculate the intelligibility scores based on ASR results 
```
preprocessing.calculate_asr(input_file_path, wav_dir_path)

```
. this step might take a long time to fininsh  
. the index in the input_file_path should match the file name in the wave file directory  
e.g.  
```
* input_file_path
Idx1#第一句話的範例有十字#...#...#...
Idx2#有沒有包含斷詞不影響#...#...#...
* wav_dir_path
├── Idx1.wav (TTS results of the sentence "第一句話的範例有十字")
├── Idx2.wav (TTS results of the sentence "有沒有包含斷詞不影響")
│   ...
└── IdxN.wav
```


#### Step 4-3: Select sentences base based on the intelligibility scores 
```
preprocessing.intelligibility_filter(input_file_path, save_rm=True, th=1.0)
```
th: threhold of intelligibility filtering, default value is 1.0
output files will be saved as result_s5.txt
*input format: Idx#sentence{#word segmentation#pos tags#perplexity_score]#intelligibility_score*   
*output format: Idx#sentence{#word segmentation#pos tags#perplexity_score}#intelligibility_score*  

## Sampling



