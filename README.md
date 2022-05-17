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
* The processed file will be saved as {file_name}_s1.txt
* Automatically add indexes to all sentences

*input format:sentence*
*output format:Idx#sentence*

### Step 2 POS Filtering

* Remove sentences contain words longer than 5 characters 
( Most Chinese words are less than five characters) 
* Select sentences candidate based on POS tagging


#### Word segmentation & Get POS tags 

Segmentation:
* [jieba](https://github.com/fxsjy/jieba)
```
preprocessing.jieba_seg("/path/to/raw_data_s1.txt")
```
output files will be saved as {input_file_name}_jieba.txt
format: Idx#sentence#word segmentation

Segmentation & POS tagging:
* [ddparser](https://github.com/baidu/DDParser)
* [opencc](https://github.com/BYVoid/OpenCC) (Conversions between Traditional Chinese, Simplified Chinese)

```
preprocessing.baidu_seg("/path/to/raw_data_s1.txt")
```
output files will be saved as {input_file_name}_baidu.txt
format: Idx#sentence#word segmentation#pos tags

* [ckiptagger](https://github.com/ckiplab/ckiptagger)

```
preprocessing.ckip_seg("/path/to/raw_data_s1.txt")
```
output files will be saved as {input_file_name}_ckip.txt
format: Idx#sentence#word segmentation#pos tags


#### Filtering
```
preprocessing.pos_seg_filter("/path/to/raw_data_s1_jieba.txt")
#only use segmentation for filtering

preprocessing.pos_seg_filter("/path/to/raw_data_s1_ckip.txt", pos="ckip")
preprocessing.pos_seg_filter("/path/to/raw_data_s1_baidu.txt", pos="baidu")
#use both segmentation and POS tagging for filtering

```
output files will be saved as {input_file_name}_s2.txt
format: Idx#sentence#word segmentation#pos tags


#### Intersect the result (optional)
Consider the results of jieda & ckip & baidu
```
preprocessing.pos_seg_filter("/path/to/raw_data_s1_baidu.txt", pos="all")

```


### Step 3 Sensitive Word Filtering
* Remove sentences contain sensitive words
  (See sensitive_word_list.txt for details)
  
```
preprocessing.sensitive_filter(input_file_path)

e.g.
preprocessing.sensitive_filter("/path/to/raw_data_seg.txt")
```

### Step 4 ASR Filtering

Select candidate sentences based on the predictions of ASR systems


## Sampling

