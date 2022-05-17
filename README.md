# Speech Corpus

## Data collection

## Preprocessing

### Word segmentation & Get POS tags 

#### Segmentation:
* [jieba](https://github.com/fxsjy/jieba)
* 
#### Segmentation & POS tagging:
* [ddparser](https://github.com/baidu/DDParser)
* [ckiptagger](https://github.com/ckiplab/ckiptagger)

#### Conversions between Traditional Chinese, Simplified Chinese
* [opencc](https://github.com/BYVoid/OpenCC) (Required if using ddparser)

- Remove sentences contain words longer than 5 characters 
  * Most Chinese words are less than five characters 
- Select sentences candidate based on POS tagging
  *
  
### Remove sentences contain sensitive words
### Select candidate sentences based on the predictions of ASR systems


## Sampling

