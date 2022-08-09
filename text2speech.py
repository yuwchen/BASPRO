import os
import time
from tqdm import tqdm

def get_filepaths(directory):
    file_paths = []  
    for root, directories, files in os.walk(directory):
        for filename in files:
                if filename.endswith('.wav'):
                    filename = filename.replace(".wav","")
                    file_paths.append(filename)  
    return file_paths 

def tts_gtts(input_path, save_info=True, convert_format=True):

    from gtts import gTTS

    if convert_format:
        import librosa
        from scipy.io import wavfile

    sen_list = open(input_path).read().splitlines()

    if save_info:
        if os.path.exists('tts_info_gtts.txt'):
            f_out = open('tts_info_gtts.txt','a')
        else:
            f_out = open('tts_info_gtts.txt','w')

    outputdir = os.path.join(input_path.replace(".txt", '_gtts'))
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
        exist_file_list = []
    else:
        exist_file_list = get_filepaths(outputdir)

    for line in tqdm(sen_list):
        index = line.split("#")[0]
        sen =  line.split("#")[1]
        if index in exist_file_list:
            continue
        else:
            outputpath = os.path.join(outputdir,index+'.wav')
            tts = gTTS(sen, lang='zh-TW')
            tts.save(outputpath)
            #convert output waveform to other format
            if convert_format:
                y, sr = librosa.load(outputpath, sr=16000)
                wavfile.write(outputpath, sr, y)
                time.sleep(1.5)  
            else:
                time.sleep(2) 

            if save_info:     
                f_out.write(index+"#"+sen+"\n") 


def tts_paddle(input_path, save_info=True):

    import paddle
    from paddlespeech.cli import TTSExecutor
    tts_executor = TTSExecutor()

    sen_list = open(input_path).read().splitlines()

    if save_info:
        if os.path.exists('tts_info_paddle.txt'):
            f_out = open('tts_info_paddle.txt','a')
        else:
            f_out = open('tts_info_paddle.txt','w')

    outputdir = os.path.join(input_path.replace(".txt", '_paddle'))
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
        exist_file_list = []
    else:
        exist_file_list = get_filepaths(outputdir)

    for line in tqdm(sen_list):
        index = line.split("#")[0]
        sen =  line.split("#")[1]
        if index in exist_file_list:
            continue
        else:
            outputpath = os.path.join(outputdir,index+'.wav')
            wav_file = tts_executor(
                text=sen,
                output=outputpath,
                am='fastspeech2_aishell3',
                am_config=None,
                am_ckpt=None,
                am_stat=None,
                phones_dict=None,
                tones_dict=None,
                speaker_dict=None,
                voc='pwgan_aishell3',
                voc_config=None,
                voc_ckpt=None,
                voc_stat=None,
                lang='zh',
                device=paddle.get_device())
            if save_info:     
                f_out.write(index+"#"+sen+"\n") 


def tts_ttskit(input_path, save_info=True):

    from ttskit import sdk_api

    sen_list = open(input_path).read().splitlines()

    if save_info:
        if os.path.exists('tts_info_ttskit.txt'):
            f_out = open('tts_info_ttskit.txt','a')
        else:
            f_out = open('tts_info_ttskit.txt','w')

    outputdir = os.path.join(input_path.replace(".txt", '_ttskit'))
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
        exist_file_list = []
    else:
        exist_file_list = get_filepaths(outputdir)

    for line in tqdm(sen_list):
        index = line.split("#")[0]
        sen =  line.split("#")[1]
        if index in exist_file_list:
            continue
        else:
            outputpath = os.path.join(outputdir,index+'.wav')
            sdk_api.tts_sdk(sen,output=outputpath)  
            if save_info:     
                f_out.write(index+"#"+sen+"\n") 

def tts_zhtts(input_path, save_info=True):

    import zhtts
    tts = zhtts.TTS(text2mel_name="TACOTRON")

    sen_list = open(input_path).read().splitlines()

    if save_info:
        if os.path.exists('tts_info_zhtts.txt'):
            f_out = open('tts_info_zhtts.txt','a')
        else:
            f_out = open('tts_info_zhtts.txt','w')

    outputdir = os.path.join(input_path.replace(".txt", '_zhtts'))
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
        exist_file_list = []
    else:
        exist_file_list = get_filepaths(outputdir)

    for line in tqdm(sen_list):
        index = line.split("#")[0]
        sen =  line.split("#")[1]
        if index in exist_file_list:
            continue
        else:
            outputpath = os.path.join(outputdir,index+'.wav')
            tts.text2wav(sen, outputpath)
            if save_info:     
                f_out.write(index+"#"+sen+"\n") 