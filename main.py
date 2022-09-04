import preprocessing
import text2speech

if __name__ == '__main__':

    raw_file_name = './raw_data.txt'
    sensitive_path = "./sensitive_word_list.txt"  # path to the sensitive word list

    preprocessing.general_filter(raw_file_name)

    #preprocessing.jieba_seg("./result_s1.txt")
    #preprocessing.ckip_seg("./result_s1.txt")
    #preprocessing.ddparser_seg("./result_s1.txt")
    #preprocessing.sensitive_filter("./result_s1_ckip.txt", sensitive_path, save_rm=True)
    #preprocessing.sensitive_filter("./result_s1_jieba.txt", sensitive_path, save_rm=True)

    #preprocessing.pos_seg_filter(input_path_ckip="./result_s1_ckip_s3.txt", save_rm=True)
    #preprocessing.pos_seg_filter(input_path_ddparser="./result_s1_ddparser.txt", save_rm=True)
    #preprocessing.pos_seg_filter(input_path_ckip="./result_s1_ckip_s3.txt", input_path_ddparser="./result_s1_ddparser.txt", save_rm=True)

    #preprocessing.get_perplexity("./result_s1_ckip_s3_ckipddp_s4.txt")
    #preprocessing.perplexity_filter("./result_s1_ckip_s3_ckipddp_s4_per.txt")

    #text2speech.tts_gtts("result_s1_ckip_s3_ckipddp_s4_per_s5.txt")
    #text2speech.tts_zhtts("result_s1_ckip_s3_ckipddp_s4_per_s5.txt")
    #text2speech.tts_ttskit("result_s1_ckip_s3_ckipddp_s4_per_s5.txt")
    #text2speech.tts_paddle("result_s1_ckip_s3_ckipddp_s4_per_s5.txt")

    #preprocessing.calculate_asr_and_intell("result_s1_ckip_s3_ckipddp_s4_per_s5.txt", "./result_s1_ckip_s3_ckipddp_s4_per_s5_gtts")
    #preprocessing.intelligibility_filter("result_s1_ckip_s3_ckipddp_s4_per_s5_asr.txt")

    #preprocessing.calculate_statistics("./raw_data.txt")
    #preprocessing.prepare_data_for_sampling("./gt_syllable_with_tone.pickle", "./candidate_sentences.txt")
