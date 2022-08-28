import os
import numpy as np
import argparse
from tqdm import tqdm


idx_content = np.load('idx_content.npy') #the mapping between a sentence index and the corresponding content
truth_syllable = np.load('gt_syllable_distribution.npy') #distribution that close to real-world condition
idx_syllable = np.load("idx_syllables.npy") #the mapping between a sentence index and the corresponding syllabus distribution


num_of_syllable = truth_syllable.shape[0]

num_of_set = 5
num_of_sen_in_set=5
dis_weight = 1
coverage_weight = 2
dis_weight_set=1

def calculate_fitness(chromosome, return_details=False):
    """
    calculate the fitness score of input chromosome
    input: 
    - chromosome (numpy)
    output: 
    - score: fitness score (float)
    """

    # the distribution of all set in population are close to the truth distribution
    the_syllables = np.zeros((num_of_set, num_of_sen_in_set, num_of_syllable))
    for s_idx in range(num_of_set):
        the_syllables[s_idx,:,:] = idx_syllable[chromosome[s_idx]]

    #sum syllables of every set
    set_syllable = np.sum(the_syllables, axis=1)
    #sum syllables of all population
    chro_syllable = np.sum(set_syllable, axis=0)

    #calculate syllable distribution of whole chromosome 
    distribution_score = cosine_similarity(chro_syllable)
    distribution_score_set = [cosine_similarity(the_set_syllable) for the_set_syllable in set_syllable]
    distribution_score_set = sum(distribution_score_set)/len(distribution_score_set)

    converage_score = np.count_nonzero(chro_syllable)/num_of_syllable

    score = dis_weight*distribution_score+dis_weight_set*distribution_score_set+coverage_weight*converage_score
    if not return_details:
        return score
    else:
        return score, distribution_score, distribution_score_set, converage_score


def cosine_similarity(input):
    cos_sim = np.dot(input, truth_syllable)/(np.linalg.norm(input)*np.linalg.norm(truth_syllable))
    return cos_sim


def found_best_chro(chromosome,set_idx,sen_idx,idx_pool):
    """
    find the best replacement sentence
    input: 
    - chromosome (numpy), set_idx(str), sen_idx(str), idx_pool(numpy)
    - chromosome[set_idx, sen_idx] is the sentence that needs to be replaced
    output: 
    - score: fitness score (float)
    """   
    score_list = []
    idx_list = []
    max_score = -1
    #loop over all candidate sentences
    for i in tqdm(idx_pool):
        chromosome_copy = np.copy(chromosome)
        chromosome_copy[set_idx,sen_idx]=i
        the_score = calculate_fitness(chromosome_copy)
        score_list.append(the_score)
        idx_list.append(i)
        if the_score > max_score:
                max_score = the_score
                max_idx = i

    return max_idx, score_list, idx_list, max_score


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--initial_dir', type=str, default=None)
    parser.add_argument('--excluded', type=str, default=None)

    args = parser.parse_args()

    # create output dir
    outputdir = os.path.join(args.initial_dir,'resampled_greedy_'+args.excluded.split(os.sep)[-1].replace(".txt",""))
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    with open(args.excluded) as f:
        rm_idx = [ ]
        for line in f.read().splitlines():
            rm_idx.append(int(line))

    f_out = open(os.path.join(outputdir,'log.txt'),'w')

    chro_path = os.path.join(args.initial_dir,"best_chro.npy")
    chromosome = np.load(chro_path)
    best_chro_flatten = chromosome.flatten()
    all_chro = np.arange(idx_syllable.shape[0])
    rest = np.setdiff1d(all_chro, best_chro_flatten)
    idx_pool = np.setdiff1d(all_chro, best_chro_flatten)
    print("Numbers of sen in the corpus: ",len(best_chro_flatten))
    print("Number of candidate chromosomes:",len(all_chro),len(rest))

    # calculate the initial fitness score
    initial_fitness, distribution_score, distribution_score_set, converage_score = calculate_fitness(chromosome, True)
    f_out.write("initial_fitness:"+str(initial_fitness)+" distribution_score:"+str(distribution_score)+" distribution_score_set:"+str(distribution_score_set)+" converage_score:"+str(converage_score)+"\n")

    for i in range(num_of_set):
        for j in range(num_of_sen_in_set):
            input = chromosome[i, j]
            if chromosome[i, j] in rm_idx:
                new_idx, score_list, idx_list, max_score = found_best_chro(chromosome,i,j,idx_pool)
                chromosome[i, j] = new_idx
                idx_pool = np.delete(idx_pool, np.argwhere(idx_pool == new_idx))
                f_out.write("remove index:"+str(chromosome[i, j])+" fitness score:"+str(max_score)+"\n")

    final_fitness, distribution_score, distribution_score_set, converage_score = calculate_fitness(chromosome, True)
    f_out.write("final_fitness:"+str(final_fitness)+" distribution_score:"+str(distribution_score)+" distribution_score_set:"+str(distribution_score_set)+" converage_score:"+str(converage_score)+"\n")
    
    np.save(os.path.join(outputdir,"best_chro_greedy.npy"),chromosome)

    f_corpus = open(os.path.join(outputdir,'corpus_greedy.txt'),'w')
    f_corpus.write("index_in_sentence_candidates:"+"set_idx:"+"sentence_idx:"+"content:"+'\n')
    for set_idx in range(chromosome.shape[0]):
        for sen_idx in range(chromosome.shape[1]):
            the_idx = chromosome[set_idx,sen_idx]
            f_corpus.write(str(the_idx)+":"+str(set_idx)+':'+str(sen_idx)+':'+idx_content[the_idx]+'\n')

