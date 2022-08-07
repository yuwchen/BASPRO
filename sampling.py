import os
import copy
import time
import random
import argparse
import numpy as np

num_of_set = 20  #numbers of the set in the corpus
num_of_sen_in_set = 20 #numbers of the sentences in a set
population_size = 100 #initial population size of the GA
iteration = 50 # numbers of interation for GA

#truth_syllable: distribution that close to real-world condition
truth_syllable = np.load('gt_syllable_distribution.npy') 
num_of_syllable = truth_syllable.shape[0]

#idx_syllable: the mapping between a sentence index and the corresponding syllabus distribution
idx_syllable = np.load("idx_syllales.npy")
num_of_total_sen = idx_syllable.shape[0]

#idx_content: the mapping between a sentence index and the corresponding content
idx_content = np.load("idx_content.npy")


# numbers of sentences in the corpus
corpus_sen_seq = list(np.arange(num_of_set*num_of_sen_in_set))

dis_weight = 1        # fitness weight of the syllabus distribution between the corpus and real-world
coverage_weight = 2   # fitness weight of the syllabus coverage
dis_weight_set=1      # fitness weight of the syllabus distribution bewteen sets


def initialization(excluded_idx, initial_chro=None):
    """
    initial the population 
    input:
    - excluded_idx(list): excluded sentence index
    - initial_chro(numpy): if not None, 
    all chromosome in the initial population will base on the initial chromosome 

    output:
    - population(numpy): initial population, dimension:(population_size, num_of_set, num_of_sen_in_set)

    """
    if initial_chro is None: # start from scratch

        total_sen_seq = list(set(np.arange(num_of_total_sen))-set(excluded_idx))
        population = np.zeros((population_size, num_of_set, num_of_sen_in_set), dtype=int)

        for p_idx in range(population_size):
            chromosome = random.sample(total_sen_seq, num_of_set*num_of_sen_in_set)
            chromosome = np.reshape(chromosome, (num_of_set, num_of_sen_in_set))
            population[p_idx,:,:] = chromosome  

    else: # start from previous result
        total_sen_seq = list(set(np.arange(num_of_total_sen))-set(list(initial_chro.flatten())))

        population_new = np.zeros((population_size, num_of_set, num_of_sen_in_set), dtype=int)
        excluded_idx = np.asarray(excluded_idx)
        index_mask = np.isin(initial_chro, excluded_idx) 
        population = np.zeros((population_size, num_of_set, num_of_sen_in_set), dtype=int)

        for p_idx in range(population_size):
            chromosome = random.sample(total_sen_seq, num_of_set*num_of_sen_in_set)
            chromosome = np.reshape(chromosome, (num_of_set, num_of_sen_in_set))
            population_new[p_idx,:,:] = chromosome
            population[p_idx,:,:] = initial_chro #all chromosomes in the population are based on the initial chromosome

        # replace the excluded sentences with other sentence candidate
        population[:,index_mask] = population_new[:,index_mask]     

    return population


def cosine_similarity(input):
    """
    calculate the cosine similarity between real-world and input syllable distribution 
    input: 
    - input: syllable distribution of a sentence (numpy), dimension:(numbers_of_syllabas, 1)
    output:
    - cos_sim: cosine similarity score (float)
    """
    cos_sim = np.dot(input, truth_syllable)/(np.linalg.norm(input)*np.linalg.norm(truth_syllable))
    return cos_sim

def calculate_fitness(chromosome, return_details=False):
    """
    calculate the fitness score of input chromosome
    input: 
    - chromosome (numpy): dimension:
    output: 
    - score: fitness score (float)
    """

    # the distribution of all set in population are close to the truth distribution
    the_syllables = np.zeros((num_of_set, num_of_sen_in_set, num_of_syllable))
    for s_idx in range(num_of_set):
        the_syllables[s_idx,:,:] = idx_syllable[chromosome[s_idx]]

    #some syllables of every set
    set_syllable = np.sum(the_syllables, axis=1)
    #some syllables of all population
    chro_syllable = np.sum(set_syllable, axis=0)

    #calculate syllable distribution of whole chromosome 
    distribution_score = cosine_similarity(chro_syllable)
    distribution_score_set = [cosine_similarity(the_set_syllable) for the_set_syllable in set_syllable]
    distribution_score_set = sum(distribution_score_set)/len(distribution_score_set)

    # average converage score of each set 
    # each set should cover as much numbers of syllables as possible
    # count_nonzero = numbers of syllables (converage)
    
    #converage_score = np.count_nonzero(set_syllable)/num_of_set/num_of_syllable
    converage_score = np.count_nonzero(chro_syllable)/num_of_syllable
    #print(distribution_score,converage_score)
    score = dis_weight*distribution_score+dis_weight_set*distribution_score_set+coverage_weight*converage_score
    if not return_details:
        return score
    else:
        return distribution_score, distribution_score_set, converage_score

def selection(population):
    """
    apply selection 
    input: 
    - population: original population (numpy), dimension:(population_size, num_of_set, num_of_sen_in_set)
    output:
    - new_population: population after selection (numpy), dimension:(population_size, num_of_set, num_of_sen_in_set)
    """

    # calculate the fitness score of all chromosome in the population
    p_score = [calculate_fitness(chro) for chro in population]
    new_population = truncation_selection(population, p_score)
    return new_population

def truncation_selection(population, p_score):
    """
    Apply truncation selection:
    replace half of the chromosomes with the other half with a higher fitness score
    input:
    - population
    - p_score: fitness score of the population(numpy). dimension:(population_size, 1)
    output:
    - new_population: populaiton after selection
    """

    # find the index of chromosomes that have higher fitness score
    selected_sen = np.argpartition(p_score, int(len(p_score)/2))[int(len(p_score)/2):]
    selected_sen = np.concatenate((selected_sen,selected_sen))
    new_population = population[selected_sen]
    np.random.shuffle(new_population)
    return new_population


def crossover_chromosome(chro_a, chro_b):
    """
    Apply crossover to the chromosome
    input:
    - chro_a and chro_b (numpy): chromosome A and chromosome B, dimension:(num_of_set, num_of_sen_in_set)
    output:
    - new_a and new_b (numpy): new chromosome dimension:(num_of_set, num_of_sen_in_set)
    """

    new_a = np.zeros(chro_a.shape)
    new_b = np.zeros(chro_b.shape)

    # find the overlap gene in two chromosome
    fix_gene = set(chro_a.flatten())&set(chro_b.flatten())

    #apply crossover to each set
    for s_idx in range(num_of_set):

        #chromosomeA set i crossover chromosome B set i
        the_set_a = set(chro_a[s_idx])
        the_set_b = set(chro_b[s_idx])

        #find index that will cause duplicate sentences
        fix_a = the_set_a & fix_gene
        fix_b = the_set_b & fix_gene
        #exclude duplicate sentences from crossover
        cross_a = list(the_set_a - fix_a)
        cross_b = list(the_set_b - fix_b)

        #crossover chromosomes that might have different length
        length = np.minimum(len(cross_a), len(cross_b))
        new_set_a = np.copy(cross_a)
        new_set_b = np.copy(cross_b)

        #one-point crossover, can perform crossover only if chromosome length >=2
        if length >=2:
            cross_point = np.random.randint(length, size=1)[0]
            new_set_a[cross_point:length] = cross_b[cross_point:length]
            new_set_b[cross_point:length] = cross_a[cross_point:length]

        new_set_a = list(set(new_set_a).union(fix_a))
        new_set_b = list(set(new_set_b).union(fix_b))
        new_a[s_idx,:] = new_set_a
        new_b[s_idx,:] = new_set_b

    return new_a, new_b

def crossover(population):
    """
    Apply crossover to population
    input:
    - population: input population, dimension:(population_size, num_of_num_of_set, num_of_sen_in_set)
    output:
    - new_population (numpy): new population, dimension:(population_size, num_of_num_of_set, num_of_sen_in_set)
    """

    new_population = np.zeros((population_size, num_of_set, num_of_sen_in_set), dtype=int)
    for p_idx in range(0, population_size, 2):
    
        new_a, new_b = crossover_chromosome(population[p_idx],population[p_idx+1])
        new_population[p_idx] = new_a
        new_population[p_idx+1] = new_b

    return new_population


def calculate_population_fitness(population, return_population=True):
    p_score = [calculate_fitness(chro) for chro in population]
    if not return_population:
        return np.mean(p_score), np.max(p_score)
    else:
        max_idx = np.argmax(p_score)
        return np.mean(p_score), p_score[max_idx], population[max_idx]

def isclose(a, b, rel_tol=1e-08, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--initial_dir', type=str, default=None)
    parser.add_argument('--excluded', type=str, default=None)
    parser.add_argument('--outputdir', type=str, default="set{}_sen{}_p{}".format(num_of_set, num_of_sen_in_set, population_size))

    args = parser.parse_args()
    if args.excluded is not None:
        with open(args.excluded) as f:
            excluded_idx = [ ]
            for line in f.read().splitlines():
                excluded_idx.append(int(line))
    else:
        excluded_idx = [ ]

    if args.initial_dir is not None:
        initial_chro = np.load(os.path.join(args.initial_dir, 'best_chro.npy'))
        output_path = os.path.join(args.initial_dir,'resampled')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print("initial population ...")
        population = initialization(excluded_idx, initial_chro)
    else:
        print("initial population ...")
        population = initialization(excluded_idx)
        output_path = args.outputdir
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    f_out = open(os.path.join(output_path,'log.txt'),'w')
    f_out.write("fitness_max:fitness_mean\n")

    f_best = 0

    f_mean, f_max, best_chromosome = calculate_population_fitness(population)
    print("initial fitness", f_mean, f_max)

    print("start interaction")
    f_mean_previous = 0
    for iter in range(iteration):
            
        population = selection(population)
        #print("after selection",calculate_population_fitness(population, False))
        population = crossover(population)

        f_mean, f_max, best_chromosome = calculate_population_fitness(population)
        print("iter",iter, f_mean, f_max)

        f_out.write(+str(f_max)+":"+str(f_mean)+"\n")


        if f_max > f_best:
            f_best = f_max
            np.save(os.path.join(output_path,'best_chro'), best_chromosome)
        if iter % 200==0:
            np.save(os.path.join(output_path,'best_chro_iter'+str(iter)), best_chromosome)

        if isclose(f_mean_previous, f_mean, rel_tol=1e-07, abs_tol=0.0):
            break
        
        f_mean_previous = f_mean


    f_mean, f_max, best_chromosome = calculate_population_fitness(population)
    np.save(os.path.join(output_path,'final_chro'), best_chromosome)
    distribution_score, distribution_score_set, converage_score = calculate_fitness(best_chromosome, return_details=True)
    f_out.write("distribution_score:"+str(distribution_score)+",distribution_score_set:"+str(distribution_score_set)+",converage_score:"+str(converage_score)+"\n")


    f_corpus = open(os.path.join(output_path,'corpus.txt'),'w')
    f_corpus.write("index_in_sentence_candidates:"+"set_idx:"+"sentence_idx:"+"content:"+'\n')
    result_content = idx_content[best_chromosome]
    for set_idx in range(result_content.shape[0]):
        for sen_idx in range(result_content.shape[1]):
            f_corpus.write(str(best_chromosome[set_idx,sen_idx])+":"+str(set_idx)+":"+str(sen_idx)+':'+result_content[set_idx,sen_idx]+'\n')
        

            
