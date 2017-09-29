import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
from multiprocessing import Process
import random

# transcript: "after_other_data/new_test_after_pairs1","after_other_data/new_test_other_pairs1"
# epoch: 17, training time: 465.99 secs, train perf: 99.97 %, val perf: 83.24 %
# 
def build_data_cv(train_data_folder, test_data_folder, candidate_pool_file, cv, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    pos_file = train_data_folder[0]
    neg_file = train_data_folder[1]
    oth_file = train_data_folder[2]
    vocab = defaultdict(float)
    pos_count = 0
    with open(pos_file, "rb") as f:
        for line in f:       
            #words = line.split()
            #if int (words[0]) != cv:
            #    continue
            pos_count += 1
            text = line
            count = 0
            #for word in words:
            #    if word == '|':
            #        count += 1
            #        continue
            #    if count == 2:
            #        text += word + ' '
            rev = []
            rev.append(text.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":0, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": 1}
                      #"split": np.random.randint(0,cv)}
            revs.append(datum)
    
    neg_count = 0
    with open(neg_file, "rb") as f:
        for line in f:       
            #words = line.split()
            #if int (words[0]) != cv:
            #    continue
            neg_count += 1
            text = line
            count = 0
            #for word in words:
            #    if word == '|':
            #        count += 1
            #        continue
            #    if count == 2:
            #        text += word + ' '
            rev = []
            rev.append(text.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":1, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": 1}
                      #"split": np.random.randint(0,cv)}
            revs.append(datum)
    
    print "after_count: ", pos_count
    print "before_count: ", neg_count
    
    with open(oth_file, "rb") as f:
        oth_len = sum(1 for line in f)
        # make other training instances two times as before and after sum
        #ratio = int(oth_len / (2 *(pos_count + neg_count)) ) 
        
        #print "ratio: ", ratio
        random.seed(int(cv))
        f.seek(0)
        for line in f:      
            
            #if  random.randint(0, ratio) != 0:
            #    continue
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":2, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": 1}
                      #"split": np.random.randint(0,cv)}
            
            revs.append(datum)
        
    
    # test part
    pos_file = test_data_folder[0]
    neg_file = test_data_folder[1]
    oth_file = test_data_folder[2]
    with open(pos_file, "rb") as f:
        for line in f:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":0, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": 0}
                      #"split": np.random.randint(0,cv)}
            revs.append(datum)
    
    with open(neg_file, "rb") as f:
        for line in f:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":1, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": 0}
                      #"split": np.random.randint(0,cv)}
            revs.append(datum)
    
    with open(oth_file, "rb") as f:
        for line in f:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":2, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": 0}
                      #"split": np.random.randint(0,cv)}
            revs.append(datum)
    
    # candidate pool part
    with open(candidate_pool_file, "rb") as f:
        for line in f:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":2, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": 2}
                      #"split": np.random.randint(0,cv)}
            revs.append(datum)
    return revs, vocab
    
def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        #print word
        #raw_input("continue?")
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

def create_database(iteration_i, context_flag, context_flag2, argv1, cluster_label):
    #w2v_file = sys.argv[1]
    w2v_file = argv1 
    
    #type = "without"
    folder = context_flag + "_event_words/" + iteration_i + "/"
    train_data_folder = [folder + "train_after_pairs", folder + "train_before_pairs", folder + "train_other_pairs"]
    #test_data_folder = [folder + "test_after_pairs", folder + "test_before_pairs", folder + "test_other_pairs"]
    test_data_folder = []

    test_data_folder = ["../TempEval3/new_preprocess/TempEval_after_" + context_flag2, "../TempEval3/new_preprocess/TempEval_before_" + context_flag2, "../TempEval3/new_preprocess/TempEval_other_" + context_flag2]
        #test_data_folder = ["../TempEval3/TempEval_after_with_event", "../TempEval3/TempEval_before_with_event", "../TempEval3/TempEval_other_with_event"]
    candidate_pool_file = folder + "test_other_pairs"
    
    
    print "loading training and validating data...",        
    

    revs, vocab = build_data_cv(train_data_folder, test_data_folder, candidate_pool_file, cluster_label, clean_string=True)
    """
    In [37]: d = {'one' : [1., 2., 3., 4.], 'two' : [4., 3., 2., 1.]}
    In [38]: pd.DataFrame(d)
    Out[38]: 
       one  two
    0    1    4
    1    2    3
    2    3    2
    3    4    1

    In [39]: pd.DataFrame(d, index=['a', 'b', 'c', 'd'])
    Out[39]: 
       one  two
    a    1    4
    b    2    3
    c    3    2
    d    4    1
    """
    print "*************** cluster_label ", cluster_label, " ******************\n"
    max_l = np.max(pd.DataFrame(revs)["num_words"]) # np.max: maximum of the flattened array
    print "data loaded!"
    #print "number of training sentences: " + str(train_val_boundary)
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)
    """
    # Save a dictionary into a pickle file.
    import pickle
    favorite_color = { "lion": "yellow", "kitty": "red" }
    pickle.dump( favorite_color, open( "save.p", "wb" ) )
    
    # Load the dictionary back from the pickle file.
    import pickle
    favorite_color = pickle.load( open( "save.p", "rb" ) )
    # favorite_color is now { "lion": "yellow", "kitty": "red" }
    """
    cPickle.dump([revs, W, W2, word_idx_map, vocab], open("mr_folder_" + context_flag + "_event_words/" + iteration_i + "/mr" + str(cluster_label) + ".p", "wb"))

    print "dataset created!"
    
def classifier_process_data_main(iteration_i, context_flag, context_flag2, w2v_file):    
    create_database(iteration_i, context_flag, context_flag2, w2v_file, 0)