from nltk.stem.snowball import SnowballStemmer
from util_class import *
#from nltk.stem.wordnet import WordNetLemmatizer


def transform_format(line):
    words = line.split()
    for i, word in enumerate(words):
        if word in ["<==", "==>", "<==>"] and i != 0:
            relation_index = i
    #pair = " ".join(words[1:-4])
    words[relation_index] = words[0]
    #return pair," ".join(words[1:])
    return " ".join(words[1:])
        

def generate_new_seed_test_main(iteration_i, type1_threshold, type2_threshold):
#if __name__ == "__main__":
#    iteration_i = "1"
    stemmer = SnowballStemmer("english")
    #lemma = WordNetLemmatizer()
    f = open("error_analysis/" + iteration_i + "/1_Classifier_combine_classifiers", 'r')
    previous_seed_list = open("event_pair_train_test/seed_pairs_" + str(int(iteration_i) - 1), 'r')
    new_seed_pairs = []
    exist_pairs = {}
    # only consider event key to do the filtering
    exist_pairs_event_only = {}
    for line in previous_seed_list:
        if not line.strip():
            continue
        new_seed_pairs.append(line)
        eventpair = EventPair(line)
        exist_pairs[eventpair.event1 + ' ' + eventpair.event2] = 1
        exist_pairs[eventpair.event2 + ' ' + eventpair.event1] = 1
        event1_key = stemmer.stem(eventpair.event1_key.replace('[','').replace(']', ''))
        event2_key = stemmer.stem(eventpair.event2_key.replace('[','').replace(']', ''))
        if eventpair.relation == '<==':
            instance = event1_key + ' <== ' + event2_key
            r_instance = event2_key + ' <== ' + event1_key
        elif eventpair.relation == '==>':
            instance = event2_key + ' <== ' + event1_key
            r_instance = event1_key + ' <== ' + event2_key
        #if r_instance in exist_pairs_event_only:
        #    print line
        #    print r_instance
        #    raw_input('continue?')
        if r_instance not in exist_pairs_event_only:
            exist_pairs_event_only[instance] = 1

    previous_seed_list.close()
    
    for line in f:
        if not line.strip():
            continue
        words = line.split()
        if int (words[-1]) < type1_threshold:
            continue
        if int (words[-1]) < type2_threshold and int(words[-1]) >= type1_threshold and '<==>' in line:
            continue


        transformed = transform_format(line)
        
        eventpair = EventPair(transformed)
        event1_key = stemmer.stem(eventpair.event1_key.replace('[','').replace(']', ''))
        event2_key = stemmer.stem(eventpair.event2_key.replace('[','').replace(']', ''))
        if eventpair.event1 + ' ' + eventpair.event2 in exist_pairs or eventpair.event2 + ' ' + eventpair.event1 in exist_pairs:
            continue
        if eventpair.relation == '<==':
            instance = event1_key + ' <== ' + event2_key
            r_instance = event2_key + ' <== ' + event1_key
        elif eventpair.relation == '==>':
            instance = event2_key + ' <== ' + event1_key
            r_instance = event1_key + ' <== ' + event2_key
        if r_instance in exist_pairs_event_only:
            #print line
            #print r_instance
            #raw_input('continue?')
            continue
        else:
            exist_pairs_event_only[instance] = 1

        exist_pairs[eventpair.event1 + ' ' + eventpair.event2] = 1
        exist_pairs[eventpair.event2 + ' ' + eventpair.event1] = 1


        new_seed_pairs.append(transformed + '\n')

    f.close()
    

    new_seed_pairs_output = open("event_pair_train_test/seed_pairs_" + str(iteration_i), 'w')

    for line in new_seed_pairs:
        if not line.strip():
            continue
        new_seed_pairs_output.write(line)
    
    f = open("event_pair_train_test/test_pairs_" + str(int(iteration_i) - 1), 'r')
    new_other_output = open("event_pair_train_test/test_pairs_" + iteration_i, 'w')
    #print exist_pairs

    for line in f:
        if not line.strip():
            continue
        words = line.split()
        eventpair = EventPair(line)
        #print eventpair.event1 + ' ' + eventpair.event2
        if eventpair.event1 + ' ' + eventpair.event2 in exist_pairs or eventpair.event2 + ' ' + eventpair.event1 in exist_pairs:
            continue

        new_other_output.write(line)
    new_other_output.close()
    print "over"