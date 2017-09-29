import glob
import operator
import timeit
import re
from util_class import *

# use EventPair and candidate class
                
def get_candidate_list(gen_flag, line, not_events, sentence, NERs):
    words = line.split()
    word_group_flag = 0
    word_group_member_list = []
    event_candidate_list = []
    for word in words[1:]:
        if word == '<':
            word_group_flag = 1
            word_group_member_list.append(word)
            continue
        if word_group_flag == 1 and word == '>':
            word_group_flag = 0
            word_group_member_list.append(word)
            event_cand = Candidate(gen_flag, sentence, NERs, word_group_member_list)
            if event_cand.keyword in not_events:
                word_group_member_list = []
                continue
            event_candidate_list.append(event_cand)
            word_group_member_list = []
            continue
        if word_group_flag == 1:
            word_group_member_list.append(word)
            continue
    return event_candidate_list

all_event_phrases = {}
all_possible_pairs = {}
def count_event_phrase_freq(File, not_events):
    start = timeit.default_timer()
    lines = open(File, 'r')
    for each_line in lines:
        if each_line[0][0] == '#' or not each_line.strip():
            continue
        words = each_line.split()
        if (words[0] == '<words>'):
            sentence = words
            NERs = []
            event_candidate_list = []
            continue
        elif (words[0] == '<NERs>'):
            NERs = words
            continue
        elif (words[0] == '<events_index>'):
            event_candidate_list = get_candidate_list(each_line, not_events, sentence, NERs)
            for index in range(0, len(event_candidate_list) ):
                phrase = event_candidate_list[index].string
                if not phrase in all_event_phrases:
                    all_event_phrases[phrase] = 1
                else:
                    all_event_phrases[phrase] += 1
    stop = timeit.default_timer()
    print( "# " + File + " time consuming:"),
    print(str(stop - start) )
    
def process_to_count_freq(File, not_events, selected_phrases):
    start = timeit.default_timer()
    lines = open(File, 'r')
    for each_line in lines:
        if each_line[0][0] == '#' or not each_line.strip():
            continue
        words = each_line.split()
        if (words[0] == '<words>'):
            sentence = words
            NERs = []
            event_candidate_list = []
            continue
        elif (words[0] == '<NERs>'):
            NERs = words
            continue
        elif (words[0] == '<events_index>'):
            event_candidate_list = get_candidate_list(each_line, not_events, sentence, NERs)
            #print "len(event_candidate_list)", len(event_candidate_list)
            for index in range(0, len(event_candidate_list) ):
                for second_index in range(index + 1, len(event_candidate_list) ):
                    if event_candidate_list[index].length <= 3 and event_candidate_list[second_index].length <= 3:
                        continue
                    
                    key1 = event_candidate_list[index].keyword.replace('[', '').replace(']', '')
                    key2 = event_candidate_list[second_index].keyword.replace('[', '').replace(']', '')
                    phrase1 = event_candidate_list[index].string
                    phrase2 = event_candidate_list[second_index].string
                    #print key1, key2, phrase1, phrase2
                    #raw_input("continue?")
                    if not phrase1 in selected_phrases or not phrase2 in selected_phrases:
                        continue
                    if key1 in phrase2 or key2 in phrase1:
                        continue
                    if phrase1 <= phrase2:
                        pair_instance = phrase1 + ' <==> ' + phrase2
                    else:
                        pair_instance = phrase2 + ' <==> ' + phrase1
                    if not pair_instance in all_possible_pairs:
                        all_possible_pairs[pair_instance] = 1
                    else:
                        all_possible_pairs[pair_instance] += 1
    stop = timeit.default_timer()
    print( "# " + File + " time consuming:"),
    print(str(stop - start) )

        
def generate_init_train_test_main(selected_phrases_threshold, gen_flag):
    # make sure only one copy of words_filter.txt
    not_events = not_events_load("../dic/words_filter.txt")
    
    """
    # STEP 1
    print "# STEP 1"
    for File in glob.glob("../../pre_process_sentence_context/pre_process_*"):
        count_event_phrase_freq(File, not_events)
    print ("sorting...")
    phrases_freq_sort = sorted(all_event_phrases.items(), key = operator.itemgetter(1))
    phrases_freq_sort.reverse()
    
    output_file = open('../rank_all_event_phrases_and_all_possible_pairs/rank_all_event_phrases_' + gen_flag, 'w')

    for item in phrases_freq_sort:
        output_file.write(str(item[0]) + ' ' + str(item[1]) + '\n')
    output_file.close()
    """
    """
    # STEP 2
    print "# STEP 2"
    selected_phrases = {}
    f = open('../rank_all_event_phrases_and_all_possible_pairs/rank_all_event_phrases_' + gen_flag, 'r')
    for each_line in f:
        words = each_line.split()
        if int(words[-1]) > selected_phrases_threshold:
            selected_phrases[" ".join(words[:-1])] = int(words[-1])
    
    print "len(selected_phrases)", len(selected_phrases)
    for File in glob.glob("../../pre_process_sentence_context/pre_process_*"):
        #print File
    #for File in glob.glob("../../pre_process_sentence_context/pre_process_afp_5"):
        if ".py" in File:
            continue
        process_to_count_freq(File, not_events, selected_phrases)
        
    print "len(all_possible_pairs)", len(all_possible_pairs)
    print ("sorting...")
    pairs_freq_sort = sorted(all_possible_pairs.items(), key = operator.itemgetter(1))
    pairs_freq_sort.reverse()
    
    output_file = open('../rank_all_event_phrases_and_all_possible_pairs/rank_all_possible_pairs_' + gen_flag, 'w')

    for item in pairs_freq_sort:
        if item[1] <= 2:
            continue
        output_file.write(str(item[0]) + ' ' + str(item[1]) + '\n')
    f.close()
    output_file.close()
    """
    """
    # STEP 3 (Analysis)
    f = open('rank_all_event_phrases_and_all_possible_pairs/rank_all_possible_pairs', 'r')
    line_count = 0
    line_num = sum(1 for line in f)
    f.seek(0)
    last = 1000
    for each_line in f:
        line_count += 1
        words = each_line.split()
        if int(words[-1]) < 150:
            break
        if int( words[-1] ) < last and int( words[-1] ) % 100 == 0:
            print words[-1], ' ', float(line_count) / float(line_num)
            last = int(words[-1])
    """
    
    # STEP 4
    seed_output = open("event_pair_train_test/seed_pairs_0", 'w')
    test_output = open("event_pair_train_test/test_pairs_0", 'w')
    pattern_generate_threshold = 3
    pattern_generate_seed_threshold = 10
    all_possible_pairs2 = {}
    first_thre = 50 # excactly match
    #second_thre = 100 # share one argument with initial seed
    third_thre = 100 # no shared argument but coocur enough times
    seed_pairs = {} # for the goal to remove seed pairs from test pairs
    print "1"
    f = open('event_pair_train_test/event_pairs_removed_conflict_' + gen_flag, 'r')
    seed_pairs_argu = {}
    for each_line in f:
        words = each_line.split()
        if int(words[-1]) < pattern_generate_threshold:
            continue
        if int(words[-1]) >= pattern_generate_seed_threshold:
            seed_output.write(each_line)
            eventpair = EventPair(each_line)
            seed_pairs [eventpair.event1 + ' ' + eventpair.event2] = 1
            seed_pairs [eventpair.event2 + ' ' + eventpair.event1] = 1
            if eventpair.event1 not in seed_pairs_argu:
                seed_pairs_argu[eventpair.event1] = 1
            if eventpair.event2 not in seed_pairs_argu:
                seed_pairs_argu[eventpair.event2] = 1
    f.close()
    
    print "2"
    selected_all_possible_pairs_dict = {}
    count = 0
    f = open('../rank_all_event_phrases_and_all_possible_pairs/rank_all_possible_pairs_' + gen_flag, 'r')

    for each_line in f:
        try:
            eventpair = EventPair(each_line)
        except:
            #print each_line
            #raw_input("continue?")
            pass
        all_possible_pairs2[eventpair.event1 + ' ' + eventpair.event2] = eventpair.freq
        if event_pair_filter(each_line, third_thre, not_events, set()) == True: # util_class.py
            continue
        test_output.write(each_line)
        selected_all_possible_pairs_dict[each_line] = 1
        count += 1
    print "third_thre = ", third_thre, " # no shared argument but coocur enough times: ", count
    f.close()
    
    print "3"
    count = 0
    f = open('event_pair_train_test/event_pairs_removed_conflict_' + gen_flag, 'r')
    for each_line in f:
        words = each_line.split()
        if int(words[-1]) < pattern_generate_threshold or int(words[-1]) >= pattern_generate_seed_threshold:
            continue
        eventpair = EventPair(each_line)
        phrase1 = eventpair.event1
        phrase2 = eventpair.event2
        if phrase1 <= phrase2:
            pair_instance = phrase1 + ' ' + phrase2
        else:
            pair_instance = phrase2 + ' ' + phrase1
        if pair_instance in all_possible_pairs2 and all_possible_pairs2[pair_instance] >= first_thre:
            test_output.write(each_line)
            count += 1
    print "first_thre = ", first_thre, " # excactly match: ", count
    f.close()
    """
    print "4"
    count = 1
    f = open('rank_all_event_phrases_and_all_possible_pairs/rank_all_possible_pairs', 'r')
    for each_line in f:
        words = each_line.split()
        if int(words[-1]) < second_thre:
            break
        eventpair = EventPair(each_line)
        phrase1 = eventpair.event1
        phrase2 = eventpair.event2
        #print phrase1, phrase2
        if (phrase1 in seed_pairs_argu or phrase2 in seed_pairs_argu) and eventpair.freq >= second_thre \
        and each_line not in selected_all_possible_pairs_dict:
            test_output.write(each_line)
            count += 1
            
    print "second_thre = ", second_thre, " # share one argument with initial seed: ", count
    f.close()
    """
    seed_output.close()
    test_output.close()
    
    print "over!"
