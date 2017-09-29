
from StringIO import StringIO
from sets import Set
import glob
import timeit
import gc
import gzip
from random import randint
import random
import networkx as nx
from util_class import *
from multiprocessing import Process
import os

# use EventPair and candidate class

def get_context_surrounding(sentence, dependencies, context_flag2, event_cand1, event_cand2):# relation_type == ' ==> ' or ' <== ' or ' <==> '
    if context_flag2 == "tree_path":
        valid_index = []
        event1_index = event_cand1.keyIndex
        event2_index = event_cand2.keyIndex
        local_context = []
        sur_context = ""
        G = nx.Graph()
        pobj_dict = {}
        for entity in dependencies[1:-1]:
            words = entity.split()
            head = words[0]
            tail = words[2]
            #if (head == event1_index and tail == event2_index) or (tail == event1_index and head == event2_index):
            #    return None, None
            edge = (head, tail)
            G.add_edge(*edge)
            if words[1] == "pobj":
                pobj_dict[head] = tail
            if head in [event1_index, event2_index]:
                local_context.append(tail)
            elif tail in [event1_index, event2_index]:
                local_context.append(head)
        try: 
            path = nx.shortest_path(G, source = event1_index, target = event2_index)
        except:
            return None, None

        additional_context = []
        for entity in dependencies[1:-1]:
            words = entity.split()
            head = words[0]
            tail = words[2]
            if head in path and words[1] in ["prep", "pobj", "aux", "auxpass", "conj", "cc", "punct"]:
                additional_context.append(tail)
                if words[1] == "prep" and tail in pobj_dict:
                    additional_context.append(pobj_dict[tail])
            elif tail in path and words[1] in ["prep", "pobj", "aux", "auxpass", "conj", "cc", "punct"]:
                additional_context.append(head)

        valid_index = list(set(path + local_context + additional_context))
        valid_index = map(int, valid_index) # change string list to int list
        valid_index.sort()
        valid_index = map(str, valid_index)
        for i in valid_index:
            if int(i) <=0 or int(i) >= len(sentence):
                valid_index.remove(i)
        for i in valid_index:
            #if sentence[ int(i) ] == 'after' or sentence[ int(i) ] == 'before':
            #    return None, None
            sur_context += sentence[int(i) ] + ' '
        return valid_index, sur_context

    elif context_flag2 == "window":
        num_words_in_between = 10
        num_words_before_or_after = 5
        valid_index = []
        event1_index = int(event_cand1.keyIndex)
        event2_index = int(event_cand2.keyIndex)
        sur_context = ""
        #if (event2_index - num_words_in_between/2 >=1 and event1_index +num_words_in_between/2 <= len(sentence) -1 ):
            
        for i in range( max( 1, event1_index - num_words_before_or_after ), event1_index):
            valid_index.append(i)
        valid_index.append(event1_index)
        if (event1_index + num_words_in_between/2 <= event2_index - num_words_in_between/2):
            for i in range(event1_index + 1, event1_index + num_words_in_between/2+1):
                valid_index.append(i)
            for i in range(event2_index - num_words_in_between/2, event2_index):
                valid_index.append(i)
        else:
            for i in range(event1_index + 1, event2_index):
                valid_index.append(i)
        valid_index.append(event2_index)
        for i in range(event2_index + 1, min(len(sentence) - 1, event2_index + num_words_before_or_after) + 1):
            valid_index.append(i)
        
        for i in event_cand1.allIndex:
            if not int(i) in valid_index:
                valid_index.append( int (i) )
        for i in event_cand2.allIndex :
            if not int(i) in valid_index:
                valid_index.append( int (i) )
        valid_index.sort()
        valid_index = map(str, valid_index)

        for i in valid_index:
            if int(i) <=0 or int(i) >= len(sentence):
                valid_index.remove(i)
        for i in valid_index:
            #if sentence[ int(i) ] == 'after' or sentence[ int(i) ] == 'before':
            #    return None, None
            sur_context += sentence[int (i) ] + ' '
        return valid_index, sur_context
        #else:
        #    return None, None

def machine_learning_set(input_f1, input_f2):
    train_f = open(input_f1, 'r')
    test_f = open(input_f2, 'r')
    train_dict = {}
    for each_line in train_f:
        words = each_line.split()
        e1 = ""
        e2 = ""
        relation = ""
        relation_index = -1
        end_index = -1
        angle_brackets_count = 0
        for i, word in enumerate(words[:-1]):
            if word == "==>" or word == "<==":
                relation_index = i
                relation = word
                continue
            if word in ['<', '>']:
                angle_brackets_count += 1
            if angle_brackets_count == 4 and word.isdigit():
                end_index = i
                break
        e1 = " ".join(words[:relation_index])
        e2 = " ".join(words[relation_index+1 : end_index])
        if relation == "==>":
            train_dict[e1+ ' ' + e2] = "==>"
            train_dict[e2+ ' ' + e1] = "<=="
        if relation == "<==":
            train_dict[e1+ ' ' + e2] = "<=="
            train_dict[e2+ ' ' + e1] = "==>"
        
        
    test_dict = {}
    for each_line in test_f:
        words = each_line.split()
        e1 = ""
        e2 = ""
        relation = ""
        relation_index = -1
        end_index = -1
        angle_brackets_count = 0
        for i, word in enumerate (words[:-1]):
            if word == "==>" or word == "<==" or word == "<==>":
                relation = word
                relation_index = i
                continue
            if word in ['<', '>']:
                angle_brackets_count += 1
            if angle_brackets_count == 4 and word.isdigit():
                end_index = i
                break
        e1 = " ".join(words[:relation_index])
        e2 = " ".join(words[relation_index+1 : end_index])
        if e1 <= e2:
            pair_instance = e1 + ' ' + e2
            test_dict[pair_instance] = relation
        else:
            pair_instance = e2 + ' ' + e1
            if relation == "<==>":
                test_dict[pair_instance] = "<==>"
            elif relation == "<==":
                test_dict[pair_instance] = "==>"
            elif relation == "==>":
                test_dict[pair_instance] = "<=="
    """
    for ele in train_dict:
        print ele
        print train_dict[ele]
        raw_input("continue?")
    """
    print "train_dict: ", len(train_dict)
    print "test_dict: ", len(test_dict)
    return train_dict, test_dict

def context_exclude_event_words(sentence, event_cand1, event_cand2, valid_index):
    result = ""
    for i in valid_index:
        if not i in event_cand1.allIndex and not i in event_cand2.allIndex:
            result += sentence[int(i)] + ' '
    return result

#event_pairs_readable = {}


def get_highlight_sentence(sentence, event_cand1, event_cand2):
    result_list = []
    for i, word in enumerate(sentence):
        if str(i) == event_cand1.keyIndex or str(i) == event_cand2.keyIndex:
            result_list.append('['+ word +']')
        elif str(i) in event_cand1.allIndex or str(i) in event_cand2.allIndex:
            result_list.append('('+ word + ')')
        else:
            result_list.append(word)
    return " ".join(result_list)

def reverse_relation(relation):
    if relation == "<==>":
        return "<==>"
    elif relation == "<==":
        return "==>"
    else:
        return "<=="



def populate(iteration_i, idx, gen_flag, context_flag2, input_f1, input_f2, File, other_ratio, final_test_classifier):
    print File
    #global train_after_before_count, train_other_count
    train_after_before_count = 0
    train_other_count = 0

    train_dict, test_dict = machine_learning_set(input_f1, input_f2)
    """
    for ele in train_dict:
        print ele
        raw_input("continue?")
    """
    
    with_folder = "with_event_words/" + iteration_i + '/'
    
    train_before_output = open(with_folder + 'train_before_pairs' + str(idx), 'w')
    train_after_output = open(with_folder + 'train_after_pairs' + str(idx), 'w')
    train_other_output = open(with_folder + 'train_other_pairs' + str(idx), 'w')

    track_train_before_output = open(with_folder + 'track_train_before_pairs' + str(idx), 'w')
    track_train_after_output = open(with_folder + 'track_train_after_pairs' + str(idx), 'w')
    track_train_other_output = open(with_folder + 'track_train_other_pairs' + str(idx), 'w')
    
    test_other_output = open(with_folder + 'test_other_pairs' + str(idx), 'w')

    track_test_other_output = open(with_folder + 'track_test_other_pairs' + str(idx), 'w')
    
    other_sentences = open(with_folder + 'other_pairs_sentences' + str(idx), 'w')
    train_after_sentences = open(with_folder + 'train_after_pairs_sentences' + str(idx), 'w')
    train_before_sentences = open(with_folder + 'train_before_pairs_sentences' + str(idx), 'w')
    
    without_folder = "without_event_words/" + iteration_i + '/'
    w_train_before_output = open(without_folder + 'train_before_pairs' + str(idx), 'w')
    w_train_after_output = open(without_folder +'train_after_pairs' + str(idx), 'w')
    w_train_other_output = open(without_folder +'train_other_pairs' + str(idx), 'w')
    
    w_track_train_before_output = open(without_folder +'track_train_before_pairs' + str(idx), 'w')
    w_track_train_after_output = open(without_folder +'track_train_after_pairs' + str(idx), 'w')
    w_track_train_other_output = open(without_folder +'track_train_other_pairs' + str(idx), 'w')
    
    w_test_other_output = open(without_folder +'test_other_pairs' + str(idx), 'w')
    
    w_track_test_other_output = open(without_folder +'track_test_other_pairs' + str(idx), 'w')
    
    
    
    lines_count = 0
    sentence_count = 0
    random.seed(1)
    #other_ratio = 500

    not_events = not_events_load("../dic/words_filter.txt")

    
    lines = open(File, 'r')
    for each_line in lines:
        #if lines_count >= 500000:
        #    break
        if each_line[0][0] == '#' or not each_line.strip():
            continue
        words = each_line.split()
        if len(words) <=2:
            continue
        if (words[0] == '<words>'):
            sentence = words
            NERs = []
            dependencies = []
            event_candidate_list = []
            affi_dict = {}
            sentence_count += 1
            continue
        elif (words[0] == '<NERs>'):
            NERs = words
            continue
        elif words[0] == '<dependencies>':
            dependencies = each_line.split('|')
            continue
        elif (words[0] == '<events_index>'):
            word_group_flag = 0
            word_group_member_list = []
            for word in words[1:]:
                if word == '<':
                    word_group_flag = 1
                    word_group_member_list.append(word)
                    continue
                if word_group_flag == 1 and word == '>':
                    word_group_flag = 0
                    word_group_member_list.append(word)
                    event_cand = Candidate(gen_flag, sentence, NERs, word_group_member_list)
                    event_candidate_list.append(event_cand)
                    word_group_member_list = []
                    
                    continue
                if word_group_flag == 1:
                    word_group_member_list.append(word)
                    continue
            for index in range(0, len(event_candidate_list) ):
                for second_index in range(index+1, len(event_candidate_list) ):
                    event1_index = int( event_candidate_list[index].keyIndex )
                    event2_index = int( event_candidate_list[second_index].keyIndex )
                    event1_key = event_candidate_list[index].keyword
                    event2_key = event_candidate_list[second_index].keyword
                    if abs(event1_index - event2_index) > 10 or abs(event1_index - event2_index) < 2:
                        continue
                    e1 = ""
                    e2 = ""
                    
                    e1 = event_candidate_list[index].string
                    e2 = event_candidate_list[second_index].string

                    if e1 + ' ' + e2 in train_dict:
                        relation = train_dict[e1 + ' ' + e2]
                        valid_index, sur_context = get_context_surrounding(sentence, dependencies, context_flag2, event_candidate_list[index], event_candidate_list[second_index] )
                        if sur_context == None:
                            continue
                        exclude_event_words = context_exclude_event_words(sentence, event_candidate_list[index], event_candidate_list[second_index], valid_index)
                        if len(exclude_event_words.split()) < 6:
                            continue
                        if relation == "==>":
                            #train_before_output.write(sur_context + e1 + ' ' + e2 + '\n')
                            train_before_output.write(sur_context + '\n')
                            pair = e1 + ' ' + relation + ' ' + e2
                            w_train_before_output.write(exclude_event_words + '\n')
                            track_train_before_output.write( pair + '\n')
                            w_track_train_before_output.write(pair + '\n')

                            train_before_sentences.write(get_highlight_sentence(sentence, event_candidate_list[index], event_candidate_list[second_index]) + '\n')
                            #event_pairs_readable[e1 + relation + e2] = readable_e1 + ' ' + relation + ' ' + readable_e2
                        if relation == "<==":
                            #train_after_output.write(sur_context + e1 + ' ' + e2 +'\n')
                            train_after_output.write(sur_context + '\n')
                            pair = e1 + ' ' + relation + ' ' + e2
                            w_train_after_output.write(exclude_event_words + '\n')
                            track_train_after_output.write(pair + '\n')
                            w_track_train_after_output.write(pair + '\n')

                            #event_pairs_readable[pair] = readable_e1 + ' ' + relation + ' ' + readable_e2
                            train_after_sentences.write(get_highlight_sentence(sentence, event_candidate_list[index], event_candidate_list[second_index]) + '\n')
                        lines_count += 1
                        train_after_before_count += 1
                            
                    elif e1 + ' ' + e2 in test_dict or e2 + ' ' + e1 in test_dict:
                        #if random.randint(0, 9) != 0:
                        #    continue
                        valid_index, sur_context = get_context_surrounding(sentence, dependencies, context_flag2, event_candidate_list[index], event_candidate_list[second_index] )
                        if sur_context == None:
                            continue
                        exclude_event_words = context_exclude_event_words(sentence, event_candidate_list[index], event_candidate_list[second_index], valid_index)
                        #if len (exclude_event_words.split() ) < 8:
                        #    continue
                        reverse_flag = 0
                        if e1 <= e2:
                            pair_instance = e1 + ' ' + e2
                        else:
                            reverse_flag = 1
                            pair_instance = e2 + ' ' + e1
                        relation = test_dict[pair_instance]
                        
                        test_other_output.write(sur_context + '\n')
                        w_test_other_output.write(exclude_event_words + '\n')
                        
                        if reverse_flag == 0:
                            track_test_other_output.write(e1 + ' ' + relation + ' ' + e2 + '\n')
                            w_track_test_other_output.write(e1 + ' ' + relation + ' ' + e2 + '\n')
                        else:
                            track_test_other_output.write(e1 + ' ' + reverse_relation(relation) + ' ' + e2 + '\n')
                            w_track_test_other_output.write(e1 + ' ' + reverse_relation(relation) + ' ' + e2 + '\n')

                        other_sentences.write(get_highlight_sentence(sentence, event_candidate_list[index], event_candidate_list[second_index]) + '\n')
                        lines_count += 1
                    elif len(e1.split()) + len( e2.split()) >= 6:
                        random_number = random.randint(0, 20)
                        if random_number == 0:
                            if train_other_count > other_ratio * train_after_before_count:
                                continue
                            if other_sentence_filter(not_events, event1_key, event2_key) == True:
                                continue

                            valid_index, sur_context = get_context_surrounding(sentence, dependencies, context_flag2, event_candidate_list[index], event_candidate_list[second_index] )
                            if sur_context == None:
                                continue
                            sur_context_without_event = context_exclude_event_words(sentence, event_candidate_list[index], event_candidate_list[second_index], valid_index)
                            if len(sur_context_without_event.split()) < 6:
                                continue
                            
                            #test_other_output.write(sur_context + e1 + ' ' + e2 +'\n')
                            train_other_output.write(sur_context + '\n')
                            w_train_other_output.write(sur_context_without_event + '\n')
                            track_train_other_output.write(e1 + ' <==> ' + e2  + '\n')
                            w_track_train_other_output.write(e1 + ' <==> ' + e2  + '\n')
                            lines_count += 1
                            train_other_count += 1
    
        
    lines.close()
    train_before_output.close()
    train_after_output.close()
    train_other_output.close()
    
    track_train_before_output.close()
    track_train_after_output.close()
    track_train_other_output.close()
    
    test_other_output.close()
    track_test_other_output.close()
    
    
    w_train_before_output.close()
    w_train_after_output.close()
    w_train_other_output.close()
    
    w_track_train_before_output.close()
    w_track_train_after_output.close()
    w_track_train_other_output.close()
    
    w_test_other_output.close()
    w_track_test_other_output.close()

    if final_test_classifier:
        test_other_output = open(with_folder + 'test_other_pairs' + str(idx), 'w')
        track_test_other_output = open(with_folder + 'track_test_other_pairs' + str(idx), 'w')
        other_sentences = open(with_folder + 'other_pairs_sentences' + str(idx), 'w')
        w_test_other_output = open(without_folder +'test_other_pairs' + str(idx), 'w')
        w_track_test_other_output = open(without_folder +'track_test_other_pairs' + str(idx), 'w')

        lines = open(File.replace("nyt", "wpb"), 'r')
        for each_line in lines:
            if each_line[0][0] == '#' or not each_line.strip():
                continue
            words = each_line.split()
            if len(words) <=2:
                continue
            if (words[0] == '<words>'):
                sentence = words
                NERs = []
                dependencies = []
                event_candidate_list = []
                affi_dict = {}
                sentence_count += 1
                continue
            elif (words[0] == '<NERs>'):
                NERs = words
                continue
            elif words[0] == '<dependencies>':
                dependencies = each_line.split('|')
                continue
            elif (words[0] == '<events_index>'):
                word_group_flag = 0
                word_group_member_list = []
                for word in words[1:]:
                    if word == '<':
                        word_group_flag = 1
                        word_group_member_list.append(word)
                        continue
                    if word_group_flag == 1 and word == '>':
                        word_group_flag = 0
                        word_group_member_list.append(word)
                        event_cand = Candidate(gen_flag, sentence, NERs, word_group_member_list)
                        event_candidate_list.append(event_cand)
                        word_group_member_list = []
                        
                        continue
                    if word_group_flag == 1:
                        word_group_member_list.append(word)
                        continue
                for index in range(0, len(event_candidate_list) ):
                    for second_index in range(index+1, len(event_candidate_list) ):
                        event1_index = int( event_candidate_list[index].keyIndex )
                        event2_index = int( event_candidate_list[second_index].keyIndex )
                        event1_key = event_candidate_list[index].keyword
                        event2_key = event_candidate_list[second_index].keyword
                        if abs(event1_index - event2_index) > 10 or abs(event1_index - event2_index) < 2:
                            continue
                        e1 = ""
                        e2 = ""
                        
                        e1 = event_candidate_list[index].string
                        e2 = event_candidate_list[second_index].string
                                
                        #elif e1 + ' ' + e2 in test_dict or e2 + ' ' + e1 in test_dict:
                        if other_sentence_filter(not_events, event1_key, event2_key) == True:
                            continue
                        valid_index, sur_context = get_context_surrounding(sentence, dependencies, context_flag2, event_candidate_list[index], event_candidate_list[second_index] )
                        if sur_context == None:
                            continue
                        exclude_event_words = context_exclude_event_words(sentence, event_candidate_list[index], event_candidate_list[second_index], valid_index)
                        
                        test_other_output.write(sur_context + '\n')
                        w_test_other_output.write(exclude_event_words + '\n')

                        track_test_other_output.write(e1 + ' <==> ' + e2 + '\n')
                        w_track_test_other_output.write(e1 + ' <==> ' + e2 + '\n')

                        other_sentences.write(get_highlight_sentence(sentence, event_candidate_list[index], event_candidate_list[second_index]) + '\n')
        
        lines.close()
        test_other_output.close()
        track_test_other_output.close()
        other_sentences.close()
        w_test_other_output.close()
        w_track_test_other_output.close()
                        
        
    print "sentence_count: ", sentence_count
    print "lines_count: ", lines_count

    
def populate_to_pairs_main(iteration_i, gen_flag, context_flag2, input_f1, input_f2, other_ratio, final_test_classifier):
    
    
    #for File in glob.glob("../../pre_process_sentence_context/pre_process_nyt_new_*"):
    #    populate(iteration_i, gen_flag, context_flag2, input_f1, input_f2, File, other_ratio)
    

    processV = []
    for idx in range(0, 10):
        File = "../../pre_process_sentence_context/pre_process_nyt_new_" + str(idx)
        processV.append(Process(target = populate, args = (iteration_i, idx, gen_flag, context_flag2, input_f1, input_f2, File, other_ratio, final_test_classifier,)))
    
    for idx in range(0, 10):
        processV[idx].start()
        
    for idx in range(0, 10):
        processV[idx].join()

    # merge files
    with_folder = "with_event_words/" + iteration_i + '/'
    
    train_before_output = open(with_folder + 'train_before_pairs', 'w')
    train_after_output = open(with_folder + 'train_after_pairs', 'w')
    train_other_output = open(with_folder + 'train_other_pairs', 'w')

    track_train_before_output = open(with_folder + 'track_train_before_pairs', 'w')
    track_train_after_output = open(with_folder + 'track_train_after_pairs', 'w')
    track_train_other_output = open(with_folder + 'track_train_other_pairs', 'w')
    
    test_other_output = open(with_folder + 'test_other_pairs', 'w')

    track_test_other_output = open(with_folder + 'track_test_other_pairs', 'w')
    
    other_sentences = open(with_folder + 'other_pairs_sentences', 'w')
    train_after_sentences = open(with_folder + 'train_after_pairs_sentences', 'w')
    train_before_sentences = open(with_folder + 'train_before_pairs_sentences', 'w')
    
    without_folder = "without_event_words/" + iteration_i + '/'
    w_train_before_output = open(without_folder + 'train_before_pairs', 'w')
    w_train_after_output = open(without_folder +'train_after_pairs', 'w')
    w_train_other_output = open(without_folder +'train_other_pairs', 'w')
    
    w_track_train_before_output = open(without_folder +'track_train_before_pairs', 'w')
    w_track_train_after_output = open(without_folder +'track_train_after_pairs', 'w')
    w_track_train_other_output = open(without_folder +'track_train_other_pairs', 'w')
    
    w_test_other_output = open(without_folder +'test_other_pairs', 'w')
    
    w_track_test_other_output = open(without_folder +'track_test_other_pairs', 'w')


    for idx in range(0, 10):
        with_folder = "with_event_words/" + iteration_i + '/'
        f = open(with_folder + 'train_before_pairs' + str(idx), 'r')
        train_before_output.write(f.read())
        f.close()
        f = open(with_folder + 'train_after_pairs' + str(idx), 'r')
        train_after_output.write(f.read())
        f.close()
        f = open(with_folder + 'train_other_pairs' + str(idx), 'r')
        train_other_output.write(f.read())
        f.close()

        f = open(with_folder + 'track_train_before_pairs' + str(idx), 'r')
        track_train_before_output.write(f.read())
        f.close()
        f = open(with_folder + 'track_train_after_pairs' + str(idx), 'r')
        track_train_after_output.write(f.read())
        f.close()
        f = open(with_folder + 'track_train_other_pairs' + str(idx), 'r')
        track_train_other_output.write(f.read())
        f.close()
        

        f = open(with_folder + 'test_other_pairs' + str(idx), 'r')
        test_other_output.write(f.read())
        f.close()
        f = open(with_folder + 'track_test_other_pairs' + str(idx), 'r')
        track_test_other_output.write(f.read())
        f.close()
        
        f = open(with_folder + 'other_pairs_sentences' + str(idx), 'r')
        other_sentences.write(f.read())
        f.close()
        f = open(with_folder + 'train_after_pairs_sentences' + str(idx), 'r')
        train_after_sentences.write(f.read())
        f.close()
        f = open(with_folder + 'train_before_pairs_sentences' + str(idx), 'r')
        train_before_sentences.write(f.read())
        f.close()


        
        without_folder = "without_event_words/" + iteration_i + '/'
        f = open(without_folder + 'train_before_pairs' + str(idx), 'r')
        w_train_before_output.write(f.read())
        f.close()
        f = open(without_folder +'train_after_pairs' + str(idx), 'r')
        w_train_after_output.write(f.read())
        f.close()
        f = open(without_folder +'train_other_pairs' + str(idx), 'r')
        w_train_other_output.write(f.read())
        f.close()
        
        f = open(without_folder +'track_train_before_pairs' + str(idx), 'r')
        w_track_train_before_output.write(f.read())
        f.close()
        f = open(without_folder +'track_train_after_pairs' + str(idx), 'r')
        w_track_train_after_output.write(f.read())
        f.close()
        f = open(without_folder +'track_train_other_pairs' + str(idx), 'r')
        w_track_train_other_output.write(f.read())
        f.close()
        
        f = open(without_folder +'test_other_pairs' + str(idx), 'r')
        w_test_other_output.write(f.read())
        f.close()
        
        f = open(without_folder +'track_test_other_pairs' + str(idx), 'r')
        w_track_test_other_output.write(f.read())
        f.close()
    

    train_before_output.close()
    train_after_output.close()
    train_other_output.close()
    
    track_train_before_output.close()
    track_train_after_output.close()
    track_train_other_output.close()
    
    test_other_output.close()
    track_test_other_output.close()
    
    
    w_train_before_output.close()
    w_train_after_output.close()
    w_train_other_output.close()
    
    w_track_train_before_output.close()
    w_track_train_after_output.close()
    w_track_train_other_output.close()
    
    w_test_other_output.close()
    w_track_test_other_output.close()


    # remove useless files
    for idx in range(0, 10):
        with_folder = "with_event_words/" + iteration_i + '/'
        os.remove(with_folder + 'train_before_pairs' + str(idx))
        os.remove(with_folder + 'train_after_pairs' + str(idx))
        os.remove(with_folder + 'train_other_pairs' + str(idx))

        os.remove(with_folder + 'track_train_before_pairs' + str(idx))
        os.remove(with_folder + 'track_train_after_pairs' + str(idx))
        os.remove(with_folder + 'track_train_other_pairs' + str(idx))

        os.remove(with_folder + 'test_other_pairs' + str(idx))
        os.remove(with_folder + 'track_test_other_pairs' + str(idx))
        
        os.remove(with_folder + 'other_pairs_sentences' + str(idx))
        os.remove(with_folder + 'train_after_pairs_sentences' + str(idx))
        os.remove(with_folder + 'train_before_pairs_sentences' + str(idx))


        
        without_folder = "without_event_words/" + iteration_i + '/'
        os.remove(without_folder + 'train_before_pairs' + str(idx))
        os.remove(without_folder +'train_after_pairs' + str(idx))
        os.remove(without_folder +'train_other_pairs' + str(idx))
        
        os.remove(without_folder +'track_train_before_pairs' + str(idx))
        os.remove(without_folder +'track_train_after_pairs' + str(idx))
        os.remove(without_folder +'track_train_other_pairs' + str(idx))
        
        os.remove(without_folder +'test_other_pairs' + str(idx))
        os.remove(without_folder +'track_test_other_pairs' + str(idx))
    

    print "over!"
