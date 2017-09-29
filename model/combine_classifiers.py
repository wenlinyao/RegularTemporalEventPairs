from StringIO import StringIO
from sets import Set
import glob
import timeit
import gc
from random import randint
import re
import operator
import random
from util_class import *

            
def get_pred_prob(true_and_pred_value):
    words = true_and_pred_value.split()

    pred_label = words[1]
    words[-1] = words[-1].replace("]", "")
    
    float_value = [float(word) for word in words[3:6]]
    pred_prob = max(float_value)
    return pred_label, pred_prob

def reverse(relation):
    if relation == "0":
        return "1"
    elif relation == "1":
        return "0"
    else:
        return "2"

def reverse2(relation):
    if relation == "<==":
        return "==>"
    elif relation == "==>":
        return "<=="
    else:
        return "<==>"
        
def combine_classifiers_main(iteration_i, type, after_before_ratio, difference_ratio):
    
    
    # 1_Classifier
    list_of_file_lines = []
    f = open("mr_folder_" + type + "_event_words/" + iteration_i + "/cand_true_and_pred_value0", 'r')
    for each_line in f:
        list_of_file_lines.append(each_line)
    output = open("error_analysis/" + iteration_i + "/1_Classifier_combine_classifiers", 'w')
    
    
    true_pred_storage = {}

    count = 0
    folder = type + "_event_words/" + iteration_i + "/"
    track_list = []

    track_f = open(folder + "track_test_other_pairs", 'r')
    for each_line in track_f:
        track_list.append(each_line)

    discover_count = {}

    instance_length = len(list_of_file_lines)
    
    for i in range(0, instance_length):
        # 0 A <== B
        # 0 A <== B
        # 0 B ==> A
        eventpair = EventPair(track_list[i])
        reverse_flag = False
        if eventpair.event1 <= eventpair.event2:
            pair = eventpair.event1 + ' ' + eventpair.relation + ' ' + eventpair.event2
        else:
            reverse_flag = True
            pair = eventpair.event2 + ' ' + reverse2(eventpair.relation) + ' ' + eventpair.event1
        
        #pair = eventpair.event1 + ' ' + eventpair.relation + ' ' + eventpair.event2
        if not pair in discover_count:
            discover_count[pair] = []
            if reverse_flag == True:
                discover_count[pair].append(reverse(list_of_file_lines[i].split()[1]))
            else:
                discover_count[pair].append(list_of_file_lines[i].split()[1])
        else:
            if reverse_flag == True:
                discover_count[pair].append(reverse(list_of_file_lines[i].split()[1]))
            else:
                discover_count[pair].append(list_of_file_lines[i].split()[1])
        

    total_num_threshold = 10

    discover_freq = {}
    for ele in discover_count:

        after_num = discover_count[ele].count('0')
        before_num = discover_count[ele].count('1')
        other_num = discover_count[ele].count('2')
        total_num = after_num + before_num + other_num
        if total_num < total_num_threshold:
            continue
        if after_num > before_num and after_num + before_num >= total_num * after_before_ratio and after_num - before_num >= total_num * difference_ratio:
            pred_label = "<=="
            #output.write( pred_label + ' ' + ele + ' ' + str (after_num) + ' ' + str(before_num) + ' ' + str(other_num) + '\n')
            discover_freq[pred_label + ' ' + ele + ' ' + str (after_num) + ' ' + str(before_num) + ' ' + str(other_num)] = total_num
        elif before_num > after_num and after_num + before_num >= total_num * after_before_ratio and before_num - after_num >= total_num * difference_ratio:
            pred_label = "==>"
            #output.write( pred_label + ' ' + ele + ' ' + str (after_num) + ' ' + str(before_num) + ' ' + str(other_num) + '\n')
            discover_freq[pred_label + ' ' + ele + ' ' + str (after_num) + ' ' + str(before_num) + ' ' + str(other_num)] = total_num
    print ("sorting...")
    discover_freq_sort = sorted(discover_freq.items(), key = operator.itemgetter(1))
    discover_freq_sort.reverse()
    
    for item in discover_freq_sort:
        output.write(str(item[0]) + ' ' + str(item[1]) + '\n')
    
    print "over!"