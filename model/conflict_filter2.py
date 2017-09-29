# This program is intended to filter event pairs instances with conflict temporal relation
#
from StringIO import StringIO
from sets import Set
import glob
import timeit
import gc
import gzip
import re
from util_class import *


# ratio: n times to another one means it is dominant
# event pair freq threshold control how many bottom pairs will be removed
def conflict_filter_main(input_f, output_f, ratio, freq_threshold): 
#if __name__ == "__main__":
#    input_f = 'rank_event_pairs_with_particle_obj/rank_event_pairs_with_particle_obj'
#    output_f = 'event_pair_train_test/event_pairs_removed_conflict'

    
    print "conflict_filter.py is processing \n", input_f, "\n to \n", output_f
    not_events = not_events_load("../dic/words_filter.txt")
    f = open(input_f, 'r')
    output = open(output_f, 'w')
    event_pairs_freq = {}
    lines = f.readlines()
    not_pairs = set()
    keys_pair = {} # if the key pair is the same, only keep the one with more arguments
    for each_line in lines:
        
        eventpair = EventPair(each_line)
        
        instance = eventpair.event1 + ' ' + eventpair.relation + ' ' + eventpair.event2
        if eventpair.relation == '<==':
            if not instance.replace("<==", "==>") in event_pairs_freq:
                event_pairs_freq[instance] = eventpair.freq
                
            else:
                conflict_freq = event_pairs_freq[instance.replace("<==", "==>")]
                if conflict_freq * ratio < eventpair.freq:
                    not_pairs.add(instance.replace("<==", "==>"))
                elif conflict_freq > eventpair.freq * ratio:
                    not_pairs.add(instance)
                else:
                    not_pairs.add(instance.replace("<==", "==>"))
                    not_pairs.add(instance)
                    
  
        if eventpair.relation == '==>':
            if not instance.replace("==>", "<==") in event_pairs_freq:
                event_pairs_freq[instance] = eventpair.freq

            else:
                #print all_words
                #raw_input("continue?")
                conflict_freq = event_pairs_freq[instance.replace("==>", "<==")]
                if conflict_freq * ratio < eventpair.freq:
                    not_pairs.add(instance.replace("==>", "<=="))
                elif conflict_freq > eventpair.freq * ratio:
                    not_pairs.add(instance)
                else:
                    not_pairs.add(instance.replace("==>", "<=="))
                    not_pairs.add(instance)
                    
    count = 0
    for each_line in lines:

        if event_pair_filter(each_line, freq_threshold, not_events, not_pairs) == True: # util_class.py
            continue
        #print each_line
        #if eventpair.relation == '==>':
        #    continue
        #print eventpair.event1_allwords
        #print eventpair.event2_allwords
        #raw_input("continue?")
        output.write(each_line)
    print "over!"
