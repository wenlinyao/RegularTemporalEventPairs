from StringIO import StringIO
from sets import Set
import glob
import timeit
import gc
import gzip
from random import randint
import random
import networkx as nx

def get_context_surrounding(sentence, dependencies, event_cand1, event_cand2):# relation_type == ' ==> ' or ' <== ' or ' <==> '
    num_words_in_between = 10
    num_words_before_or_after = 5
    valid_index = []
    event1_index = event_cand1
    event2_index = event_cand2
    sur_context = ""
    G = nx.Graph()
    for entity in dependencies[1:-1]:
    	print entity
    	words = entity.split()
        head = words[0]
        tail = words[2]
        edge = (head, tail)
        G.add_edge(*edge)
    path = nx.shortest_path(G, source = event1_index, target = event2_index)
    print path
    raw_input("continue?")

File = "../../pre_process_sentence_context/pre_process_nyt_new_5"

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

        continue
    elif (words[0] == '<NERs>'):
        NERs = words
        continue
    elif words[0] == '<dependencies>':
        dependencies = each_line.split('|')
        continue
    elif (words[0] == '<events_index>'):
        get_context_surrounding(sentence, dependencies, "1", "5")

