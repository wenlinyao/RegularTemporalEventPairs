# this program is going to find verb word's particle and object to make each event pair more precise
from StringIO import StringIO
from sets import Set
import glob
import timeit
import gc
import gzip
import re
 
#----------------------------------------------------------------------

class WordToken:
    def __init__ (self, id, word, lemma, POS, NER):
        self.id = id
        self.word = word
        self.lemma = lemma
        #self.CharacterBegin = CharacterBegin
        #self.CharacterEnd = CharacterEnd
        self.POS = POS
        self.NER = NER

class BasicDependency:
    def __init__ (self, type, gov, dep):
        self.type = type
        self.gov = gov
        self.dep = dep
        
class CollapsedDependency:
    def __init__ (self, type, gov, dep):
        self.type = type
        self.gov = gov
        self.dep = dep
"""   
def translate_NER(WordToken):
    person_pronoun = ["i", "you", "he", "she", "we", "they"]
    if WordToken.word.lower() in person_pronoun:
        return "PERSON"
    elif WordToken.word[0].isupper() and WordToken.NER != 'O':
        return WordToken.NER
    else:
        return WordToken.word
"""
def translate_NER(WordToken):
    return WordToken.word


def get_particle_obj(word_id, tokenList, collapsed_dependenciesList):
    index_result = []
    words_result = []
    temp_index_result = []
    temp_words_result = []
    noun_flag = 0
    obj_flag = 0
    words_result.append( '[' + tokenList[int(word_id) - 1].word + ']')
    index_result.append('[' + tokenList[int(word_id) - 1].id + ']')
    
    key_dependency_type_dict = {} # map dependency type to its dependent word
    for element in collapsed_dependenciesList:
        if int(element.gov) == int(word_id):
            key_dependency_type_dict[element.type] = element.dep
    if tokenList[int(word_id) - 1].POS[0] =='N':
        if 'prep_of' in key_dependency_type_dict:
            dep = key_dependency_type_dict['prep_of']
            translated = translate_NER (tokenList[int(dep) - 1])
            words_result.append('of')
            index_result.append('of')
            words_result.append(translated)
            index_result.append(tokenList[int(dep) - 1].id)
        elif 'prep_by' in key_dependency_type_dict:
            dep = key_dependency_type_dict['prep_by']
            translated = translate_NER (tokenList[int(dep) - 1])
            words_result.append('by')
            index_result.append('by')
            words_result.append(translated)
            index_result.append(tokenList[int(dep) - 1].id)
        else:
            prep_other_flag = 0
            prep_other_type = ""
            dep = 0
            for ele in key_dependency_type_dict:
                if 'prep_' in ele:
                    prep_other_flag = 1
                    prep_other_type = ele.replace("prep_", "")
                    dep = key_dependency_type_dict[ele]
            if prep_other_flag == 1 and prep_other_type not in ["after", "before"]:
                translated = translate_NER (tokenList[int(dep) - 1])
                words_result.append(prep_other_type)
                index_result.append(prep_other_type)
                words_result.append(translated)
                index_result.append(tokenList[int(dep) - 1].id)
        
    elif tokenList[int(word_id) - 1].POS[0] =='V':
        if 'prt' in key_dependency_type_dict:
            dep = key_dependency_type_dict['prt']
            words_result.append(tokenList[int(dep) - 1].word)
            index_result.append(tokenList[int(dep) - 1].id)
        if 'dobj' in key_dependency_type_dict:
            dep = key_dependency_type_dict['dobj']
            translated = translate_NER (tokenList[int(dep) - 1])
            words_result.append(translated)
            index_result.append(tokenList[int(dep) - 1].id)
        elif 'nsubjpass' in key_dependency_type_dict:
            dep = key_dependency_type_dict['nsubjpass']
            translated = translate_NER (tokenList[int(dep) - 1])
            words_result.insert(0, 'be') # this order is for the convenience of words_result.insert()
            index_result.insert(0, 'be')
            words_result.insert(0, translated)
            index_result.insert(0, tokenList[int(dep) - 1].id)
        elif 'xcomp' in key_dependency_type_dict:
            dep = key_dependency_type_dict['xcomp']
            translated = translate_NER (tokenList[int(dep) - 1])
            if tokenList[int(dep) - 2].word == 'to':
                words_result.append('to')
                index_result.append(tokenList[int(dep) - 2].id)
                words_result.append(translated)
                index_result.append(tokenList[ int(dep) - 1].id)
            else:
                words_result.append(translated)
                index_result.append(tokenList[ int(dep) - 1].id)
        elif 'nsubj' in key_dependency_type_dict:
            dep = key_dependency_type_dict['nsubj']
            translated = translate_NER (tokenList[int(dep) - 1])
            words_result.insert(0, translated)
            index_result.insert(0, tokenList[int(dep) - 1].id)
        else:
            prep_other_flag = 0
            prep_other_type = ""
            dep = 0
            for ele in key_dependency_type_dict:
                if 'prep_' in ele:
                    prep_other_flag = 1
                    prep_other_type = ele.replace("prep_", "")
                    dep = key_dependency_type_dict[ele]
            if prep_other_flag == 1 and prep_other_type not in ["after", "before"]:
                translated = translate_NER (tokenList[int(dep) - 1])
                words_result.append(prep_other_type)
                index_result.append(prep_other_type)
                words_result.append(translated)
                index_result.append(tokenList[int(dep) - 1].id)
    return index_result, words_result

def parseXML(xmlFile, count, gen_flag):
    start = timeit.default_timer()
    
    output_file = open('../rank_event_pairs_with_particle_obj/event_pairs_with_particle_obj_' + gen_flag, 'a')
    print count,
    print " ",
    print xmlFile
    print( "# " + str(count) + " " + str(xmlFile) + '\n' )
    f = open(xmlFile, "r")
    
    sentence_flag = False
    tokens_flag = False
    token_flag = False
    collapsed_dependencies_flag = False
    basic_dependencies_flag = False
    basic_dep_flag = False
    collapsed_dep_flag = False
    after_find_flag = False
    before_find_flag = False
    after_id = -1
    before_id = -1
    gov_id = -1
    dep_id = -1
    tokenList = []
    basic_dependenciesList = []
    event1 = ""
    event2 = ""
    dobjDict = {}
    word = ""
    lemma = ""
    POS = ""
    NER = ""
    sentence = ""
    relation_flag = ""
    collapsed_dep_type = ""
    event_sentence = []
    event_sentence_flag = False
    
    for each_line in f:        
        
        words = each_line.split()
        #print words
        if (len(words) == 0):
            continue
        # save sentences information which include event pairs
        if (words[0] == '<DOC'):
            continue
            
        #structure start
        if (words[0] == '<sentence'):
            after_find_flag = False
            before_find_flag = False
            after_id = -1
            before_id = -1
            tokenList = []
            collapsed_dependenciesList = []
            event1 = ""
            event2 = ""
            dobjDict = {}
            relation_flag = ""
            sentence_flag = True #sentences structure start
            continue # process next line
        if (sentence_flag == True and words[0] == '<tokens>'):
            tokens_flag = True #tokens structure start
            continue
        if (tokens_flag == True and words[0] == '<token' and len(words) >= 2):
            token_flag = True
            token_id = int (words[1].replace("id=\"", "").replace("\">", ""))
            continue
        
        if (sentence_flag == True and words[0] == '<collapsed-ccprocessed-dependencies>'):
            collapsed_dependencies_flag = True
            continue
        if (collapsed_dependencies_flag == True and words[0] == '<dep' and len(words) >= 2):
            collapsed_dep_flag = True
            collapsed_dep_type = words[1].replace("type=\"", "").replace("\">", "")
            #print collapsed_dep_type
            continue
        
        if (collapsed_dep_flag == True):
            if (words[0].find('<governor>') != -1):
                collapsed_gov = words[0].replace("<governor>", "").replace("</governor>", "")
                #print collapsed_gov
                #raw_input("continue?")
                continue
            if (words[0].find('<dependent>') != -1):
                collapsed_dep = words[0].replace("<dependent>", "").replace("</dependent>", "")
                #print collapsed_dep
                #raw_input("continue?")
                continue
            
        #structure end
        if (token_flag == True and words[0] == '</token>'):
            # reminder: token list start with index 0, but token id start with 1
            tokenList.append(WordToken(str(token_id), word, lemma, POS, NER))
            token_flag = False
            continue
        if (tokens_flag == True and words[0] == '</tokens>'):
            tokens_flag = False
            continue
        if (sentence_flag == True and words[0] == '</sentence>'):
            """
            if before_find_flag == True or after_find_flag == True:
                counter = 0
                for each_element in tokenList:
                    counter += 1
                    print '[', counter,': ', each_element.word, ']'
                print '\n'
                raw_input("continue?")
            """
            after_id = -1
            before_id = -1
            after_find_flag = False
            before_find_flag = False
            tokenList = []
            basic_dependenciesList = []
            dobjDict = {}
            sentence_flag = False
            continue
        if (collapsed_dependencies_flag == True and words[0] == '</collapsed-ccprocessed-dependencies>'):
            for each_element in collapsed_dependenciesList:
                event1 = ""
                event2 = ""
                temp_event1 = ""
                temp_event2 = ""
                
                if each_element.type == 'prep_after' or each_element.type == 'prepc_after':
                    #obj1_flag = 0 # object flag
                    #obj2_flag = 0
                    gov_id = int(each_element.gov)
                    dep_id = int(each_element.dep)
                    index1, words1 = get_particle_obj(gov_id, tokenList, collapsed_dependenciesList)
                    index2, words2 = get_particle_obj(dep_id, tokenList, collapsed_dependenciesList)
                    relation_flag = "<=="
                    event1 = ' < ' + " ".join(words1) + ' > '
                    event2 = ' < ' + " ".join(words2) + ' > '
                    output_file.write(event1 + relation_flag + event2 + '\n')
                    
                event1 = ""
                event2 = ""
                temp_event1 = ""
                temp_event2 = ""
                if each_element.type == 'prep_before' or each_element.type == 'prepc_before':
                    gov_id = int(each_element.gov)
                    dep_id = int(each_element.dep)
                    index1, words1 = get_particle_obj(gov_id, tokenList, collapsed_dependenciesList)
                    index2, words2 = get_particle_obj(dep_id, tokenList, collapsed_dependenciesList)
                    relation_flag = "==>"
                    event1 = ' < ' + " ".join(words1) + ' > '
                    event2 = ' < ' + " ".join(words2) + ' > '
                    output_file.write(event1 + relation_flag + event2 + '\n')
            
            collapsed_dependencies_flag = False
            continue
        if (collapsed_dep_flag == True and words[0] == '</dep>'):
            collapsed_dependenciesList.append(CollapsedDependency(collapsed_dep_type, collapsed_gov, collapsed_dep))
            # find the direct object of selected events
            """
            if (collapsed_dep_type == 'dobj'):
                dobjDict[collapsed_gov] = collapsed_dep
                #print dobjDict
                #raw_input("continue?")
                #print basic_gov,
                #print basic_dep
                continue
            """
            collapsed_dep_type = ""
            collapsed_dep_flag = False
            continue
        if (token_flag == True):
            if (words[0].find('<word>') != -1):
                word = words[0].replace("<word>", "").replace("</word>", "")
                continue
            if (words[0].find('<lemma>') != -1):
                lemma = words[0].replace("<lemma>", "").replace("</lemma>", "")
                #sentence_dic[token_id] = word
                if (lemma == 'after'):
                    after_find_flag = True
                    after_id = token_id
                if (lemma == 'before'):
                    before_find_flag = True
                    before_id = token_id
                continue
            if (words[0].find('<POS>') != -1):
                POS = words[0].replace("<POS>", "").replace("</POS>", "")
                continue
            if (words[0].find('<NER>') != -1):
                NER = words[0].replace("<NER>", "").replace("</NER>", "")
                continue
        
        """
        print collapsed_dependencies_flag
        print collapsed_dep_flag
        print collapsed_dep_type
        
        raw_input("continue?")
        """
        
        
    f.close()
    output_file.close()
    stop = timeit.default_timer()
    print stop-start
    
def event_pairs_with_particle_obj_main(gen_flag):
    #valid_pairs_set = valid_event_pairs_set()
    count = 1
    open('../rank_event_pairs_with_particle_obj/event_pairs_with_particle_obj_' + gen_flag, 'w').close()
    for xmlFile in glob.glob("../../event_pairs_sentences_result/event_pairs_sentences_result_*"):
        
        parseXML(xmlFile, count, gen_flag)
        gc.collect()
        count = count +1 
        
        
    print "over!"
    
