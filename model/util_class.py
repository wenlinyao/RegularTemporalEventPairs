class EventPair:
    def __init__(self, line):
        #print line
        self.event1 = "" # entire event representation string
        self.event2 = ""
        self.event1_key = "" # event trigger
        self.event2_key = ""
        event_key_list = []
        self.relation = ""
        self.freq = -1
        words = line.split()
        relation_index = 0
        end_index = len(words)
        angle_brackets_count = 0
        for i, word in enumerate ( words ) :
            if word == "==>" or word == "<==" or word == "<==>":
                self.relation = word
                relation_index = i
            if word[0] == '[':
                event_key_list.append(word)
            if word in ['<', '>']:
                angle_brackets_count += 1
            if angle_brackets_count == 4 and word.isdigit():
                end_index = i
                break
        self.event1 = " ".join(words[:relation_index])
        self.event2 = " ".join(words[relation_index + 1:end_index])
        
        self.event1_key = event_key_list[0]
        self.event2_key = event_key_list[1]

        self.event1_allwords = words[:relation_index]
        self.event2_allwords = words[relation_index + 1:end_index]

        if words[-1].isdigit():
            self.freq = int(words[-1])

# without generalization
class Candidate:
    def __init__(self, gen_flag, sentence, NERs, ele_list):
        if gen_flag == "nogen":
            s = ""
            self.allIndex = []
            self.keyword = ''
            symbol_list = []
            for ele in ele_list:
                if ele[0] == '[':
                    self.keyIndex = ele.replace("[", "").replace("]", "")
                    if int (self.keyIndex) >= len(sentence):
                        symbol_list.append( '[None]' )
                        self.allIndex.append(len(sentence) - 1)
                    else:
                        if int(self.keyIndex) >= len(sentence):
                            continue
                        symbol_list.append( '[' + sentence[int(self.keyIndex)].lower() + ']' )
                        self.keyword = '[' + sentence[int(self.keyIndex)].lower() + ']'
                        self.allIndex.append(self.keyIndex)
                elif not ele.isdigit():
                    symbol_list.append( ele )
                else:
                    if int(ele) >= len(sentence):
                        continue
                    s = sentence[int(ele)].lower()
                    symbol_list.append( s )
                    self.allIndex.append(ele)
            self.string = " ".join(symbol_list)
            self.length = len(ele_list)
        elif gen_flag == "gen":
            #symbol = ['<', '>', 'of', 'be']
            person_pronoun = ["i", "you", "he", "she", "we", "they"]
            number_words = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
            s = ""
            self.allIndex = []
            self.keyword = ''
            symbol_list = []
            for ele in ele_list:
                if ele[0] == '[':
                    self.keyIndex = ele.replace("[", "").replace("]", "")
                    if int (self.keyIndex) >= len(sentence):
                        symbol_list.append( '[None]' )
                        self.allIndex.append(len(sentence) - 1)
                    else:
                        if int(self.keyIndex) >= len(sentence):
                            continue
                        
                        symbol_list.append( '[' + sentence[int(self.keyIndex)].lower() + ']' )
                        self.keyword = '[' + sentence[int(self.keyIndex)].lower() + ']'
                        self.allIndex.append(self.keyIndex)
                elif not ele.isdigit():
                    symbol_list.append( ele )
                else:
                    if int(ele) >= len(sentence):
                        continue
                    if sentence[int(ele)].lower() in person_pronoun:
                        s = "PERSON"
                    elif sentence[int(ele)].lower() in number_words:
                        s = "NUMBER"
                    elif sentence[int(ele)][0].isupper() and NERs[int(ele)] != 'O':
                        s = NERs[int(ele)]
                    else:
                        s = sentence[int(ele)].lower()
                    symbol_list.append( s )
                    self.allIndex.append(ele)
            self.string = " ".join(symbol_list)
            self.length = len(ele_list)
"""
class Candidate:
    def __init__(self, sentence, NERs, ele_list):
        #symbol = ['<', '>', 'of', 'be']
        person_pronoun = ["i", "you", "he", "she", "we", "they"]
        s = ""
        self.allIndex = []
        self.keyword = ''
        symbol_list = []
        for ele in ele_list:
            if ele[0] == '[':
                self.keyIndex = ele.replace("[", "").replace("]", "")
                if int (self.keyIndex) >= len(sentence):
                    symbol_list.append( '[None]' )
                    self.allIndex.append(len(sentence) - 1)
                else:
                    if int(self.keyIndex) >= len(sentence):
                        continue
                    
                    symbol_list.append( '[' + sentence[int(self.keyIndex)].lower() + ']' )
                    self.keyword = '[' + sentence[int(self.keyIndex)].lower() + ']'
                    self.allIndex.append(self.keyIndex)
            elif not ele.isdigit():
                symbol_list.append( ele )
            else:
                if int(ele) >= len(sentence):
                    continue
                if sentence[int(ele)].lower() in person_pronoun:
                    s = "PERSON"
                elif sentence[int(ele)][0].isupper() and NERs[int(ele)] != 'O':
                    s = NERs[int(ele)]
                else:
                    s = sentence[int(ele)].lower()
                symbol_list.append( s )
                self.allIndex.append(ele)
        self.string = " ".join(symbol_list)
        self.length = len(ele_list)
"""

def not_events_load(file):
    f = open(file, 'r')
    
    not_events = set()
    lines = f.readlines()
    
    for each_line in lines:
        words = each_line.split()
        if len(words)<1 or words[0] == '#':
            continue
        else:
            not_events.add('[' + words[0] + ']')
    f.close()
    return not_events

def event_pair_filter(line, freq_threshold, not_events, not_pairs):

    words = line.split()
    if int(words[-1]) < freq_threshold: # 5 time filter rule only apply to above 12
        return True
    eventpair = EventPair(line)
    #print line
    
    instance = eventpair.event1 + ' ' + eventpair.relation + ' ' + eventpair.event2
    if (eventpair.event1_key.lower() in not_events or eventpair.event2_key.lower() in not_events) or (instance in not_pairs)\
    or eventpair.event1_key.lower() == eventpair.event2_key.lower(): # four conditions to disregard
        return True

    # exclude all pairs which include "percent"
    # e.g. < [rose] percent > <== < [falling] percent > 390, < [fell] percent > <== < [rising] percent > 318, < [increased] percent > <== < [rising] percent > 140
    if "percent" in eventpair.event2 or "percent" in eventpair.event1:
        return True

    # exclude all pairs which include "$"
    # e.g. < [rose] $ > <== < [rising] > 32, < [fell] $ > <== < [falling] > 27
    if "$" in eventpair.event2 and "$" in eventpair.event1:
        return True

    if "MISC" in eventpair.event1 or "MISC" in eventpair.event2:
        return True

    if "what" in eventpair.event1 or "what" in eventpair.event2:
        return True

    # event1.key in event2 or event2.key in event1
    if (eventpair.event1_key.replace("[", "").replace("]", "") in eventpair.event2 \
    or eventpair.event2_key.replace("[", "").replace("]", "") in eventpair.event1):
        return True
    if len(eventpair.event1_allwords) <= 3 and len(eventpair.event2_allwords) <= 3:
        return True

    

    return False

def other_sentence_filter(not_events, event1_key, event2_key): # filter some sentences
    
    if event1_key in not_events or event2_key in not_events:
        return True
    return False

    