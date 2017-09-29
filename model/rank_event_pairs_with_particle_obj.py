from sets import Set
import glob
import timeit
import operator


def rank_event_pairs_with_particle_obj_main(gen_flag):
    start = timeit.default_timer()
    words_dic = {}
    output_file = open('../rank_event_pairs_with_particle_obj/rank_event_pairs_with_particle_obj_' + gen_flag, 'w')
    # participate part
    for File in glob.glob("../rank_event_pairs_with_particle_obj/event_pairs_with_particle_obj_" + gen_flag):
        print ("processing " + str(File) + " ...")
        f = open(File, "r")
        lines = f.readlines()
        
        for each_line in lines:
            if each_line[0] == '#' or not each_line.strip():
                continue
            to_lower = []
            words = each_line.split()
            for word in words:
                if word[0] == '[':
                    to_lower.append( word.lower() )
                else:
                    to_lower.append( word )
            instance = ' '.join(to_lower)
            if not instance in words_dic:
                words_dic[instance] = 1
            else:
                words_dic[instance] = words_dic[instance] + 1
    
    print ("sorting...")
    words_freq_sort = sorted(words_dic.items(), key = operator.itemgetter(1))
    words_freq_sort.reverse()
    
    for item in words_freq_sort:
        output_file.write(str(item[0]) + ' ' + str(item[1]) + '\n')
        
    stop = timeit.default_timer()
    print( "# over! time consuming:"),
    print(str(stop - start) )
    f.close
    output_file.close()