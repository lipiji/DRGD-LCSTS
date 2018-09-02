# -*- coding: utf-8 -*-

import os
from utils_preprocess import read_news
from utils_pg import rebuild_dir
from commons import *

def write_news_info(dst_path, raw_info, features, is_unicode):
    abstract, all_news = raw_info
    if features != None:
        similarities, concept_weights, num_entities, positions = features

    with open(dst_path, "w") as f_dst:
        if is_unicode:
            f_dst.write(abstract.encode('utf-8'))
            f_dst.write("\n")
        else:
            f_dst.write(abstract + "\n")
        if features != None:
            for news, sim, cw, en, pos in zip(all_news, similarities, concept_weights, num_entities, positions):
                if is_unicode:
                    f_dst.write(news.encode("utf-8"))
                    f_dst.write("\n")
                else:
                    f_dst.write(news + "\n")
                    f_dst.write(str(sim) + "\n")
                    f_dst.write(str(cw) + "\n")
                    f_dst.write(str(en) + "\n")
                    f_dst.write(str(pos) + "\n")
        else:
            for news in all_news:
                if is_unicode:
                    f_dst.write(news.encode("utf-8"))
                    f_dst.write("\n")
                else:
                    f_dst.write(news + "\n")
        f_dst.write("\n")

def write_news(src_path, dst_path, has_y):
    print "process files under", src_path
    rebuild_dir(dst_path)
    fnames = os.listdir(src_path)
    fnames = sorted(fnames)
    print len(fnames), "files to be processed"

    i = 0
    for fname in fnames:
        try:
            raw_info = read_news(src_path + fname, has_y, True)
        except (Exception) as e:
            print "READING: exception occurs in fname:", fname
            print e
            continue
    
        write_news_info(dst_path + fname, raw_info, None, True)
        i += 1
        if i % 50000 == 0:
            print i, "files have been written"
