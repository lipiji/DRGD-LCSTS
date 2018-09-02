# -*- coding: utf-8 -*-
from utils_preprocess import *
from utils_pg import rebuild_dir
from commons import *

# return unicode result
def read_lcsts_xml(xml_path, shape_cls):
    raw_news = []

    i = 0
    with open(xml_path, "r") as f_src:
        while 1:
            abstract = u""
            doc = u""
            l = f_src.readline()
            if l == "":
                break
            l = l.decode("utf-8")

            if l.startswith(u"<doc"):
                file_id = l.split(u"=")[1][:-2]

                l = f_src.readline().strip().decode("utf-8")
                if l.startswith(u"<human_label>"):
                    score = int(l[13])
                    l = f_src.readline().strip().decode("utf-8")
                else:
                    score = None

                if l == u"<summary>":
                    while 1:
                        l = f_src.readline().strip().decode("utf-8")
                        if l == u"</summary>":
                            break
                        abstract += l
                else:
                    raise RuntimeError("summary_error", file_id, l)
                l = f_src.readline().strip().decode("utf-8")
                if l == u"<short_text>":
                    while 1:
                        l = f_src.readline().strip().decode("utf-8")
                        if l == u"</short_text>":
                            break
                        doc += l
                else:
                    raise RuntimeError("short_text error", file_id, l)
                l = f_src.readline().strip().decode("utf-8")
                if l != u"</doc>":
                    raise RuntimeError("doc end error", file_id, l)
            else:
                raise RuntimeError("doc begin error", file_id, l)
        
            if score == None or score >= 3:
                contents = extract_chinese_clause(doc, shape_cls.MAX_NUM_X)
                raw_news.append((abstract, contents))

            i += 1
            if i % 50000 == 0:
                print i, "raw news have been read"
    return raw_news

def split2char(news, is_unicode):
    abstract, contents = news

    new_abstract = interpolate_space(abstract, is_unicode)

    new_contents = []
    for content in contents:
        new_contents.append(interpolate_space(content, is_unicode))

    return (new_abstract, new_contents)

def write_one_news(f_raw, f_trimmed, abstract, new_abstract, contents, new_contents):
    new_line = u"\n".encode("utf-8")
    if abstract != None:
        f_raw.write(abstract.encode("utf-8"))
        f_raw.write(new_line)
    if new_abstract != None:
        f_trimmed.write(interpolate_space(new_abstract, True).encode("utf-8"))
        f_trimmed.write(new_line)

    for i in xrange(len(new_contents)):
        if new_contents[i] != None:
            f_raw.write((contents[i]).encode("utf-8"))
            f_raw.write(new_line)
            f_trimmed.write(interpolate_space(new_contents[i], True).encode("utf-8"))
            f_trimmed.write(new_line)

    f_raw.write(new_line)
    f_trimmed.write(new_line)


def transform_raw_data(src_path, raw_dst_root, trimmed_dst_root, read_func, shape_cls, has_y):
    print "src_path:", src_path
    rebuild_dir(raw_dst_root)
    rebuild_dir(trimmed_dst_root)

    print "start reading raw news"
    raw_news = read_func(src_path, shape_cls)
    print "finish reading raw news, length:", len(raw_news)

    all_status = [0, 0, 0, 0, 0]
    ws = washer(None, True,
            (shape_cls.MIN_LEN_X, shape_cls.MAX_LEN_X), (shape_cls.MIN_LEN_Y, shape_cls.MAX_LEN_Y),
            (shape_cls.MIN_NUM_X, shape_cls.MAX_NUM_X))

    clean_news = []
    i = 0
    for news in raw_news:
        if news[0] == u"" or len(news[1]) == 0:
            all_status[4] += 1
            continue
        abstract, contents = split2char(news, True)

        new_abstract, new_contents, status = ws.wash_news(abstract, contents, False)
        all_status[status] += 1
        if has_y:
            if new_abstract == None or new_contents == None:
                continue
            clean_news.append((abstract, new_abstract, contents, new_contents))
        else:
            if new_contents == None:
                continue
            clean_news.append((None, None, contents, new_contents))
        i += 1
        if i % 50000 == 0:
            print i, "files have been washed"
            print "unqualified absrtact in washing:", all_status[1]
            print "unqualified contents in washing:", all_status[2]
            print "unqualified abstracts and contents in washing:", all_status[3]
            print "unqualified files before washing:", all_status[4]


    print len(clean_news), "files after washing"
    print "unqualified absrtact in washing:", all_status[1]
    print "unqualified contents in washing:", all_status[2]
    print "unqualified abstracts and contents in washing:", all_status[3]
    print "unqualified files before washing:", all_status[4]

    idx_file = 0
    for news in clean_news:
        with open(raw_dst_root + str(idx_file), "w") as f_raw:
            with open(trimmed_dst_root + str(idx_file), "w") as f_trimmed:
                abstract, new_abstract, contents, new_contents = news
                write_one_news(f_raw, f_trimmed, abstract, new_abstract, contents, new_contents)
                idx_file += 1
                if idx_file % 50000 == 0:
                    print idx_file, "files have been processed"

def transform_lcsts_datasets(lcsts_path, shape_cls):
    read_func = read_lcsts_xml

    print "start transforming lcsts training"
    transform_raw_data(lcsts_path + "PART_I.txt", ROOT_PATH + "training_data/lcsts/raw/", ROOT_PATH + "training_data/lcsts/char_trimmed/", read_func, shape_cls, True)
    print "\nstart transforming lcsts testing"
    transform_raw_data(lcsts_path + "PART_III.txt", ROOT_PATH + "testing_data/lcsts/raw/", ROOT_PATH + "testing_data/lcsts/char_trimmed/", read_func, shape_cls, True)
   
    print "\nstart transforming lcsts validation"
    transform_raw_data(lcsts_path + "PART_II.txt", ROOT_PATH + "validation_data/lcsts/raw/", ROOT_PATH + "validation_data/lcsts/char_trimmed/", read_func, shape_cls, True)
   

    print
