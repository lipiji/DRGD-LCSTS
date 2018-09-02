# -*- coding: utf-8 -*-
#pylint: skip-file
import sys
import os
import re
import string
from nltk import tokenize

def extract_chinese_clause(doc_str, num_split):
    sentences = []
    if doc_str != u"":
        tmp = re.split(ur"[。！？]+", doc_str)
        for e in tmp:
            if e != u"":
                sentences.append(e)

    len_doc = 0
    clauses = []
    for doc in sentences:
        if doc != u"":
            tmp = re.split(ur"[，；]+", doc)
            for e in tmp:
                len_doc += len(e)
            clauses += tmp
    
    len_clause = len_doc / num_split
    result = []
    s = []
    len_s = 0

    for cl in clauses[:-1]:
        if len_s > len_clause:
            result.append(u"".join(s))
            s = []
            len_s = 0
        s.append(cl)
        len_s += len(cl)
    if len_s != 0:
        result.append(u"".join(s))
    if len(result) < num_split:
        result.append(clauses[-1])
    else:
        result[-1] += clauses[-1]

    return result

def read_news(src_path, has_y, is_unicode):
    abstract = ""
    contents = []

    with open(src_path, "r") as f_src:
        if has_y:
            if is_unicode:
                abstract = f_src.readline().strip().decode("utf-8")
            else:
                abstract = f_src.readline().strip()
        while 1:
            line = f_src.readline()
            if line == "\n":
                break
            if is_unicode:
                line = line.strip().decode("utf-8")
            else:
                line = line.strip()
            contents.append(line)

    return (abstract, contents)

def read_info(src_path, has_abstract, is_unicode):
    abstract = None
    with open(src_path, "r") as f_src:
        if has_abstract:
            line = f_src.readline()
            if not line:
                return None
            abstract = line.decode("utf-8").split() if is_unicode else line.split()

        contents = []
        '''
        if has_feature:
            sim = []
            cw = []
            en = []
            pos = []
        '''
        while 1:
            line = f_src.readline()
            if not line:
                break
            line = line.strip('\n')
            if line == "":
                continue
            if is_unicode:
                line = line.decode("utf-8")
    
            contents.append(line.split())
            '''
            if has_feature:
                sim.append(float(f_src.readline().strip()))
                cw.append(float(f_src.readline().strip()))
                en.append(float(f_src.readline().strip()))
                pos.append(int(f_src.readline().strip()))
            '''

    #feature = (sim, cw, en, pos) if has_feature else None

    return abstract, contents#, feature

class washer(object):
    def __init__(self, stopwords_path, is_unicode, x_num_words, y_num_words, x_num_sents):
        self.stopwords = set()
        self.is_unicode = is_unicode
        if stopwords_path != None:
            with open(stopwords_path, "r") as f_stop:
                for line in f_stop:
                    line = line.strip()
                    if is_unicode:
                        line = line.decode("utf-8")
                    self.stopwords.add(line)
        
        self.white_str = u"" if is_unicode else ""
        self.space = ur"[\s]+"
        if is_unicode:
            self.chinese_punc = ur"[【】、·：『』「」“”《》……￥#（）‘’]+"
            self.punc_table = dict((ord(char), None) for char in string.punctuation)
            self.digit_table = dict((ord(char), None) for char in string.digits)
        else:
            self.punc_table = string.maketrans("", "")

        self.X_MIN_NUM_WORDS, self.X_MAX_NUM_WORDS = x_num_words
        self.Y_MIN_NUM_WORDS, self.Y_MAX_NUM_WORDS = y_num_words
        self.MIN_NUM_SENTS, self.MAX_NUM_SENTS = x_num_sents


    def wash_word(self, word, delete_stopwords):
        word = re.sub(self.space, self.white_str, word) # delete white space
        if self.is_unicode:
            word = word.translate(self.punc_table)
            word = word.translate(self.digit_table) # delete numbers
            word = re.sub(self.chinese_punc, self.white_str, word)
        else:
            word = word.translate(self.punc_table, string.punctuation) # delete punctuations
            word = word.translate(None, string.digits) # delete numbers
        if delete_stopwords:
            if word in self.stopwords:
                return None
        if word == self.white_str:
            return None
    
        return word

    # sent should be a list
    def wash_abstract(self, sent):
        if len(sent) < self.Y_MIN_NUM_WORDS or len(sent) > self.Y_MAX_NUM_WORDS:
            return None
        
        return sent

    # sent should be a list
    def wash_sent(self, sent):
        if len(sent) < self.X_MIN_NUM_WORDS or len(sent) > self.X_MAX_NUM_WORDS:
            return None
        
        return sent

    def wash_news(self, abstract, contents, delete_stopwords):
        wash_status = 0
        if abstract != None:
            new_abstract = []
            tmp = abstract.lower().split()
            for e in tmp:
                e = self.wash_word(e, delete_stopwords)
                if e != None:
                    new_abstract.append(e)
            new_abstract = self.wash_abstract(new_abstract)
            if new_abstract == None:
                wash_status |= 1
        else:
            new_abstract = None
    
        new_contents = []
        for s in contents:
            new_s = []
            tmp = s.lower().split()
            for e in tmp:
                e = self.wash_word(e, delete_stopwords)
                if e != None:
                    new_s.append(e)
            new_s = self.wash_sent(new_s)
            new_contents.append(new_s)
    
        num_contents = 0
        for e in new_contents:
            if e != None:
                num_contents += 1
                if num_contents > self.MAX_NUM_SENTS:
                    break

        if num_contents < self.MIN_NUM_SENTS:
            new_contents = None
            wash_status |= 2
        if num_contents > self.MAX_NUM_SENTS: # save first MAX_NUM_SENTS sentences
            stop = 0
            i = 0
            while stop < self.MAX_NUM_SENTS:
                if new_contents[i] != None:
                    stop += 1
                i += 1
            new_contents = new_contents[0 : stop]
            
        return (new_abstract, new_contents, wash_status)

def interpolate_space(words, is_unicode):
    space = u" " if is_unicode else " "
    return space.join(words)

def remove_space(string, is_unicode):
    empty = u"" if is_unicode else " "
    return empty.join(string.split())

# split a paragraph into sentences (only for English)
def split_paragraph(sent):
    return tokenize.sent_tokenize(sent)
