#pylint: skip-file
import sys
import os
from commons import * 

# config file for ROUGE
ROUGE_PATH = ROOT_PATH + "result/lcsts_training/char/lcsts_testing/"
SUMM_PATH = ROUGE_PATH + "summ/"
MODEL_PATH = ROUGE_PATH + "model/"
id2summs = {}
id2model = {}

# for models

f_list = os.listdir(MODEL_PATH)
for f_name in f_list:
    #line.strip('\n').lower()
    fileds = f_name.strip().split(".")
    k = fileds[1]

    if k in id2model:
        id2model[k].append(f_name)
    else:
        id2model[k] = [f_name]

# for result
f_list = os.listdir(SUMM_PATH)
for f_name in f_list:
    fileds = f_name.strip().split(".")
    k = fileds[1]
    id2summs[k] = f_name

if len(id2model) != len(id2summs):
    print "ERROR: in rouge file"
    sys.exit()

# write to config file
rouge_s = "<ROUGE-EVAL version=\"1.0\">"
file_id = 0
for k, v in id2summs.items():
    file_id = k
    rouge_s +=  "\n<EVAL ID=\"" + file_id + "\">" \
            + "\n<PEER-ROOT>" \
            + SUMM_PATH \
            + "\n</PEER-ROOT>" \
            + "\n<MODEL-ROOT>" \
            + "\n" + MODEL_PATH \
            + "\n</MODEL-ROOT>" \
            + "\n<INPUT-FORMAT TYPE=\"SPL\">" \
            + "\n</INPUT-FORMAT>" \
            + "\n<PEERS>" \
            + "\n<P ID=\"C\">" + v + "</P>" \
            + "\n</PEERS>" \
            + "\n<MODELS>"

    models = id2model[k]
    m_ids = ["C"]
    #m_ids = ["C", "D", "E", "F"]
    #if len(models) != 4:
    #    print "ERROR: models"
    #    sys.exit()
    for m_id in xrange(len(models)):
        rouge_s += "\n<M ID=\"" + m_ids[m_id] + "\">" + models[m_id] + "</M>"
    rouge_s += "\n</MODELS>\n</EVAL>"
                    
rouge_s += "\n</ROUGE-EVAL>"

with open(ROUGE_PATH + "myROUGE_Config.xml", "w") as f_rouge:
    f_rouge.write(rouge_s)
