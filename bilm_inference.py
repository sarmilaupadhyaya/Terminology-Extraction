import pickle
import dataloader
import models.bilstm_crf as bc
from future.utils import iteritems
import pandas as pd
import params as p
import numpy as np
from sklearn.metrics import classification_report

batch_size = p.batch_size
max_len= p.max_len
word_embedding_size=p.word_emb_size
chkpt_path=p.checkpoint_path
tag2id_path = p.tag2id_path

def main(dframe_test):
    with open(tag2id_path, 'rb') as handle:
        tag2idx = pickle.load(handle)
        idx2tag = {v: k for k, v in iteritems(tag2idx)}
    dframe = pd.read_csv(dframe_test, sep="\t").dropna()
    dl = dataloader.Dataloader(dframe, tag2idx, max_len, batch_size,3).loader()
    bilm = bc.ElmoBilstmCrf(max_len, batch_size, word_embedding_size, 3)
    model = bilm.model
    model.load_weights(chkpt_path) 

    TP = {}
    TN = {}
    FP = {}
    FN = {}
    for tag in tag2idx.keys():
        TP[tag] = 0
        TN[tag] = 0    
        FP[tag] = 0    
        FN[tag] = 0    
    p = model.predict_generator(dl) 
    y_test = dl.y[:len(p)]
    print(y_test.shape)
    #return y_test
    print(classification_report(np.argmax(y_test, 2).ravel(), np.argmax(p, axis=2).ravel(),labels=list(idx2tag.keys()), target_names=list(idx2tag.values())))

main(p.test_file)
