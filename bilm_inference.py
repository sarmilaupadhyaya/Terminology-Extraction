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
chkpt_path_tfidf=p.checkpoint_path_tfidf
tag2id_path = p.tag2id_path
tag2id_path_tfidf=p.tag2id_path_tfidf

def main(dframe, filter_type):

    if filter_type == "tfidf":
        tagf2id_path=tag2id_path_tfidf
        chkpt=chkpt_path_tfidf
    else:
        tagf2id_path=tag2id_path
        chkpt=chkpt_path

    with open(tag2id_path, 'rb') as handle:
        tag2idx = pickle.load(handle)
        idx2tag = {v: k for k, v in iteritems(tag2idx)}
    
    dl = dataloader.Dataloader(dframe, tag2idx, max_len, batch_size,3).loader()
    actual = len(dframe["sen_id"].unique())
    bilm = bc.ElmoBilstmCrf(max_len, batch_size, word_embedding_size, 3)
    model = bilm.model
    model.load_weights(chkpt) 
    p = model.predict_generator(dl) 
    total_tags = []
    import pdb
    pdb.set_trace()
    for i, (pred, sentence) in enumerate(zip(p[:actual],dl.x[:actual])):
        gt = np.argmax(p[i], axis=-1)
        for idx, (ep,word) in enumerate(zip(gt,sentence)):
            if word != "__PAD__":
                total_tags.append(idx2tag[gt[idx]])
    return total_tags


#main(p.test_file)
