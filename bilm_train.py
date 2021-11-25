import sys
import models.bilstm_crf as bc
import params as p
import dataloader
import pandas as pd
from future.utils import iteritems
from math import nan
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

batch_size = p.batch_size
max_len= p.max_len
word_embedding_size=p.word_emb_size
chkpt_path=p.checkpoint_path
tag2id_path = p.tag2id_path

if __name__=="__main__":



    train_path = sys.argv[1]
    test_path = sys.argv[2]

    dframe = pd.read_csv(train_path, sep = "\t", index_col=None, header=None,  encoding = "ISO-8859-1", error_bad_lines=False)
    dframe_test = pd.read_csv(test_path, sep = "\t", index_col=None, header=None,  encoding = "ISO-8859-1", error_bad_lines=False)
    dframe.columns = ["id", "sentenceid","word", "tag"]
    dframe_test.columns = ["id", "sentenceid","word", "tag"]

    words = list(set(dframe["word"].values))
    words.append("ENDPAD")
    n_words = len(words)
    tags = []
    for tag in set(dframe["tag"].values):
        if tag is nan or isinstance(tag, float):
            tags.append('unk')
        else:
            tags.append(tag)


    n_tags = len(tags)
    word2idx = {w: i for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}
    idx2tag = {v: k for k, v in iteritems(tag2idx)}



    ## save these as a pickle file to load during inference
    import pickle
    with open(tag2id_path, 'wb') as handle:
        pickle.dump(tag2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)

    bilm = bc.ElmoBilstmCrf(max_len, batch_size, word_embedding_size, n_tags)
    model = bilm.model

    # Saving the best only
    checkpoint = ModelCheckpoint(chkpt_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    
    dframe_train, dframe_val = train_test_split(dframe, test_size=0.2, random_state=42)

    dl = dataloader.Dataloader(dframe_train, word2idx, tag2idx, max_len, batch_size,n_tags).loader()
    dl_val = dataloader.Dataloader(dframe_val, word2idx, tag2idx, max_len, batch_size, n_tags).loader()
    dl_test = dataloader.Dataloader(dframe_test, word2idx, tag2idx, max_len, batch_size, n_tags).loader()

    model.fit_generator(dl, steps_per_epoch=len(dframe_train)//batch_size,epochs=4, validation_data=dl_val, validation_steps=len(dframe_val)//batch_size, verbose=1, callbacks=callbacks_list)






