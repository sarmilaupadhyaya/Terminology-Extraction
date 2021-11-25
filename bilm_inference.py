import pickle

batch_size = p.batch_size
max_len= p.max_len
word_embedding_size=p.word_emb_size
chkpt_path=p.checkpoint_path
tag2id_path = p.tag2id_path


if __name__=="__main__":

    doc_path = sys.argv[1]

    with open(tag2id_path, 'rb') as handle:
        tag2idx = pickle.load(handle)

