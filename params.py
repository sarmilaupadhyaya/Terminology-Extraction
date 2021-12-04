batch_size = 32
max_len=40
word_emb_size=1024
checkpoint_path="output/chkpt/markable-bi-lstm-td-model-pointwise.hdf5"
tag2id_path="output/tag2idx_pointwise.pk"



raw_corpus="data/"
train_file="data/silver_annotated/train.tsv"
test_file="data/silver_annotated/test_pointwise.tsv"
train_file_pointwise="data/silver_annotated/train_pointwise.tsv"
test_file_pointwise="data/silver_annotated/test_pointwise.tsv"
train_file_tfidf="data/silver_annotated/train_tfidf.tsv"
test_file_tfidf="data/silver_annotated/test_tfidf.tsv"
inference_txt="data/inference/inference.txt"
extracted_terms="output/extracted_terminology_multiword.csv"
extracted_terms_tfidf="output/extracted_terms_multi_tfidf.csv"
extracted_terms_pointwise="output/extracted_terms_multi_pointwise.csv"
gold_annotated="data/gold_annotation.json"
gold_annotation="data/gold_annotation.csv"

