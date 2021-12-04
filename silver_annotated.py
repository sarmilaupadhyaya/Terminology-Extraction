
## this code is to create silver annotation

import params
from nltk.tokenize import sent_tokenize, word_tokenize
import rule_based_extraction as re
import fnmatch
import spacy
import pandas as pd
import utils
from sklearn.model_selection import GroupShuffleSplit

raw_corpus = params.raw_corpus
train_file = params.train_file
test_file = params.test_file
train_file_tfidf = params.train_file_tfidf
test_file_tfidf = params.test_file_tfidf
train_file_pointwise = params.train_file_pointwise
test_file_pointwise = params.test_file_pointwise
filtered_data = params.extracted_terms
filtered_data_tfidf=params.extracted_terms_tfidf
filtered_data_pointwise=params.extracted_terms_pointwise
nlp = spacy.load('en_core_web_sm')


import os
import sys

def _find_files(directory, pattern='*.jpg'):
    """Recursively finds all files matching the pattern."""
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

def read_file(filepath):

    data = open(filepath, "r").read().replace("  ", " ")
    return sent_tokenize(data)

def main(type_filter="tfidf"):
    num = 0 
    files = _find_files(raw_corpus, pattern="*.txt")
    total_docid = []
    total_senid = []
    total_word = []
    total_tag = []
    i = 0
    df = pd.DataFrame([], columns=["doc_id", "sen_id","word","tag"])
    for d,each in enumerate(files):
        sentences = read_file(each)
        num += len(sentences)
        # get token
        if type_filter=="tfidf":
            tokens = re.main(sentences, nlp, filtered_data_tfidf)
            train_filename=train_file_tfidf
            test_filename=test_file_tfidf
        elif type_filter=="pointwise":
            tokens = re.main(sentences, nlp, filtered_data_pointwise)
            train_filename=train_file_pointwise
            test_filename=test_file_pointwise
        else:
            tokens = re.main(sentences, nlp, filtered_data)
            train_filename=train_file
            test_filename=test_file

        assert len(sentences) == len(tokens)
        for sentence, token in zip(sentences, tokens):
            tokenized =  word_tokenize(sentence)
            iobs = utils.convert_iob(tokenized, token)
            for word, tag in iobs:
                total_docid.append(d)
                total_senid.append(i)
                total_word.append(word)
                total_tag.append(tag)
            i += 1
    assert len(set(total_senid)) == num
    df["doc_id"] = pd.Series(total_docid)
    df["sen_id"] = pd.Series(total_senid)
    df["word"] = pd.Series(total_word)
    df["tag"] = pd.Series(total_tag)

    train_inds, test_inds = next(GroupShuffleSplit(test_size=.20, n_splits=2, random_state = 7).split(df, groups=df['sen_id']))

    train = df.iloc[train_inds]
    test = df.iloc[test_inds]
    assert len(set(train["sen_id"].tolist()).intersection(set(test["sen_id"].tolist()))) == 0
    train.to_csv(train_filename, sep="\t")
    test.to_csv(test_filename, sep="\t")
    print(num)
            
            





main()
main("pointwise")
main("normal")
