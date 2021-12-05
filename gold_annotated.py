import re
import json
import params
from nltk.tokenize import word_tokenize
import utils
import pandas as pd
gold_annotation_json=params.gold_annotated

## structure data["annotation"][1]["entities"] and data["annptation"][0], sentence

sentences = []
total_terms = []

f = open(gold_annotation_json, "r")
data = json.load(f)
errors_sentences = []
for each_annotation in data["annotations"]:
    sentence = each_annotation[0]
    entities=each_annotation[1]["entities"]
    terms = []
    words = word_tokenize(sentence)
    for entity in entities:
        start= int(entity[0])
        end = int(entity[1])
        term = sentence[start:end]
        wo_term = sentence.split(term)
        start_index = len(word_tokenize(wo_term[0].strip()))
        end_index = len(words) - len(word_tokenize(wo_term[1].strip()))
        terms.append((start_index, end_index))
        try:
            assert word_tokenize(term) == words[start_index: end_index]
        except:
            print(sentence)
            print(term)
            print("**************")
    total_terms.append(terms)
    sentences.append(sentence)
print(total_terms)
i = 0
df = pd.DataFrame([], columns=["doc_id", "sen_id","word","tag"])
total_docid = []
total_senid = []
total_word = []
total_tag = []
d = 0
for sentence, token in zip(sentences, total_terms):
    tokenized =  word_tokenize(sentence)
    print(token)
    iobs = utils.convert_iob(tokenized, token)
    for word, tag in iobs:
        total_docid.append(d)
        total_senid.append(i)
        total_word.append(word)
        total_tag.append(tag)
    i += 1
df["doc_id"] = pd.Series(total_docid)
df["sen_id"] = pd.Series(total_senid)
df["word"] = pd.Series(total_word)
df["tag"] = pd.Series(total_tag)

df.to_csv("output/gold_annotation.csv")


