import os
import math
import string
import spacy
from nltk.tokenize import word_tokenize
import fnmatch
from spacy.matcher import Matcher
from collections import Counter, defaultdict
import spacy
import pandas as pd
from collections import defaultdict

nlp = spacy.load('en_core_web_lg')
nlp.add_pipe('sentencizer')
matcher = Matcher(nlp.vocab)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering


def read_file(file_path):
    """

    """
    data = open(file_path, "r").read().replace("  ", " ")
    data = data.translate(str.maketrans('', '', string.punctuation))
    string_encode = data.encode("ascii", "ignore")
    data = string_encode.decode()
    data_nlp = nlp(data)
    return data

def validate(terminology):
    for x in terminology.split(" "):
        if len(x) <4:
            return False
    return True

def get_terminology(data, type, nlp=nlp):
    """
    """
    data = nlp(data)
    total = []
    if type == "single-word":
        patterns= [[{"POS":"NOUN"}]]
    else:
        patterns = [[{"POS":"ADJ"},{"POS":"ADJ"},{"POS": "NOUN"}],[{"POS":"ADJ"},{"POS":"ADJ"},{"POS":"ADJ"},{"POS": "NOUN"}],[{"POS":"ADJ"},{"POS":"ADJ"},{"POS":"ADJ"},{"POS":"ADJ"},{"POS":"ADJ"},{"POS": "NOUN"}],[{"POS": "NOUN"}, {"POS": "NOUN"}], [{"POS": "ADJ"}, {"POS": "NOUN"}],[{"POS": "PROPN"}, {"POS": "PROPN"},{"POS": "PROPN"}],[{"POS": "PROPN"}, {"POS": "PROPN"}]]
    for pattern in patterns:
        matcher = Matcher(nlp.vocab)
        matcher.add("nlg", [pattern])

        matches = matcher(data)
        for match_id, start, end in matches:
            string_id = nlp.vocab.strings[match_id]  # Get string representation
            span = data[start:end]  # The matched span
            total.append(span.text)
            
        matcher.remove("nlg")
    return total

def pointwise_mutual_info(vocab_freq, term_freq):

    pointwise_value = defaultdict()
    for each, count  in dict(term_freq).items():
        assert " ".join(each.split(" ")) == each
        obs_fre = count
        exp_fre = 1
        for each_comp in each.split(" "):

            if vocab_freq.get(each_comp) is None:
                pass
            else:
                exp_fre *= vocab_freq.get(each_comp)
                pointwise_value[each] = math.log(obs_fre/exp_fre)
    return pointwise_value


def _find_files(directory, pattern='*.txt'):
        """Recursively finds all files matching the pattern."""
        files = []
        for root, dirnames, filenames in os.walk(directory):
            for filename in fnmatch.filter(filenames, pattern):
                files.append(os.path.join(root, filename))
        return files

def extract_terminology(filepath, type):
    """
    file path: for your all text in a domain

    """
    total_data = []
    files = _find_files(directory=filepath, pattern='*.txt')

    total_words = []
    total_terms = []
    doc_freq = defaultdict(list)
    for each_file in files:
        data= read_file(each_file)
        total_words += word_tokenize(data)
        terminologies = get_terminology(data, type)
        terminologies = [x for x in terminologies if validate(x)]
        for word in set(terminologies):
            if word in doc_freq:
                doc_freq[word] += 1
            else:
                doc_freq[word] = 1
        total_terms.extend(terminologies)

    ## making terms title
    #total_terms = [term.title() for term in total_terms]
    total_terms = Counter(total_terms)
    total_terms.most_common()
    vocab_freq = Counter(total_words) 
    ## Statistical approaches to filterm term
    # Step 1: filter by frequency
    print("Common 20 term from the article are listed below:")
    print(total_terms.most_common()[:20])
    #step 2: Tf-idf
    result = {each: math.log(len(files)/doc_freq.get(each, 0)) * total_terms.get(each, 0) for each, value in doc_freq.items()}
    #result = pointwise_mutual_info(vocab_freq, total_terms)
    result = total_terms.copy()
    result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
    return list(result.keys())[:300]

if __name__ == "__main__":
    terminology_extracted = extract_terminology("data", type="single-word")
    df = pd.DataFrame(terminology_extracted, columns = ["Terminology"])
    df.to_csv("output/extracted_terminology_singleword.csv")
    terminology_extracted2 = extract_terminology("data", type=None)
    print(terminology_extracted2)
    df2 = pd.DataFrame(terminology_extracted2, columns = ["Terminology"])
    df2.to_csv("output/extracted_terminology_multiword.csv")
    df3 = pd.concat([df, df2], axis=0)
    df3 =df3[["Terminology"]]
    df3.to_csv("output/merged.csv")


