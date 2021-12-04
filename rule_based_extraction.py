## read manually filtered term and apply rule based system to get term from all corpus
# generate random integer values
from random import seed
from random import randint
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
import spacy
from spacy.matcher import Matcher
import nltk


#get list of manually filtered terms
def read_term_list(filename):
    with open(filename, encoding="utf8") as f:
        lines = f.readlines()
        return [line.split(",")[1].strip("\n").lower() for line in lines]

#convert list into spacy docs    
def get_search_keys(filtered, nlp):
    raw_terms = read_term_list(filtered) 
    return  [nlp(term) for term in raw_terms]

#generate patterns to match docs exactly
def get_verbatim(doc):
    return [{"LOWER": w.lower_} for w in doc]

#patterns to match by lemma
def get_lemmas(doc):
    return [{"LEMMA": w.lemma_} for w in doc]

#patterns to match by lemma with additional adjectives
def specified(patterns):
    out = []
    for pattern in patterns:
        out.append([{"POS" : "ADJ"}] + pattern)
        out.append([{"POS" : "ADJ"}, {"POS" : "ADJ"}] + pattern)
    return out    

#create a matcher with all the patterns we will use
def initialize_matcher(nlp, filtered):
    matcher = Matcher(nlp.vocab)
    #list of patterns matching our lowercased filtered list
    verbatim = [get_verbatim(term) for term in filtered]
    lemmas = [get_lemmas(term) for term in filtered]
    specific = specified(lemmas + verbatim)
    matcher.add("verbatim", verbatim)
    matcher.add("lemmas", lemmas)
    matcher.add("specific", specific)
    return matcher

#get rid of overlapping terms; that is, remove terms whose ends include parts of the next term


def filter_overlap(terms):
    non_overlapping = []
    for i, term in enumerate(terms):
        if (i < len(terms) - 1 
            and (term[1] <= terms[i + 1][0] or (term[0] < terms[i + 1][0] and term[1] >=  terms[i + 1][1]))
            and (not non_overlapping or term[1] > non_overlapping[-1][1])):
            #if this term ends before the next one begins            if :
            #print(non_overlapping)
            non_overlapping.append(term)
        elif i == len(terms) - 1 and (not non_overlapping or term[1] > non_overlapping[-1][1]):
            non_overlapping.append(term)    
    return non_overlapping

def rule_based_inference(sentence, nlp, matcher):
    sentence_terms = []
    doc = nlp(sentence)
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = (start, end)
        sentence_terms.append(span)
    #print(sentence_terms.append)
    return filter_overlap(sentence_terms)

def main(test_sentences, nlp, filtered_file):
    #nlp = spacy.load('en_core_web_sm')

    filtered = get_search_keys(filtered_file, nlp)
    matcher = initialize_matcher(nlp, filtered)
    term_locs = []
    for i, sentence in enumerate(test_sentences):
        term_loc = rule_based_inference(sentence, nlp, matcher)
        term_locs.append(term_loc)
    return term_locs

def inference_gold(filter_method, sentences):
    
    if filter_method == "tfidf":

    nlp = spacy.load('en_core_web_sm')
    t = main(test_sentences, nlp, "output/merged_tfidf.csv")
    import utils
    tokenized =  word_tokenize(test_sentences[0])
    print(utils.convert_iob(tokenized, t[0]))

