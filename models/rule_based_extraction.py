## read manually filtered term and apply rule based system to get term from all corpus
# generate random integer values
from random import seed
from random import randint

def rule_based_inference(sentence):
    terms = []
    seed(1)
    for i in range(0,1):
        length = len(sentence.split(" "))
        terms.append([randint(0, length),randint(0, length)])
    return terms

sentence_term("I am doing a test")
