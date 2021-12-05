import params
import argparse
import spacy
import pickle
import pandas as pd
from future.utils import iteritems
import params
import bilm_inference as bi
import rule_based_extraction as rb
import utils

filtered_data = params.extracted_terms_analysed
filtered_data_tfidf=params.extracted_terms_tfidf_analysed

nlp = spacy.load('en_core_web_sm')

# get the gold annotation, get result and terms from rule based as well as bilstm model then saving them in a txt file and reporting the accuracy
# please change the path of gol

tag2id_path = params.tag2id_path
tag2id_path_tfidf=params.tag2id_path_tfidf

def get_analysis(df,tag2idx, method):

    TP = {}
    TN = {}
    FP = {}
    FN = {}
    for tag in tag2idx.keys():
        TP[tag] = 0
        TN[tag] = 0
        FP[tag] = 0
        FN[tag] = 0

    def accumulate_score_by_tag(gt, pred):
        """
        For each tag keep stats
        """
        if gt == pred:
            TP[gt] += 1
        elif gt != 'O' and pred == 'O':
            FN[gt] +=1
        elif gt == 'O' and pred != 'O':
            FP[gt] += 1
        else:
            TN[gt] += 1
   
    print("method:", method)
    if method == "rule based":
        method_name = "tag_rule"
    else:
        method_name = "tag_nn"

    for idx, (w,pred) in enumerate(zip(df["tag"],df[method_name])):
        accumulate_score_by_tag(w,pred)

    
    for tag in tag2idx.keys():
        print(f'tag:{tag}')
        print('\t TN:{:10}\tFP:{:10}'.format(TN[tag],FP[tag]))
        print('\t FN:{:10}\tTP:{:10}'.format(FN[tag],TP[tag]))

    for tag in tag2idx.keys():
        print(f'tag:{tag}')
        accuracy = (TP[tag] + TN[tag])/(TP[tag]+TN[tag]+FN[tag]+FP[tag])
        precision = (TP[tag]/TP[tag]+FP[tag])
        recall = (TP[tag]/TP[tag]+TN[tag])
        f1 = (2*precision*recall)/(precision+recall)
        print('\t Precision:{:10}\tRecall:{:10}'.format(precision,recall))
        print('\t F1 Score:{:10}\tAccuracy:{:10}'.format(f1,accuracy))
    

def get_analysis_terms(gold_term, pred_term):

    
    gold_term = [x.lower() for x in gold_term]
    pred_term = [x.lower() for x in pred_term]
    gold_term = set(gold_term)
    pred_term = set(pred_term)

    TP = len(gold_term.intersection(pred_term))
    FN = len([x for x in gold_term if x not in pred_term])
    FP = len([x for x in pred_term if x not in gold_term])
    print(len(gold_term), "number of gold terms")
    print(TP, "terms were rightly identified")
    print(FN, "terms were not identified")
    print(FP, "terms were identified which were not terms")



def getting_sentences(df):

    sentences = []
    groups = df.groupby(['sen_id'])

    for key, group in groups:

        sentence = " ".join(group["word"].tolist())
        sentences.append(sentence)
    return sentences

if __name__=='__main__':

    parser = argparse.ArgumentParser()# Add an argument
    parser.add_argument('--filter_method', type=str, required=True)# Parse the argument
    parser.add_argument('--gold_annotation_path', type=str, required=True)# Parse the argument
    args = parser.parse_args()

    filter_method=args.filter_method
    if filter_method == "tfidf":
        tag2id_path=tag2id_path_tfidf
    
    with open(tag2id_path, 'rb') as handle:
        tag2idx = pickle.load(handle)
        idx2tag = {v: k for k, v in iteritems(tag2idx)}

    gold_path=args.gold_annotation_path 
    df = pd.read_csv(gold_path).dropna()
    sentences = getting_sentences(df)
    if filter_method == "tfidf":
        result_rule_based = rb.main(sentences, nlp, filtered_data_tfidf)
    else:
        result_rule_based = rb.main(sentences,nlp, filtered_data)
    total_tags = []
    for sentence, token in zip(sentences, result_rule_based):
            tokenized =  sentence.split(" ")
            iobs = utils.convert_iob(tokenized, token)
            for word, tag in iobs:
                total_tags.append(tag)
    df["tag_rule"] = pd.Series(total_tags)
 

    result_bilm= bi.main(df, filter_method)
    result_bilm = result_bilm[:len(df)]
    df["tag_nn"] = pd.Series(result_bilm)
    df.to_csv("tete.csv")
    get_analysis(df, tag2idx, method="rule based")
    get_analysis(df, tag2idx, method="NN based")
    terms_rulebased = utils.iob2span(list(zip(range(len(df)),df["word"].tolist(), df["tag_rule"].tolist())))
    terms_nnbased = utils.iob2span(list(zip(range(len(df)),df["word"].tolist(), df["tag_nn"].tolist())))
    terms_to_extract = utils.iob2span(list(zip(range(len(df)),df["word"].tolist(), df["tag"].tolist())))
    
    terms_rule = []
    with open("output/extracted_rule"+filter_method+".txt","w") as f:
        for x in terms_rulebased:
            f.write(" ".join(x[2]))
            terms_rule.append(" ".join(x[2]))
            f.write("\n")
    terms_nn = []
    with open("output/extracted_nn"+filter_method+".txt", "w") as f:
       for x in terms_nnbased:
           f.write(" ".join(x[2]))
           terms_nn.append(" ".join(x[2]))
           f.write("\n")
    terms_gold = []
    with open("output/extracted_gold"+filter_method+".txt", "w") as f:
       for x in terms_to_extract:
           f.write(" ".join(x[2]))
           terms_gold.append(" ".join(x[2]))
           f.write("\n")
    print("Terms based analysis on rule based extraction result")
    get_analysis_terms(terms_gold, terms_rule )
    print("Terms based analysis on NN based extraction result")
    get_analysis_terms(terms_gold, terms_nn )


