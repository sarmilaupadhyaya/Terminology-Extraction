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
filtered_data_pointwise=params.extracted_terms_pointwise_analysed

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
        result_rule_based = rb.main(filtered_data,nlp, filtered_data)
    total_tags = []
    for sentence, token in zip(sentences, result_rule_based):
            tokenized =  sentence.split(" ")
            iobs = utils.convert_iob(tokenized, token)
            for word, tag in iobs:
                total_tags.append(tag)
    df["tag_rule"] = pd.Series(total_tags)
 

    result_bilm= bi.main(df, filter_method)
    import pdb
    pdb.set_trace()
    result_bilm = result_bilm[:len(df)]
    df["tag_nn"] = pd.Series(result_bilm)
    get_analysis(df, tag2idx, method="rule based")
    get_analysis(df, tag2idx, method="NN based")
    print("terms extracted rulebased")
    terms_rulebased = utils.iob2span(list(zip(range(len(df)),df["word"].tolist(), df["tag_rule"].tolist())))
    print("terms extracted nn based")
    terms_nnbased = utils.iob2span(list(zip(range(len(df)),df["word"].tolist(), df["tag_nn"].tolist())))
    print("terms to neextracted rulebased")
    terms_to_extract = utils.iob2span(list(zip(range(len(df)),df["word"].tolist(), df["tag"].tolist())))
    df.to_csv("tete.csv")

    with open("output/extracted_rule.txt","w") as f:
        for x in terms_rulebased:
            f.write(" ".join(x[2]))
            f.write("\n")
    with open("output/extracted_nn.txt", "w") as f:
       for x in terms_nnbased:
           f.write(" ".join(x[2]))
           f.write("\n")

    with open("output/extracted_gold.txt", "w") as f:
       for x in terms_to_extract:
           f.write(" ".join(x[2]))
           f.write("\n")



