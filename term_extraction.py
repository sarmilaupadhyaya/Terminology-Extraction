import params
import argparse
import bilm_inference as bi

# get the gold annotation, get result and terms from rule based as well as bilstm model then saving them in a txt file and reporting the accuracy
# please change the path of gol

if __name__=='__main__':

    parser = argparse.ArgumentParser()# Add an argument
    parser.add_argument('--filter_method', type=str, required=True)# Parse the argument
    parser.add_argument('--gold_annotation_path', type=str, required=True)# Parse the argument
    args = parser.parse_args()
    filter_method=args.filter_method
    gold_path=args.gold_annotation_path
    
    result_rule_based = rb.main(filter_method, filepath)
    result_bilm= bi.main(filter_method, filepath)



