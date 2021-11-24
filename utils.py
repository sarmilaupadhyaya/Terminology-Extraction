## convert corpus and its list of term extracted into iob format and save

def convert_iob(sentence):
    text = sentence.split(" ")
    iob = []
    for i,each in enumerate(text):
        for term in sentence_term(sentence):
            start = term[0]
            ends = term[-1]
            if i == start:
                iob.append("B")
            elif i <= ends and i> start:
                iob.append("I")
            else:
                iob.append("0")
    return list(zip(text, iob))

convert_iob("I am doing a test")            
