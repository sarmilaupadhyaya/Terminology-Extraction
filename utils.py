## convert corpus and its list of term extracted into iob format and save

def convert_iob(sentence, tokens):
    
    iob = []
    for i,each in enumerate(sentence):
        found=False
        for term in tokens:
            start = term[0]
            ends = term[-1]
            if i == start:
                iob.append("B")
                found=True
            elif i < ends and i> start:
                iob.append("I")
                found=True
        if not found:
            iob.append("O")
    return list(zip(sentence, iob))

