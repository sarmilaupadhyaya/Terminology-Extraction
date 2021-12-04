from nltk.tokenize import sent_tokenize


data = open("inference.txt", "r").read()

sentences = sent_tokenize(data)

with open("clean_inference.txt", "w") as f:
    for sentence in sentences:
        f.write(sentence)
        f.write("\n")
