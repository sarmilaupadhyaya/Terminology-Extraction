
class document:
    def __init__(self,filepath):
        self.text = open(filepath, "r").read()
        self.sent_text = nltk.sent_tokenize(self.text)
        
    def create_iob(self):
        self.sen_to_iob = dict()
        for i,each in enumerate(self.sent_text):
            self.sen_to_iob[i] = convert_iob(each)
        

