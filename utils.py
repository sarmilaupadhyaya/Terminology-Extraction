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

def iob2span(iob_data):
# returns list of (begin, end, words)
    spans = []
    current_tag = 'O'
    current_words = []
    current_begin = 0
    current_end = 0
    begin_of_line, _, _ = iob_data[0]
    end_of_line, _ , _ = iob_data[-1]
    for index, word, tag in iob_data:
        if tag == 'B':
            if current_tag == 'I' or current_tag == 'B':
            # if a markable just ended
                new_span = (current_begin, current_end, current_words)
                spans.append(new_span)
            # start new
            current_begin = index
            current_end = index
            current_words = [word]
            if index == end_of_line:
                new_span = (current_begin, current_end, current_words)
                spans.append(new_span)
        if tag == 'I':
            if current_tag == "O":
                if index-1 >= begin_of_line:
                    current_begin = index-1
                    current_words = [current_word]
                else:
                    current_begin = begin_of_line
            current_end = index
            current_words.append(word)
            # if this is the last tag:
            if index == end_of_line:
                new_span = (current_begin, current_end, current_words)
                spans.append(new_span)
        if tag == 'O':
            if current_tag == 'I' or current_tag == 'B':
            # if a markable just ended, save it
                new_span = (current_begin, current_end, current_words)
                spans.append(new_span)
            # otherwise do nothing and move on
        # save the new tag to current_tag
        current_tag = tag
        current_word = word
    return spans

