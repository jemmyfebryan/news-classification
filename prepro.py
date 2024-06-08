import re
import string

# ----- Case Folding -----
def change_lower_case(text):
    return text.lower()

def remove_hyphen(text):
    return text.replace("-", " ").replace("/", " ")

def remove_news_special(text):
    # remove tab, new line, and back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\'," ")
    # remove non ASCII (emoticon, kanji, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # remove mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    # remove incomplete url
    return text.replace("http://", " ").replace("https://", " ")

#remove number
def remove_number(text):
    return re.sub(r'[-+]?[0-9]+', '', text)

#remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))

#remove whitespace leading & trailing
def remove_whitespace_LT(text):
    return text.strip()

#remove multiple whitespace into single whitespace
def remove_whitespace_multiple(text):
    return re.sub('\s+',' ',text)

#remove single char
def remove_single_char(text):
    return re.sub(r"\b[a-zA-Z]\b", " ", text)

def all_prepro(text):
    text = change_lower_case(
        remove_hyphen(
            remove_news_special(
                remove_number(
                    remove_punctuation(
                        remove_whitespace_LT(
                            remove_whitespace_multiple(
                                remove_single_char(
                                    text
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    return text