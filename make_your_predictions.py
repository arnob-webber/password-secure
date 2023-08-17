import dill
import email 
from collections import Counter
from bs4 import BeautifulSoup
import nltk
import re
import urlextract
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

with open('/home/elneklawy/Desktop/emails/model.pkl', 'rb') as f:
    loaded_model = dill.load(f)


def html_to_plain(mail):
    soup = BeautifulSoup(mail.get_payload(), "html.parser")
    plain = soup.text.replace("=\n", "")
    plain = re.sub(r"\s+", " ", plain)
    
    return plain.strip()

def mail_to_plain(mail):
    txt_cont = ''
    for part in mail.walk():
        part_content_type = part.get_content_type()
        if part_content_type not in ['text/plain', 'text/html']: continue
        if part_content_type == 'text/plain':
            txt_cont += part.get_payload()
        else:
            txt_cont += html_to_plain(part)

    return txt_cont

    
class WordCounterTransformer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        stemmer = nltk.PorterStemmer()
        url_extractor = urlextract.URLExtract()
        email_pattern = r'(?:[a-z0-9!#$%&\'*+\=?^_`{|}~-]+(?:\.[a-z0-9!#$%&\'*+\=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])'
        X_transformed = []
        
        for mail in X:
            txt: str = mail_to_plain(mail)
            if txt is None:
                txt = 'nothing'
            txt = txt.lower()

            urls = url_extractor.find_urls(txt)
            for url in urls: # replace any url with "URL"
                txt = txt.replace(url, ' URL ')

            txt = re.sub(r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', 'NUMBER', txt) # replace any number with "NUMBER"

            #txt = cleantext.replace_emails(txt, replace_with="MAIL")    # repace any email with "EMAIL"
            txt = re.sub(email_pattern, 'MAIL', txt)

            txt = re.sub(r'\W+', ' ', txt, flags=re.M) # remove punctuation

            for word in txt.split(): # stem words (TO BE REVISED)
                stemmed = stemmer.stem(word, to_lowercase=False)
                txt = txt.replace(word, stemmed)

            word_counts = Counter(txt.split())
            X_transformed.append(word_counts)

        return X_transformed


class WordCountVectorizerTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size

    def fit(self, X, y=None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += min(count, 10)
                
        most_common = total_count.most_common()[:self.vocabulary_size]
        self.vocabulary_ = {word: index + 1
                            for index, (word, count) in enumerate(most_common)}
        
        return self
    
    def transform(self, X, y=None):
        rows, cols, data = [], [], []

        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)

        return csr_matrix((data, (rows, cols)),
                          shape=(len(X), self.vocabulary_size + 1))


user_input = input('Enter email: ')
input_email = email.message.Message()
input_email.set_payload(user_input)

proba = round(loaded_model.predict_proba([input_email])[0][1] * 100, 2)
pred = loaded_model.predict([input_email])


print()
print(f"There is a {proba}% chance that the email is spam.")
print('My guess: Spam') if pred == 1 else print('My guess: Not Spam')