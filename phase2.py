import os
import nltk
import numpy as np
import json
import pandas as pd
import math
import nltk
from nltk.tokenize import word_tokenize
from natsort import natsorted
from nltk.stem import PorterStemmer
nltk.download('punkt')


if __name__ == "__main__":
    print("Term Frequency:\n")

document_of_tokens=[]

files_name = natsorted(os.listdir('files'))

for file_name in files_name:
  with open(f'files/{file_name}','r') as f:
     document = f.read()
  tokenized_doc = word_tokenize(document)
  tokens=[]

  for word in tokenized_doc:
        
          tokens.append(word)
  document_of_tokens.append(tokens)




all_words = []
for doc in document_of_tokens:
    for word in doc:
        all_words.append(word)
 
 
def get_term_freq(doc):
    words_found = dict.fromkeys(all_words, 0)
    for word in doc:
        words_found[word] += 1
    return words_found
 
term_freq = pd.DataFrame(get_term_freq(document_of_tokens[0]).values(), index=get_term_freq(document_of_tokens[0]).keys())
 
for i in range(1, len(document_of_tokens)):
    term_freq[i] = get_term_freq(document_of_tokens[i]).values()
 
term_freq.columns = ['doc'+str(i) for i in range(1, 11)]
if __name__ == "__main__":
    print(term_freq)
    print("\n\n")




if __name__ == "__main__":
    print("Weighted Frequency:\n")

def weighted_tf(x):
    if x > 0:
        return math.log10(x) + 1
    return 0
w_tf = term_freq.copy()
for i in range(0, len(document_of_tokens)):
    w_tf['doc'+str(i+1)] = term_freq['doc'+str(i+1)].apply(weighted_tf)
if __name__ == "__main__":
    print(w_tf)


if __name__ == "__main__":  
    print("Document Frequency:\n")


tdf = pd.DataFrame(columns=['df', 'idf'])
for i in range(len(term_freq)):
    in_term = w_tf.iloc[i].values.sum()

    tdf.loc[i, 'df'] = in_term

    tdf.loc[i, 'idf'] = math.log10(10 / (float(in_term)))

tdf.index=w_tf.index
if __name__ == "__main__":
    print(tdf)
    print("\n\n")


if __name__ == "__main__":
    print("TF*IDF:\n")

tf_idf = w_tf.multiply(tdf['idf'], axis=0)


if __name__ == "__main__":
    print(tf_idf)
    print("\n\n")


if __name__ == "__main__":
    print("Document length:\n")


def get_doc_len(col):
    return (np.sqrt(tf_idf[col].apply(lambda x: x**2).sum()))
doc_len = pd.DataFrame()
for col in tf_idf.columns:
    doc_len.loc[0, col+'_length']= get_doc_len(col)


if __name__ == "__main__":
    print(doc_len)
    print("\n\n")


if __name__ == "__main__":
    print("Normalized TF.IDF:\n")


def get_normalized_tf_idf(col, x):
    try:
        return x / doc_len[col+'_length'].values[0]
    except:
        return 0
normalized_tf_idf = pd.DataFrame()
for col in tf_idf.columns:
    normalized_tf_idf[col] = tf_idf[col].apply(lambda x : get_normalized_tf_idf(col, x))

if __name__ == "__main__":
    print(normalized_tf_idf)
    print("\n\n")

