import os
import nltk
import pandas as pd
import math
import nltk
from nltk.tokenize import word_tokenize
from natsort import natsorted
from nltk.stem import PorterStemmer
nltk.download('punkt')

## Tokenization and Stemming

print("Tokenized Terms:\n")

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

    
print(document_of_tokens) 
print("       ")

print("Stemmed Words:\n")

def stem_func():
    docs_directory = r'E:\College\Level 3\Semester 1\Data Storage and Rtrievel\Project\Code\files'
    doc_of_stem=[]
    for i in range(1, 11):
        file_path=f"{i}.txt"
        with open(os.path.join(docs_directory, file_path), 'r', encoding='utf-8') as file:
            file_contents = file.read()
            input_tokens = nltk.word_tokenize(file_contents)
            stemmer = PorterStemmer()
            stem_words=[]

        for word in input_tokens:
            stemmed_word = stemmer.stem(word)
            stem_words.append(stemmed_word)
        doc_of_stem.append(stem_words)

    return doc_of_stem
     
doc_of_stem=stem_func()     
print(doc_of_stem)





# Positional Index
print("Positional Index:\n")

document_number = 1
positional_index = {}

for document in doc_of_stem:
    for positional, term in enumerate(document):
        if term in positional_index:
            positional_index[term][0] += 1
            if document_number in positional_index[term][1]:
                positional_index[term][1][document_number].append(positional)
            else:
                positional_index[term][1][document_number] = [positional]
        else:
            positional_index[term] = [1, {document_number: [positional]}]

    document_number += 1

print(positional_index)
print("\n")


# # phrase query 

# query =input("Query:")

# final_list = [[] for i in range(len(doc_of_stem))]

# for word in query.split():
#     if word in positional_index:
#         for key in positional_index[word][1].keys():
#             if final_list[key-1] != []:
#                 if final_list[key-1][-1] == positional_index[word][1][key][0] -1 :
#                     final_list[key-1].append(positional_index[word][1][key][0])
#             else:
#                 final_list[key-1].append(positional_index[word][1][key][0])

# print(final_list)


# # # for position, list in enumerate(final_list, start=1):
# # #     # print(position, list)
# # #     if (len(list) == len(query.split())):
# # #         print(position)


# # documents = [
# #     ['antony', 'brutus', 'ceasar', 'cleopatra', 'mercy', 'worser'],
# #     ['antony', 'brutus', 'ceasar', 'calpurnia'],
# #     ['mercy', 'worser']
# #     ]

# # all_words = []
# # for doc in documents:
# #     for word in doc:
# #         all_words.append(word)


# # def get_term_freq(doc):
# #     words_found = dict.fromkeys(all_words, 0)
# #     for word in doc:
# #         words_found[word] += 1
# #     return words_found

# # term_freq = pd.DataFrame(get_term_freq(documents[0]).values(), index=get_term_freq(documents[0]).keys())

# # for i in range(1, len(documents)):
# #     term_freq[i] = get_term_freq(documents[i]).values()

# # term_freq.columns = ['doc'+str(i) for i in range(1, 4)]

# # # print(term_freq)
