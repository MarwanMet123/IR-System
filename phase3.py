import os
import sys
import nltk
import pandas as pd
import math
import nltk
from nltk.tokenize import word_tokenize
from natsort import natsorted
from nltk.stem import PorterStemmer
nltk.download('punkt')
import phase2

# Tokenization and Stemming

# print("Tokenized Terms:\n")

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

    
# print(document_of_tokens) 
# print("       ")

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
# print(doc_of_stem)






# Positional Index
# print("Positional Index:\n")

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

# print(positional_index)
print("\n")

def StemmingFunction(list):
    stemmer= PorterStemmer()
    www=[]
    for word in list:
            stemmed_wordss = stemmer.stem(word)
            www.append(stemmed_wordss)
    return www



query = input("Query: ")
query_words = query.split()  # Split the query into a list of words
stemmed_query_words=StemmingFunction(query_words)
query_len = len(stemmed_query_words)
isAnd = False
isOr = False
isNot = False

Index = []
num = []
Boolean=[]

for i, word in enumerate(stemmed_query_words):
    if word.lower() in ["and", "or", "not"]:
        Index.append([i, word])
        num.append(i)
        Boolean.append(word)

all_sentence = []
j = 0

for i in num:
    # Exclude conjunctions from the subset
    subset = [word for word in stemmed_query_words[j:i] if word.lower() not in ["and", "or", "not"]]
    
    result_string = ' '.join(subset)
    all_sentence.append(result_string)
    j = i + 1  # Move to the next index after the conjunction

# Add the last subset
subset = [word for word in stemmed_query_words[j:query_len] if word.lower() not in ["and", "or", "not"]]
result_string = ' '.join(subset)
all_sentence.append(result_string)

# print(all_sentence)
# print(Index)
# print (num)
# print(Boolean)

all_positions=[]
for sentencee in all_sentence:
    splitted=sentencee.split()
    final_list = [[] for i in range(len(doc_of_stem))]
    for word in splitted:
        # print(word)
        if word in positional_index:
            for key in positional_index[word][1].keys():
                if final_list[key-1] != []:
                    if final_list[key-1][-1] == positional_index[word][1][key][0] -1 :
                        final_list[key-1].append(positional_index[word][1][key][0])
                else:
                    final_list[key-1].append(positional_index[word][1][key][0])

            # print(final_list)


        positions = []

        for position, sublist in enumerate(final_list, start=1):
            if len(sublist) == len(splitted):
                positions.append(position)

        if positions:
            # print(positions)
            all_positions.append(positions)
            final_list=[]
# print(Boolean)            
# print(all_positions)
if len(Boolean)==len(all_positions):
    print("Please enter a valid query !")
    sys.exit()
all_positions_length=len(all_positions)

def Calculate(list1,list2,Operation):
    if Operation=='and':
        result = [item for item in list1 if item in list2]
    elif Operation=='or':
        result = list(set(list1) | set(list2))  
    elif Operation=='not':
        result = [item for item in list1 if item not in list2]     
    return result      


temp_positions=all_positions.copy()


for i in range(0,all_positions_length-1):
    j=i+1
    if (j in range(all_positions_length-1,all_positions_length+10))&len(Boolean)!=1&i!=0:
        break 
    for t in Boolean:
        temp_positions[j]=Calculate(temp_positions[i],temp_positions[j],t)
        temp_positions.remove(temp_positions[i])
        # print(temp_positions)
        
temp_positions.sort()
# print(temp_positions)
listtt=[]
for item in temp_positions:
    for i in item:
        listtt.append('doc'+str(i))

# print(listtt)



# Query Details

def preprocessing(query):
    query.lower()
    token=word_tokenize(query)
    prepared_doc=[]
    for term in token:
        if term not in('and','or','not') :
            prepared_doc.append(term)
    return prepared_doc



docs_found = listtt
if docs_found == []:
    print("Not Found")
new_q = preprocessing(query)
# print(new_q)
query = pd.DataFrame(index=phase2_mariem.normalized_tf_idf.index)
query['tf'] = [1 if x in new_q else 0 for x in list(phase2_mariem.normalized_tf_idf.index)]
query['w_tf'] = query['tf'].apply(lambda x : phase2_mariem.weighted_tf(x))
product = phase2_mariem.normalized_tf_idf.multiply(query['w_tf'], axis=0)
query['idf'] = phase2_mariem.tdf['idf'] * query['w_tf']
query['tf_idf'] = query['w_tf'] * query['idf']
query['normalized'] = 0
for i in range(len(query)):
    query['normalized'].iloc[i] = float(query['idf'].iloc[i]) / math.sqrt(sum(query['idf'].values**2))
print('Query Details')
print(query.loc[new_q])
product2 = product.multiply(query['normalized'], axis=0)
scores = {}
for col in listtt:
        scores[col] = product2[col].sum()
product_result = product2[list(scores.keys())].loc[new_q]
print()
print('Product (query*matched doc)')
print(product_result)
print()
print('product sum')
print(product_result.sum())
print()
print('Query Length')
q_len = math.sqrt(sum([x**2 for x in query['idf'].loc[new_q]]))
print(q_len)
print()
print('Cosine Simliarity')
print(product_result.sum())
print()
sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
print('Returned docs')
for typle in sorted_scores:
    print(typle[0], end=" ")



