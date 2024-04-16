#Text Data Preprocessing Lib
import nltk

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

import json
#pandas-csv
import pickle
#store compressed form
import numpy as np

words=[]
#take out unique words and sort;empty list;;all the words
classes = []
#tag/categories
word_tags_list = []
ignore_words = ['?', '!',',','.', "'s", "'m"]
train_data_file = open('intents.json').read()
intents = json.loads(train_data_file)
#json->dictionary file

# function for appending stem words
def get_stem_words(words, ignore_words):
    stem_words = []
    #root of all words 
    for word in words:
        #in-membership operator
        if word not in ignore_words:
            w = stemmer.stem(word.lower())
            #lowercase
            stem_words.append(w)  
    return stem_words

for intent in intents['intents']:
    #intents key in intents.json
        # Add all words of patterns to list
        for pattern in intent['patterns']:            
            pattern_word = nltk.word_tokenize(pattern)   
            #split sentences to word:token 1st step in sequecing to no.= word         
            words.extend(pattern_word)  
            #add in list (end) (like append but comparison w/lists)                    
            word_tags_list.append((pattern_word, intent['tag']))
            #tupple form
        # Add all tags to the classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            stem_words = get_stem_words(words, ignore_words)

print(stem_words)
print(word_tags_list[0]) 
print(classes)   

#Create word corpus for chatbot
def create_bot_corpus(stem_words, classes):

    stem_words = sorted(list(set(stem_words)))
    #set-unique remove duplicates-make list-sort
    classes = sorted(list(set(classes)))

    pickle.dump(stem_words, open('words.pkl','wb'))
    #change datatype-dump wb-write binary make pikle
    pickle.dump(classes, open('classes.pkl','wb'))

    return stem_words, classes

stem_words, classes = create_bot_corpus(stem_words,classes)  

print(stem_words)
print(classes)

#Create Bag Of Words
training_data=[]
number_of_tags=len(classes)
label=[0]*number_of_tags
#repeat 0's
for word_tags in word_tags_list:
     bow=[]
     pattern_words=word_tags[0]
     for word in pattern_words:
          index=pattern_words.index(word)
          word=stemmer.stem(word.lower())
          pattern_words[index]=word
     for word in stem_words:
          if word in pattern_words:
               bow.append(1)
          else:
               bow.append(0)
     print(bow)

     

#Create training data
