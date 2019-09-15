# -*- coding: utf-8 -*-
"""
This script triesto find topic of the reviews based on word similarity (word embedding)

"""


### IMPORT LIBRARIES

import os
import warnings
warnings.filterwarnings('ignore')
os.chdir('/Users/gosc/Desktop/Kaggle/Zomato - Bangalore') 
import time
import random
import numpy as np 
import pandas as pd 
import operator
#import matplotlib.pyplot as plt
#import seaborn as sns
#from matplotlib import rcParams
#import pandas_profiling


### text processing
import re
from nltk.corpus import stopwords
import gensim
from gensim.models import Word2Vec, KeyedVectors
from collections import defaultdict  # For word frequency
#import spacy  # For preprocessing
from gensim.scripts.glove2word2vec import glove2word2vec

#import nltk
#from nltk.stem.porter import PorterStemmer
#from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from nltk.tokenize import word_tokenize
#from nltk.stem import WordNetLemmatizer
#import shap
#import lime

##### LIST CHECKING LEGTH OF THE LIST

def lenListOfLits(listOfLists):
    total_length = 0
    for i in listOfLists:
        total_length+=len(i)
    return total_length

def lenListOfLits_2(listOfLists):
    total_length = 0
    for i in listOfLists:
        total_length+=len(i[1])
    return total_length

#lemmatizer = WordNetLemmatizer() 
#porter = PorterStemmer()


####### READING DATA #########
start = time.time()

data=pd.read_csv('zomato.csv',encoding='latin-1')

######## REMOVING DUPLICATES ######
data= data.drop_duplicates(subset=['name','address'],keep='first').reset_index()

######DROPPING IRRELEVANT COLUMNS ########
data.drop(['url','phone','address'],inplace=True,axis=1)


###### RENAMING COLUMNS ########
data.rename(columns=
        {'name':'rest_name','approx_cost(for two people)': 'meal_cost', 'listed_in(city)':
         'Neighbourhood','listed_in(type)': 'restaurant_type','rest_type':'restaurant_category'}, inplace=True)

'''
Chunk of code below proces review_list columns and splits reviews and rates.
Then duplicates by review are removed with corresponding rates.
Based on that two new variables are added to dataset - 'rate_list' and 'review_list'
that can former can be used as prediction label and later can be used 
for further NLP variable creation.
'''   

######## RATE PREPROCESSING ####
def preprocessing(data=data):
    data.rate = data.rate.astype(str).apply(lambda x: x.replace('/5','')).apply(lambda x: x.replace(',/n',''))
    data.loc[(data.rate =='NEW') | (data.rate =='-'), 'rate'] = np.nan
    data.rate = data.rate.apply(lambda x: float(x))
   # data['meal_cost'] = data['meal_cost'].str.replace(',', '').astype(float)
   # data['votes'] = data['votes'].astype(float)
    #data['rest_name'] = data['rest_name'].str.strip().replace(['[^A-Za-z0-9_.,!"\s]+'], [''], regex=True)

preprocessing(data=data)
end = time.time()

print("Preprocessing data: ", end - start)


#### Copying data for processing revievs
reviews=data.reviews_list.copy()

############ RATES EXTRACTION ##############

start = time.time()
rate_pattern = re.compile("\(\'Rated [0-9].[0-9]")
rate_list_t = [re.findall(rate_pattern,rev) for rev in reviews]

    
rate_list=[]

rate_pattern = re.compile("\(\'Rated ")

for i in range (len(rate_list_t)):
    rate_list_temp=[]
    if len(rate_list_t[i]) != 0: 
        for k in range(len(rate_list_t[i])): 
            single_rate  = re.sub(rate_pattern, "", rate_list_t[i][k])
            rate_list_temp.append(float(single_rate))
        rate_list.append(rate_list_temp)
    else:
        rate_list.append(rate_list_temp)
end = time.time()

print("Rate cleaning time: ", end - start)

del rate_list_temp, rate_list_t, single_rate, i, k



############ REVIEWS EXTRACTION ###########

start = time.time()
review_list= []
for rev in reviews:#first 100
    if rev=='[]':
        review_list.append([])
    else:
        single_review = re.sub(r'\\n|\\x|\.', ' ', rev.lower())
        single_review  = re.sub(r'\), \(\'rated', 'splithere', single_review) # first this since it nee dto be splitted by 
        single_review  = re.sub("[^a-z]+", " ", single_review) # all not being a-z letters to remove
        ### version 1
        single_review = [x for x in single_review.split() if x not in ['rated']]
        
        single_review = ' '.join(single_review)
        single_review1 = single_review.split('splithere')
        review_list.append(single_review1)
end = time.time()

print("Review cleaning time: ", end - start)

del single_review, single_review1, rev


########### CHECKING IF LENGTH LIST OF LIST IS THE SAME ########
RateLength=lenListOfLits(rate_list)
ReviewLength=lenListOfLits(review_list)
print('Length Of Rate list of list:', RateLength)
print('Length Of Review list of list:', ReviewLength)



######## CHECKING DUPLICATES ####

RevRate_list=[]
for idx, rest in enumerate(rate_list): 
    tuplist=[]
    for i, rev in enumerate(rest):
        tup = (rate_list[idx][i], review_list[idx][i].strip())
        tuplist.append(tup)
    RevRate_list.append(tuplist) 



###### REMOVING DUPLICATES ##################
Dedup_Review_List = [list(set(x)) for x in RevRate_list] 
RevRateLength=lenListOfLits(Dedup_Review_List)
print('Length Of deduplicatd Review/Rate list of tuples:', RevRateLength)  


del RateLength, ReviewLength, RevRateLength
del i, idx, rev, tup, tuplist



##### ADDING RATES AND REVIEWS TO ORIGINAL DF

rate_list2=[]
review_list2=[]
for i in Dedup_Review_List:
    if len(i) != 0:
        for z in i:
            list1, list2 = zip(*i)
    else:
        list1=[]
        list2 = []
    rate_list2.append(list(list1))
    review_list2.append(list(list2))
    
del list1,list2, i, z   
#### Adding new variables to pandas - rates, reviews, count of reviews
    
data['rate_list']=rate_list2
data['review_list']=review_list2

review_count=[]
for i in review_list2:
    review_count.append(len(i))
    
data['review_cnt']=review_count

del rate_list, review_list, RevRate_list

print('------------------------------------------------------------------')
print('REVIEW PREPROCESSING END')
print('------------------------------------------------------------------')

#################### WORDS SIMILARITY #########################


'''
This bit of code aims on trying to find topic related reviews by
1) Iterating through alll restaurants reviews and saving restaurant/review index
2) Check similarity of all the tokens within single review to one chosen topic word ex.food
3) Saving maximum word similarity and thsi word
4) If similarity meets treshold those review is assumed to be realted
5) Then feature is created and added to original data DF

    
To achieve the task above I have tried:
    
a) Pretrained Gensim Model
    Problems encountered:
        a) To generic corpora - not resturant related
        b) I was eble load it only as a KeydVectors -  essentially a mapping
    between words and vectors. It is not full model itself and regardless of the 
    memeory needed I wanted to load whole. I couldn't find how for long... later 
    a found a solutin but I have a this train again not adding embeddings for new wors.
    
    https://radimrehurek.com/gensim/models/keyedvectors.html
    https://rare-technologies.com/word2vec-tutorial/?fbclid=IwAR2VvJOhSQxEufTGj8d7GGTXg3khWbkrGuyWiyUBTAzdwJV8NIaIMfTBrEA
b) Own data - zomato
    a) It would be good to look for a pattern in our corpora. However - this is maybe to specific area with many indian origin dishes
    that can be not recognised as food related. Also there are several typos.
    b) WOrth enhancing by lemmanization and richer stop words to remove ex. super as food reated word
    
c) Yelp restaurant corpora
    This is realy heavy file, but public data. Training on this we can then retrain with our 
    data and hopefully arrive to feasible solutions


Later steps:
    
    a) Sentiment analysis for whole review
    b) Sentiment analysis per bigram/trigram to get word and its context
    ex. super expensive food should be found as negative and hopefully food related
    therefore lowering review
    
'''


print('------------------------------------------------------------------')
print('WORD2VEC MODEL LOADING...')


#### READING WORD2VEC MODEL
os.chdir('/Users/gosc/Desktop/Kaggle/Zomato - Bangalore')

#### PRETRAINED MODEL  - GENSIM 
start = time.time()
word2vec = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
end = time.time()
print("Loading pre - trained Word to Vec model - GENSIM", end - start) 



#### PRETRAINED MODEL  - GLOVE



''' #For now not tried
os.chdir('glove.6B')
start = time.time()
glove2word2vec(glove_input_file="glove.6B.200d.txt", word2vec_output_file="gensim_glove_vectors.txt")
w2v_glv = KeyedVectors.load_word2vec_format('gensim_glove_vectors.txt', binary=False)
end = time.time()
print("Loading pre - trained Word to Vec model - GLOVE: ", end - start) 
'''
print('WORD2VEC MODEL LOADED')
print('------------------------------------------------------------------')

###### PREPARNG DATA - LIST OF LIST OF TOKENS

#### stop words applied, however they may be not comprehensive enough
print('')
print('------------------------------------------------------------------')
print('REVIEW LIST TOKENIZATION')


ReviewSample = random.sample(list(enumerate(review_list2)), 1000)
ReviewSample = list(enumerate(review_list2))

start = time.time()
RevSampleTokens=[]
for Index, RestaurantReview in ReviewSample:
    RevSample=[]
    for Idx, Review in enumerate(RestaurantReview):
        RevSample = (Index, Idx, [x for x in Review.split(' ') if not x in set(stopwords.words('english'))])
        RevSampleTokens.append(RevSample) ### List Of Tuple with Random Index in it
end = time.time()
print("Tokenization time: ", end - start) 
print('------------------------------------------------------------------')
print('REVIEW LIST TOKENIZED')

####### CHECKING THE MOST COMMON WORDS #####

start = time.time()

word_freq = defaultdict(int)
for i in range (len(RevSampleTokens)):
    for token in RevSampleTokens[i][2]:
        word_freq[token] += 1
print('')
print('UNIQUE WORDS IN A REVIEW VOCABULARY:', len(word_freq))
print('')
print('Most Common Words:')
print(sorted(word_freq, key=word_freq.get, reverse=True)[:10])
print('')
print("Most common  words processing time: ", end - start) 



#########     WORDS SIMILARITY       #########

####  TRAINING NEW MODEL ON PRETRAINED MODEL



''' Solution found here
#https://gist.github.com/AbhishekAshokDubey/054af6f92d67d5ef8300fac58f59fcc9


I have a feeling it gives worse results so for now I am commenting this


'''
# # creating sequence of sequence as an model input

'''
sent = [i[2] for i in RevSampleTokens]

# model initialisation

model = gensim.models.Word2Vec(size=300,window=3,min_count=10)### change parameters here !
model.build_vocab(sent)#building vocabulary
training_examples_count = model.corpus_count


model.build_vocab([list(word2vec.vocab.keys())], update=True)#taking voc from pretrained model
os.chdir('/Users/gosc/Desktop/Kaggle/Zomato - Bangalore')
model.intersect_word2vec_format("GoogleNews-vectors-negative300.bin.gz",binary=True, lockf=1.0) #model intesection and training again
model.train(sent,total_examples=training_examples_count, epochs=model.iter)

'''

################################################################
###1 FOOD RELATED
print('')

print('---------------------------------------------------------------------')
print('----------------    CHECKING WORDS SIMILARITY      ------------------')
print('')
print('')
print('1. FOOD')


##  SIMILAR WORDS

'''Creating tuple of:
    a) Index Of Restuarnat
    b) Index Of Review
    c) Tokenized Review
    d) Arg Max similarity to topic word
    e) Max similarity to topic word

'''
start = time.time()
FoodSimilarity=[]
for idx, idx2, rev in RevSampleTokens:
    temp_max=0
    max_word=''
    for word in rev:
        if word in word2vec:
            temp_sim = word2vec.similarity('food', word)
        if temp_sim>temp_max:
            temp_max=temp_sim
            max_word=word
    FoodSimilarity.append((idx,  idx2, rev, temp_max, max_word))
    
end = time.time()


#### SORTING LIST

FoodSimilarity.sort(key = operator.itemgetter(3), reverse=True)

del idx, idx2, word, RevSample, Index, Idx, rev, Review, RestaurantReview
print("Word similarity processing time: ", end - start)

#### CREATING DICTIONARIES OF WORDS COUNTS AND SIMILARITIES _ FOR LATER


start = time.time()
FoodWords={}
for i in range(len(FoodSimilarity)):
    if FoodSimilarity[i][4] in FoodWords.keys():
        FoodWords[FoodSimilarity[i][4]]+=1
    else:
        FoodWords[FoodSimilarity[i][4]]=1

FoodWordsScore={}
for i in range(len(FoodSimilarity)):
    if FoodSimilarity[i][4] in FoodWordsScore.keys():
        pass
    else:
        FoodWordsScore[FoodSimilarity[i][4]]=FoodSimilarity[i][3]

FoodWordsScore_Sorted = sorted(FoodWordsScore.items(), key=lambda kv: kv[1])
FoodWordsScore_Sorted.reverse()
FoodWordsCount_Sorted = sorted(FoodWords.items(), key=lambda kv: kv[1])
FoodWordsCount_Sorted.reverse()

end = time.time()

print("Dictionaries of counts and similarities creating time: ", end - start)

#CLEANING -  RARE STRANGE WORDS OCCURING ONLY ONCE AND STOP WORDS

#### CREAING A LIST FOR EXCLUSIONS - TO REVIEW   
'''

FoodWordsScore_dict = dict(FoodWordsScore_Sorted)
FoodWordsCount_dict = dict(FoodWordsCount_Sorted)


for k,v in FoodWordsScore_dict.items():
    if (FoodWordsCount_dict[k]== 1) and (FoodWordsScore_dict[k])>= 0.7:
        print(k, FoodWordsScore_dict[k])
##this print  words worth reviewing - however it would be time consuming...
##decided not worth effort

for k,v in FoodWordsScore_dict.items():
    if FoodWordsCount_dict[k]== 1:
        FoodWordsScore_dict[k]= 0

WordsToExclude=[]
for k, v in FoodWordsScore_dict.items():
    if FoodWordsScore_dict[k]== 0:F
        WordsToExclude.append(k)
'''




####  FEATURE CREATION - COUNT OF FOOD RELATED REVIEWS

FoodRelatedCountRev={}
for i in range(len(FoodSimilarity)):
    if FoodSimilarity[i][3]>0.4:
        if FoodSimilarity[i][0] not in FoodRelatedCountRev.keys():
            FoodRelatedCountRev[FoodSimilarity[i][0]]=1 
        else:
            FoodRelatedCountRev[FoodSimilarity[i][0]]+=1
        

#### CREATING FEATURE

s = pd.Series(FoodRelatedCountRev, name='FoodRevs')
data['food_review_cnt'] = s

del s,i,max_word


### 2 
#### the same for service, localization, price related words
print('------------------------------------------------------------------------')
print('')
print('')
print('3. SERVICE')



##  SIMILAR WORDS

start = time.time()

ServiceSimilarity=[]
for idx, idx2, rev in RevSampleTokens:
    temp_max=0
    max_word=''
    for word in rev:
        if word in word2vec:
            temp_sim = word2vec.similarity('service', word)
        if temp_sim>temp_max:
            temp_max=temp_sim
            max_word=word
    ServiceSimilarity.append((idx,  idx2, rev, temp_max, max_word))
    
end = time.time()


#### SORTING LIST #####

ServiceSimilarity.sort(key = operator.itemgetter(3), reverse=True)

del idx, idx2, word, rev
print("Word similarity processing time: ", end - start)


#### CREATING DICTIONARIES OF WORDS COUNTS AND SIMILARITIES - FOR LATER


start = time.time()
ServiceWords={}
for i in range(len(ServiceSimilarity)):
    if ServiceSimilarity[i][4] in ServiceWords.keys():
        ServiceWords[ServiceSimilarity[i][4]]+=1
    else:
        ServiceWords[ServiceSimilarity[i][4]]=1

ServiceWordsScore={}
for i in range(len(ServiceSimilarity)):
    if ServiceSimilarity[i][4] in ServiceWordsScore.keys():
        pass
    else:
        ServiceWordsScore[ServiceSimilarity[i][4]]=FoodSimilarity[i][3]

ServiceWordsScore_Sorted = sorted(ServiceWordsScore.items(), key=lambda kv: kv[1])
ServiceWordsScore_Sorted.reverse()
ServiceWordsCount_Sorted = sorted(ServiceWords.items(), key=lambda kv: kv[1])
ServiceWordsCount_Sorted.reverse()

end = time.time()

print("Dictionaries of counts and similarities creating time: ", end - start)

#### FEATURE CREATION - COUNT OF FOOD RELATED REVIEWS

ServiceRelatedCountRev={}
for i in range(len(ServiceSimilarity)):
    if ServiceSimilarity[i][3]>0.4:
        if ServiceSimilarity[i][0] not in ServiceRelatedCountRev.keys():
            ServiceRelatedCountRev[ServiceSimilarity[i][0]]=1 
        else:
            ServiceRelatedCountRev[ServiceSimilarity[i][0]]+=1
        


s = pd.Series(ServiceRelatedCountRev, name='ServiceRevs')
data['service_review_cnt'] = s

print('------------------------------------------------------------------------')
print('')
print('')
print('3. PRICE')

start = time.time()

PriceSimilarity=[]
for idx, idx2, rev in RevSampleTokens:
    temp_max=0
    max_word=''
    for word in rev:
        if word in word2vec:
            temp_sim = word2vec.similarity('money', word)
        if temp_sim>temp_max:
            temp_max=temp_sim
            max_word=word
    PriceSimilarity.append((idx,  idx2, rev, temp_max, max_word))
    
end = time.time()


#### SORTING LIST #####

PriceSimilarity.sort(key = operator.itemgetter(3), reverse=True)

del idx, idx2, word, rev
print("Word similarity processing time: ", end - start)



#### CREATING DICTIONARIES OF WORDS COUNTS AND SIMILARITIES - FOR LATER


start = time.time()
PriceWords={}
for i in range(len(PriceSimilarity)):
    if PriceSimilarity[i][4] in PriceWords.keys():
        PriceWords[PriceSimilarity[i][4]]+=1
    else:
        PriceWords[PriceSimilarity[i][4]]=1

PriceWordsScore={}
for i in range(len(PriceSimilarity)):
    if PriceSimilarity[i][4] in PriceWordsScore.keys():
        pass
    else:
        PriceWordsScore[PriceSimilarity[i][4]]=PriceSimilarity[i][3]

PriceWordsScore_Sorted = sorted(PriceWordsScore.items(), key=lambda kv: kv[1])
PriceWordsScore_Sorted.reverse()
PriceWordsCount_Sorted = sorted(PriceWords.items(), key=lambda kv: kv[1])
PriceWordsCount_Sorted.reverse()

end = time.time()

print("Dictionaries of counts and similarities creating time: ", end - start)


####FEATURE CREATION - COUNT OF FOOD RELATED REVIEWS

PriceRelatedCountRev={}
for i in range(len(PriceSimilarity)):
    if PriceSimilarity[i][3]>0.4:
        if PriceSimilarity[i][0] not in PriceRelatedCountRev.keys():
            PriceRelatedCountRev[PriceSimilarity[i][0]]=1 
        else:
            PriceRelatedCountRev[PriceSimilarity[i][0]]+=1

s = pd.Series(PriceRelatedCountRev, name='PriceRevs')
data['price_review_cnt'] = s

print('------------------------------------------------------------------------')
print('')
print('')
print('4. AMBIENCE')

##  SIMILAR WORDS

start = time.time()

AmbienceSimilarity=[]
for idx, idx2, rev in RevSampleTokens:
    temp_max=0
    max_word=''
    for word in rev:
        if word in word2vec:
            temp_sim = word2vec.similarity('ambience', word)
        if temp_sim>temp_max:
            temp_max=temp_sim
            max_word=word
    AmbienceSimilarity.append((idx,  idx2, rev, temp_max, max_word))
    
end = time.time()


#### SORTING LIST #####

AmbienceSimilarity.sort(key = operator.itemgetter(3), reverse=True)

del idx, idx2, word, rev
print("Word similarity processing time: ", end - start)


start = time.time()
AmbienceWords={}
for i in range(len(AmbienceSimilarity)):
    if AmbienceSimilarity[i][4] in AmbienceWords.keys():
        AmbienceWords[AmbienceSimilarity[i][4]]+=1
    else:
        AmbienceWords[AmbienceSimilarity[i][4]]=1

AmbienceWordsScore={}
for i in range(len(AmbienceSimilarity)):
    if AmbienceSimilarity[i][4] in AmbienceWordsScore.keys():
        pass
    else:
        AmbienceWordsScore[AmbienceSimilarity[i][4]]=AmbienceSimilarity[i][3]

AmbienceWordsScore_Sorted = sorted(AmbienceWordsScore.items(), key=lambda kv: kv[1])
AmbienceWordsScore_Sorted.reverse()
AmbienceWordsCount_Sorted = sorted(AmbienceWords.items(), key=lambda kv: kv[1])
AmbienceWordsCount_Sorted.reverse()

end = time.time()


AmbienceRelatedCountRev={}
for i in range(len(AmbienceSimilarity)):
    if AmbienceSimilarity[i][3]>0.4:
        if AmbienceSimilarity[i][0] not in AmbienceRelatedCountRev.keys():
            AmbienceRelatedCountRev[AmbienceSimilarity[i][0]]=1 
        else:
            AmbienceRelatedCountRev[AmbienceSimilarity[i][0]]+=1
        

#### FEATURE CERATION

s = pd.Series(AmbienceRelatedCountRev, name='AmbienceRevs')
data['ambience_review_cnt'] = s

print("Dictionaries of counts and similarities creating time: ", end - start)


print('FINISHED FEATURE CREATION')


'''
LATER:
    
    
    1)FURTHER USING OF CREATED TOPIC FEATURES
    
   1A)AVERAGING RATE PER TOPIC
    * for those reviews that related to given topic - avg rate
    * we should obtain that resturant with overall abg 4.8 
    
    
   1B) RATE TOPIC SPLIT
    * n - gram surrending food/service/local/price related reviews
    * sentiment analysis on them
    * substracting or adding +/-1 for section from main overall review
    * traingn word2vec model on ours reviews
    
    2)  IMPROVING TOPIC ASSIGMENT BY TRAINING ON BIGGER CORPORA
    * YELP Reviews

'''

del i, max_word, rest, temp_max, temp_sim, s, v, k, start, end, token
del FoodWordsScore, FoodWords
del AmbienceWordsScore, AmbienceWords, PriceWordsScore, PriceWords, ServiceWordsScore, ServiceWords 
del reviews


  
#        
#### duplicates removal
#from gensim.models.keyedvectors import KeyedVectors
##https://github.com/v1shwa/document-similarity
#        
#from DocSim import DocSim
#ds = DocSim(word2vec)  
#
#from gensim.test.utils import common_texts
#from gensim.models.doc2vec import Doc2Vec, TaggedDocument


#RevSampleTokens
#documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(RevSampleTokens[0][2])]
#model = Doc2Vec(documents, vector_size=5, window=2, min_count=1)




#start = time.time()
#reviews_all=[]
#reviews_tokenized=[]
#part_reviews = random.sample(review_list, k=100)
#len_rev=len(part_reviews)
#for restaurant_reviews in part_reviews:
#    for review in restaurant_reviews:
#        word_tokens = word_tokenize(review) 
#        filtered_sentence = [w for w in word_tokens if not w in set(stopwords.words('english'))] 
##        filtered_sentence = [] 
#        for w in word_tokens: 
#            w = porter.stem(w)
#            w = lemmatizer.lemmatize(w)
#            if w not in set(stopwords.words('english')) : 
#                filtered_sentence.append(w) 
#        reviews_tokenized.append(filtered_sentence)  
#    reviews_all.append(reviews_tokenized)
#end = time.time()
#print("Review tokenizatiom, lemmatization, stemming time for",len_rev,"resturants reviews is: ", end - start)
#
#del start, end, w, word_tokens, len_rev, filtered_sentence, review


#start = time.time()
#import itertools
#reviewFlatList = list(itertools.chain(*reviews_all))
#reviewFlatList2 = list(itertools.chain(*reviewFlatList))
#
#dictionary = gensim.corpora.Dictionary(reviewFlatList)
#bow_corpus = [dictionary.doc2bow(doc) for doc in reviewFlatList]
#end = time.time()
#
#print("Converting reviews to dictionary of words: ", end - start)
#
#
#start = time.time()
#
#lda_model =  gensim.models.LdaMulticore(bow_corpus, 
#                                   num_topics = 5, 
#                                   id2word = dictionary,                                    
#                                   passes = 10)
#
#
#end = time.time()

#model = word2vec.Word2Vec(review_list_sent, size=200)
        
        
        

#model = Word2Vec(size=300, min_count=20,
#                     window=2, iter=10)
#model.build_vocab(my_sentences)
#training_examples_count = model.corpus_count
## below line will make it 1, so saving it before
#model.build_vocab([list(word2vec.vocab.keys())], update=True)
#model.intersect_word2vec_format("GoogleNews-vectors-negative300.bin",binary=True, lockf=1.0)
#model.train(my_sentences,total_examples=training_examples_count, epochs=model.iter)

