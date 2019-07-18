#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import warnings
warnings.filterwarnings('ignore')
#os.chdir('/Users/gosc/Desktop/Kaggle/Zomato - Bangalore') 


# ## Read data

# In[7]:


### import libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
### text processing
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#import geopy
#from geopy import geocoders
#from geopy.geocoders import GoogleV3


# In[8]:


# Options for pandas
pd.options.display.max_columns = 30

# lib for pandas profiling
import pandas_profiling


# In[9]:


#### read data
data=pd.read_csv('zomato.csv',encoding='latin-1')
data.head()


# In[10]:


print(len(data))


# In[11]:


#data= data.drop_duplicates(subset='name',keep='first')


# In[12]:


for col in data.columns:
    dtype = str(data[col].dtype)
    print(col, dtype)


# #### Number of Nulls

# In[13]:


print('Number of nulls:')
round((data.isnull() | data.isna()).sum()/len(data),3)


# In[14]:


#data[(data.menu_item.notnull()) & ~(data.menu_item.isna())]


# #### Number of Unique Values

# In[15]:


print('Number of unique values in the data')
print(data.nunique())


# In[ ]:





# In[16]:


# Generate data report using pandas profile analysis
profile = data.profile_report(title='Zomato profile analysis report')
profile.to_file(output_file="zomato data analysis.html")


# # Data understanding & Preprocessing

# #### Dropping some columns

# In[17]:


data.drop(['url','phone'],inplace=True,axis=1)


# #### Renaming columns

# In[18]:


data.rename(columns=
        {'name':'rest_name','approx_cost(for_two_people)': 'meal_cost', 'listed_in(city)':
         'Neighbourhood','listed_in(type)': 'restaurant_type','rest_type':'restaurant_category'}, inplace=True)


# ### Preprocessing 

# In[19]:


data.rate = data.rate.astype(str).apply(lambda x: x.replace('/5',''))
data.rate = data.rate.astype(str).apply(lambda x: x.replace(',/n',''))
data.loc[(data.rate =='NEW') | (data.rate =='-'), 'rate'] = np.nan
data.rate = data.rate.apply(lambda x: float(x))


# In[20]:


data['meal_cost'] = data['meal_cost'].str.replace(',', '').astype(float)

data['votes'] = data['votes'].astype(float)

data['rest_name'] = data['rest_name'].str.strip()

data['rest_name'] = data['rest_name'].replace(['[^A-Za-z0-9_.,!"\s]+'], [''], regex=True)


# In[21]:


data.columns.values


# # Variable enginering
# 
# #MAy be number of dishesh liked # There is a mistake as nan has been counted as one
# #All the dishes liked are distinct and unique.

# In[22]:


data['num_dishes_liked'] = data['dish_liked'].astype('str').apply(lambda x: len(x.split(',') if x!=np.nan else 0))


# In[23]:


from collections import Counter

z =[]


for i in data['cuisines'].astype('str').apply(lambda x: x.split(',')):
    for j in i:
        z.append(j.strip())
# create a dictionary based on values in list
cusine_dict = Counter(z)

# sort dictonary based on values of dictionary
cusine_dict=sorted(cusine_dict.items(), key=lambda x: x[1], reverse=True)


# # Create boolean variables with cuisine type

# #List for all the categories created

# In[24]:


cuisine_type=['north indian','chinese','south indian','continental','cafe','fast food','beverages','italian','american','desserts']


# #Convert all the cuisine types into lower charatcers
# #Remove all the leading and trailing spaces from strings

# In[25]:


data['cuisines']=data['cuisines'].astype('str').apply(lambda x: x.lower())
data['cuisines']=data['cuisines'].astype('str').apply(lambda x: x.strip())


# #replace all the categories which can be converted into meaningful categories
# #This categories have been created based on knowledge of food

# In[26]:


data['cuisines']=data['cuisines'].replace({'pizza':'italian'}, regex= True)

data['cuisines']=data['cuisines'].replace({'bakery':'cafe'}, regex= True)

data['cuisines']=data['cuisines'].replace({'coffee':'cafe'}, regex= True)

data['cuisines']=data['cuisines'].replace({'ice cream':'desserts'}, regex= True)

data['cuisines']=data['cuisines'].replace({'street food':'fast food'}, regex= True)

data['cuisines']=data['cuisines'].replace({'andhra':'south indian'}, regex= True)

data['cuisines']=data['cuisines'].replace({'kerala':'south indian'}, regex= True)

data['cuisines']=data['cuisines'].replace({'biryani':'south indian'}, regex= True)

data['cuisines']=data['cuisines'].replace({'mughlai':'north indian'}, regex= True)

data['cuisines']=data['cuisines'].replace({'bihari':'north indian'}, regex= True)

data['cuisines']=data['cuisines'].replace({'rajasthani':'north indian'}, regex= True)

data['cuisines']=data['cuisines'].replace({'bengali':'north indian'}, regex= True)

data['cuisines']=data['cuisines'].replace({'salad':'continental'}, regex= True)

data['cuisines']=data['cuisines'].replace({'juices':'beverages'}, regex= True)

data['cuisines']=data['cuisines'].replace({'mithai':'desserts'}, regex= True)

data['cuisines']=data['cuisines'].replace({'mangalorean':'south indian'}, regex= True)

data['cuisines']=data['cuisines'].replace({'burger':'american'}, regex= True)
data['cuisines']=data['cuisines'].replace({'sandwich':'american'}, regex= True)


# In[27]:


# some how not working
#data_cusine['cusine']=data_cusine['cusine'].replace({"eastasian","chinese"}, regex= True)
#data_cusine['cusine']=data_cusine['cusine'].replace({'east asian':'thai'}, regex= True)
#data_cusine['cusine']=data_cusine['cusine'].replace({'east asian':'asian'}, regex= True)
#data_cusine['cusine']=data_cusine['cusine'].replace({'east asian':'pan asian'}, regex= True)


# # Code to create boolean variables

# #Create dummy variables based on each string in variables cuisines.

# In[28]:


temp=data.cuisines.str.get_dummies(sep=', ')


# #Create list of cuisines which are not part of cuisines category created.
# #We consider category which is not part of cuisine_type is rest_cuisines.

# In[29]:


not_in_list = [col for col in temp.columns if col not in cuisine_type]


# In[30]:


temp['rest_cusines']= temp[not_in_list].max(1)


# In[31]:


cuisine_type.append('rest_cusines') # append list with rest_cuisine category


# #reindex the dataframe temp with only categories we want from cuisine_type

# In[32]:


temp= temp.reindex(cuisine_type, axis=1, fill_value=0)


# #concat the dataframes to get a single dataframe

# In[33]:


data= pd.concat((data, temp), axis=1)


# # Variable votes based on intervals

# In[34]:


data['votes_new']= pd.qcut(data['votes'],4, labels = False)


# In[35]:


data['votes_new'].value_counts()


# In[36]:


#data.drop(['votes'],axis=1, inplace = True)


# ## EDA & Plotting

# ### Feature analysis

# #### Number of records per feature

# In[37]:


def countplot(data, feature,limit=30, size=(16,4)):
    plt.figure(figsize=size)
    sns.countplot(data[feature], 
    palette='GnBu_d',order = data[feature].value_counts().head(limit).index)
    plt.xticks(rotation=90)


# 30 Neihbourhoods with the highest number of reviews.

# In[38]:


countplot(data, 'Neighbourhood')


# 10 types of restaurants with the highest number of reviews

# In[39]:


countplot(data, 'location')


# In[40]:


countplot(data, 'restaurant_category', 10, (8,4))


# Restaurants by Category 

# In[41]:


countplot(data, 'restaurant_type',10,(8,4))


# Restaurants by cuisine

# In[42]:


countplot(data, 'cuisines')


# In[43]:


countplot(data, 'book_table',10,(4,4))


# In[44]:


countplot(data, 'online_order',10,(4,4))


# #### Number of Votes per feature

# In[45]:


def VotesCount(feature, limit=50, size=(12, 6)):
    group = data['votes'].groupby(data[feature])
    sum_vote = group.sum().sort_values(ascending=False)
    sum_vote.head(limit).plot(kind='bar', figsize=size, color='indianred', alpha=0.8);


# In[46]:


VotesCount('Neighbourhood')


# In[47]:


VotesCount('location', 30)


# In[48]:


VotesCount('restaurant_category',40)
# This feature could be splittted


# In[49]:


VotesCount('restaurant_type',20, (8, 4))


# In[50]:


VotesCount('cuisines', 20, (12, 6))
# It could be splitted


# In[51]:


VotesCount('book_table',2,(6, 5))


# In[52]:


VotesCount('online_order',2,(6, 5))


# #### Splitting cusine and restaurant category to visualisise

# In[53]:


color_map=['#FF9AA2','#FFDAC1', '#F1F0CF','#d6f8e9','#B5EAD7','#D3EEFF','#C1BBDD','#DCFFFB','#ADE6D0']


# In[54]:


color_map2 = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']


# In[55]:


def StackBarVotes(level_1, level_2, Agg, color=color_map, limit=25):
    group=data.groupby([level_1,level_2])
    g= group.sum()[Agg]
    g = g.unstack()
    g['sumval']=g.sum(axis = 1, skipna = True)
    g=g.sort_values('sumval', ascending=False).head(limit)
    g = g.drop(['sumval'], axis=1)
    g.plot(kind='bar', figsize=(10, 4), color=color, stacked=True)


# In[56]:


StackBarVotes('Neighbourhood','restaurant_type','votes')


# In[57]:


StackBarVotes('Neighbourhood','online_order','votes')


# In[58]:


StackBarVotes('Neighbourhood','book_table','votes')


# ## Not finished 
# Here I was trying to create sth that cuts groups with less than 5% numver of observation in each 

# In[59]:


def StackBarVotes2(level_1, level_2, Agg, perc=5, colors="tab20c"):
    group=data.groupby([level_1,level_2]).agg({Agg: ['sum']})
    g=group.reset_index()
    gperc = group.groupby(level=0).apply(lambda x:100 * x / float(x.sum()))
    gpercf= gperc[gperc > perc].reset_index().dropna()
    gg = g.iloc[gpercf.index]
    k=gg.set_index([level_1, level_2]).sort_index(level=[level_1, level_2])
    ki=k.unstack()
    ki['sumval']=ki.sum(axis = 1, skipna = True)
    ki=ki.sort_values('sumval', ascending=False)
    ki = ki.drop(['sumval'], axis=1)
    ki.plot(kind='bar', figsize=(16, 10), cmap=colors, stacked=True)


# In[60]:


StackBarVotes2('Neighbourhood','restaurant_category','votes')


# ### Carefully
# 
# This below does not make any sense cause we missing biggest neigbourhoods because of divison

# In[61]:


#StackBarVotes2('Neighbourhood','cuisines','votes',5,colors="tab20b")
#requires fixing wrong labels - sth to do with indexing


# In[62]:


#proof
group=data.groupby(['Neighbourhood','cuisines']).agg({'votes': ['sum']})
g=group.reset_index()
gperc = group.groupby(level=0).apply(lambda x:100 * x / float(x.sum()))
gpercf= gperc[gperc > 5].reset_index().dropna()
#gpercf


# #### Cusinine Dict
# 
# I have decided to manipulate with this feature to represent more accurately.

# In[63]:


cuisine_list={}
for index, row in data.iterrows():
    #print(type(row['cuisines']), type(row['votes']))
    tokens = row['cuisines']
    for t in str(tokens).split(','):
        if t in cuisine_list.keys():
            cuisine_list[t] += row['votes']
        else:
            cuisine_list[t] = row['votes']


# In[64]:


lists = sorted(cuisine_list.items(), key=lambda item: item[1], reverse=True)
x, y = zip(*lists) # unpack a list of pairs into two tuples
fig= plt.figure(figsize=(12,8))
plt.bar(x[:30],y[:30])
plt.xticks(rotation=90)
plt.show()


# In[65]:


#for key, value in sorted(cuisine_list.items(), key=lambda item: item[1], reverse=True ):
#    print("%s: %s" % (key, value))


# ### Rate's analysis

# #### One feature analysis

# ##### 10 localization with the higher avg rate

# In[66]:


data_agg=data.groupby('Neighbourhood')['rate'].agg(np.mean).sort_values(ascending=False)[:10]
data_agg


# In[67]:


data_agg=data.groupby('Neighbourhood')['votes'].agg(np.sum).sort_values(ascending=False)[:10]
data_agg


# Church street has the highest avg. rate but has less votes than Koramangala's block that are also in top. Diferences beetwen avg. rating per localization are not high

# ##### 10 restaurant types with the higher avg rate

# In[68]:


data.groupby('restaurant_type')['rate'].agg(np.mean).sort_values(ascending=False)[:10]


# In[69]:


data.groupby('restaurant_type')['votes'].agg(np.sum).sort_values(ascending=False)[:10]


# Pubs and bars has less votes but o avg they are higher. The most represented category Deliveries has the lowest rating.

# ##### 15 restaurant categories with the higher avg rate

# In[70]:


data_agg=data.groupby('restaurant_category')['rate'].agg(np.mean).sort_values(ascending=False)[:15]
data_agg


# In[71]:


data.groupby('restaurant_category')['votes'].agg(np.sum).sort_values(ascending=False)[:15]


# In[72]:


df_agg = data.groupby(['restaurant_type']).agg({'rate':'mean', 'votes':'sum'}).sort_values('votes',ascending=False)
df_agg


# In[73]:


df_agg = data.groupby(['restaurant_category']).agg({'rate':'mean', 'votes':'sum'}).sort_values('votes',ascending=False)
df_agg


# #### Overlaps?

# In[74]:


df_agg = data.groupby(['restaurant_type','restaurant_category']).agg({'rate':'mean', 'votes':'sum'}).sort_values('votes',ascending=False)
df_agg


# In[75]:


data_agg=data.groupby('restaurant_type')['rate'].agg(np.mean).sort_values(ascending=False)[:10]
data_agg


# #### The highest scored restuarant

# In[76]:


data.sort_values('rate',ascending=False)[['rest_name','Neighbourhood','rate']].head(20).drop_duplicates()


# In[77]:


grouped = data.groupby('Neighbourhood')['rate'].agg([min, max, np.mean]) 
grouped


# ### Scatterplot

# In[78]:


def scatterplot(x, y, hue, data, h=13.7, w=10.27):
    sns.set(rc={'figure.figsize':((h,w))})
    sns.scatterplot(x=x, y=y, hue=hue,
                palette='Set3', data=data)


# #### One level aggregations

# In[79]:


df_by_niegbourhood = data.groupby('Neighbourhood').agg({'rate':'median','votes':'sum'}).reset_index()
df_by_location = data.groupby('location').agg({'rate':'median','votes':'sum'}).reset_index()
df_by_restcat = data.groupby('restaurant_category').agg({'rate':'median','votes':'sum'}).reset_index()
df_by_restcat = data.groupby('restaurant_type').agg({'rate':'median','votes':'sum'}).reset_index()


# In[80]:


scatterplot(x="rate", y="votes", hue="Neighbourhood", data=df_by_niegbourhood)


# ### 2 level aggregatios

# In[81]:


df_by_locneigh = data.groupby(['Neighbourhood','restaurant_type']).agg({'rate':'median','votes':'sum'}).reset_index()


# In[82]:


g = sns.set(rc={'figure.figsize':((14,8))})
g = sns.scatterplot(x="Neighbourhood", y="rate", size="votes", hue='restaurant_type',
                palette='Set3', data=df_by_locneigh[df_by_locneigh['votes']>10000])
plt.xticks(rotation=45)
plt.show()


# In[83]:


#data.sort_values('rate',ascending=False)[['name','Cusine','rate']].head(20).drop_duplicates()


# ### TEXT PROCESSING

# #### DISH LIKED

# Spliting text columns

# In[84]:


words_list=[]
for dish in data['dish_liked']:
    dish = str(dish)
    tokens = dish.split()
    for t in tokens:
        t = t.replace(',', '')
        if t != 'nan':
            words_list.append(t)


# In[85]:


DishedDF=pd.DataFrame(words_list)
DishedDF[0].value_counts().head(5)


# In[86]:


type(words_list)


# In[87]:


def WordCount(WordsList, limit=30):
    plt.figure(figsize=(10,4))
    sns.countplot(pd.DataFrame(WordsList)[0], 
    palette='Set3',order = pd.DataFrame(WordsList)[0].value_counts().head(limit).index)
    plt.xticks(rotation=90)


# In[88]:


WordCount(words_list, 15)


# In[89]:


def wordCloud(data, backgroundcolor="white", wordmax=80, sizefont=20, color='steelblue'):
    wc = WordCloud(background_color=backgroundcolor, 
                   max_words=wordmax, max_font_size=sizefont, scale=10, contour_color=color)
    wc.generate(' '.join(data))
    plt.figure()
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()


# In[90]:


wordCloud(words_list)


# In[91]:


d=data['menu_item']


# In[92]:


menu=[]
for dish in d:
    dish = str(dish)
    dish = dish.replace('\\', '')
    dish = dish.replace('[', '')
    dish = dish.replace(']', '')
    tokens = dish.split("', '")
    for t in tokens:
        t = t.replace(',', '')
        if len(t)>0:
            menu.append(t)


# In[93]:


menudf=pd.DataFrame(menu)
menudf[0].value_counts().head(5)


# In[94]:


WordCount(menudf, 15)


# In[95]:


wordCloud(menu)


# #### Reviews

# In[96]:


data.reviews_list.head(5)


# Splitting columns

# In[97]:


reviews=data.reviews_list.copy()
sentences_tokens = []
for i in range(len(reviews[:100])):#first 100
    review = reviews[i]
    review = re.sub(r'\\n|\\x|\.', ' ', review)
    tokens = re.sub("[^a-z ]+", "", review.lower())
    tokens = tokens.split()
    tokens = [w for w in tokens if (w not in set(stopwords.words('english')) and (w!='rated'))]
    sentences_tokens.extend(tokens)
        
print(sentences_tokens) 


# In[98]:


len(sentences_tokens)


# First 100 reviews

# In[99]:


wordCloud(sentences_tokens)


# In[100]:


WordCount(sentences_tokens, 15)


# In[101]:


# Creating the Bag of Words model using CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(sentences_tokens).toarray()
y = data.iloc[:, 1].values


# ### This will be expanded later 

# ### Rate distribution

# In[102]:


plt.figure(figsize=(12,4))
plt.title("Distribution")
sns.distplot(data['rate'],color="red", kde=True,bins=30, label='Rate')
plt.legend()
plt.show()


# In[103]:


sns.kdeplot(data['rate'], shade=True)


# ### Cost of dish

# In[104]:


data['meal_cost'].head() # needs preprocessing


# In[105]:


#data.meal_cost = data.meal_cost.astype(str).apply(lambda x: x.replace(',',''))
#data.meal_cost = data.meal_cost.apply(lambda x: float(x))


# In[106]:


sns.kdeplot(data['meal_cost'], shade=True)


# In[107]:


data.meal_cost.describe()


# #### Removing abnormalities
# 
# Assuming that this high valueof order is an error

# In[108]:


AbnormValue=data.meal_cost.mean()+data.meal_cost.std()*3
final_list = [x for x in data.meal_cost if (x >AbnormValue)]
final_set = set(final_list) 
unique_list = (list(final_set))


# In[109]:


data.loc[data.meal_cost.isin(unique_list)] = np.nan


# In[110]:


sns.kdeplot(data['meal_cost'], shade=True)


# ### Is there any use of address?
# 
# Plotting?

# In[111]:


address=[]
d = data.address
for i in d:
    t = i.split("', '")
    address.append(t)
data.address[1]


# In[ ]:


#g = geocoders.GoogleV3(api_key='AIzaSyB0W6vk0fR6D864bFHF0WrDNCgfSSTdMUw')
#location = g.geocode("175 5th Avenue NYC", timeout=10)


# https://github.com/geopy/geopy/issues/171
# 
# https://gis.stackexchange.com/questions/198530/plotting-us-cities-on-a-map-with-matplotlib-and-basemap/198570

# In[ ]:





# In[112]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()


# In[120]:


analyser.polarity_scores('A beautiful place to dine in.The interiors take you back to the Mughal era')


# In[137]:


data['reviews_list'].fillna(' ',inplace = True)


# In[ ]:


r'\\n|\\x|\.', ' ', review)


# In[140]:


data['reviews_list'] =data['reviews_list'].apply(lambda x:x.replace('\\n|\\x|\.', ' '))


# In[152]:


data['reviews_list']=data['reviews_list'].apply(lambda x: re.sub(r'\\n|\\x|\.', ' ',x.lower()))


# In[169]:


data['reviews_list']=data['reviews_list'].apply(lambda x: ''.join(i for i in x if not i.isdigit()))


# In[170]:


data['reviews_list']=data['reviews_list'].apply(lambda x: re.sub("[^a-z ]+", "", x))


# In[173]:


data['reviews_list']=data['reviews_list'].apply(lambda x: re.sub("rated", "", x))


# In[193]:


for i in range(100):
    print(analyser.polarity_scores(data['reviews_list'][i]))


# In[187]:


data['reviews_list'][0]


# In[191]:





# In[ ]:




