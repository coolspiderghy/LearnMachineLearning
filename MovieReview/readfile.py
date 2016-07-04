import pandas as pd
#load data
train = pd.read_csv('labeledTrainData.tsv',sep='\t')
# using BS4 to remove html Mark
def review_to_words(review):
    from bs4 import BeautifulSoup
    exa = BeautifulSoup(review,'lxml')
    exa  = exa.get_text()
    #remove number and putation?
    import re
    letter_only = re.sub("[^a-zA-Z]"," ",exa)
    # do tokenization
    lower_case = letter_only.lower()
    words  = lower_case.split()
    #import nltk
    #nltk.download('stopwords')
    from nltk.corpus import stopwords
    #print stopwords.words('english')
    stopwords = set(stopwords.words('english'))
    words = [w for w in words if not w in stopwords]    
    return ' '.join(words)
# how many steps I can use to do preprocessing of text?
num_reviews = train['review'].size
clean_train_review = []
for i in xrange(0,num_reviews):
    clean_train_review.append(review_to_words(train['review'][i]))
    if((i+1)%1000 ==0):
        print "review:"+str(i+1)
        
print "print bag of words:"

# how to do bag of words by using sklearn
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer='word',
                             tokenizer=None,
                             preprocessor=None,
                             stop_words = None,
                             max_features=5000)
train_data_features = vectorizer.fit_transform(clean_train_review)
train_data_features = train_data_features.toarray()
#vocab = vectorizer.get_feature_names()
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100)
forest  = forest.fit(train_data_features,train['sentiment'])

# load test data and predict, save the results
test = pd.read_csv('testData.tsv',sep='\t')
num_reviews = test['review'].size
clean_test_review = []
for i in xrange(0,num_reviews):
    clean_test_review.append(review_to_words(test['review'][i]))
    if((i+1)%1000 ==0):
        print "review:"+str(i+1)
        
vectorizer = CountVectorizer(analyzer='word',
                             tokenizer=None,
                             preprocessor=None,
                             stop_words = None,
                             max_features=5000)
test_data_features = vectorizer.transform(clean_test_review) # what is the difference between fit_transform and transform?? what underlying this function
test_data_features = test_data_features.toarray()  #why switch to array??
result = forest.predict(test_data_features)
output = pd.DataFrame(data = {'id':test['id'],'sentiment':result})
output.to_csv('Bag_of_Words_model.csv',index=False,quoting=3)

