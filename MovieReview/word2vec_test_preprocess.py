#coding=utf-8
def word2vec_test():
    import pandas as pd
    import numpy as np
    import nltk
    #import sys
    #reload(sys)
    #sys.setdefaultencoding('utf8')
    def review_to_words(review,remove_stopwords=False):
        from bs4 import BeautifulSoup
        exa = BeautifulSoup(review,'lxml')
        exa  = exa.get_text()
        #remove number and putation?
        import re
        letter_only = re.sub("[^a-zA-Z]"," ",exa)
        # do tokenization
        lower_case = letter_only.lower()
        words  = lower_case.split()
        # remove stop word
        from nltk.corpus import stopwords
        #print stopwords.words('english')
        if remove_stopwords:
            stopwords = set(stopwords.words('english'))
            words = [w for w in words if not w in stopwords] 
        #去除标点符号
        remove_punc = False
        if remove_punc:
            english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
            words = [word for word in words if not word in english_punctuations]
        #词干化
        stemming = False
        if stemming:
            from nltk.stem.lancaster import LancasterStemmer
            st = LancasterStemmer()
            words = [st.stem(word) for word in words]
        #去除过低频词
        low_freq_filter = True
        word_min = 0
        if low_freq_filter:
            all_stems = words#sum(words, [])
            stems_once = set(stem for stem in set(all_stems) if all_stems.count(stem) < word_min) # this is the threshold of freq of word
            words = [word for word in words if word not in stems_once]
        else:
            words = words  
        # spell check
        spell_check =False
        if spell_check:
            import enchant
            d = enchant.Dict("en_US")
            words = [word for word in set(words) if d.check(word)]
        return( " ".join(words))
    def review_to_sentences(review,remove_stopwords=False):
        #nltk.download('punkt')
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        #print review.strip()
        raw_sentences = tokenizer.tokenize(review.decode('utf-8').strip())#
        sentences = []
        for raw_sentence in raw_sentences:
            #print type(raw_sentence)
            sentences.append(review_to_words(raw_sentence))
        return sentences
    '''
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)
    '''
    def sentences():
        #tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        train = pd.read_csv('labeledTrainData.tsv', header=0, 
        delimiter="\t", quoting=3)
        unlabeled_train = pd.read_csv('unlabeledTrainData.tsv', header=0, 
        delimiter="\t", quoting=3)
        sentences = []
        for review in train['review']:
            #print review
            sentences += review_to_sentences(review,remove_stopwords=False)
        for review in unlabeled_train['review']:
            sentences += review_to_sentences(review,remove_stopwords=False)
        return sentences
    def get_doc2vector_model(sentences):
        # Initialize and train the model (this will take some time)
        def labelizeReviews(reviews, label_type):
            from gensim.models.doc2vec import LabeledSentence
            labelized = []
            for i,v in enumerate(reviews):
                label = '%s_%s'%(label_type,i)
                labelized.append(LabeledSentence(v, [label]))
            return labelized
        from gensim.models import Doc2Vec
        print "Training model..."
        model_dm = Doc2Vec(min_count=40, window=10, size=300, sample=1e-3, negative=5, workers=3)
        # If you don't plan to train the model any further, calling 
        # init_sims will make the model much more memory-efficient.
        sentences = labelizeReviews(sentences,'train')
        model_dm.build_vocab(sentences)
        model_dm.train(sentences)
        # It can be helpful to create a meaningful model name and 
        # save the model for later use. You can load it later using Word2Vec.load()
        model_name = "300features_40minwords_10context_doc2vec_dm"
        model_dm.save(model_name)
    def get_word2vector_model(sentences):
        # Set values for various parameters
        num_features = 500    # Word vector dimensionality                      
        min_word_count = 40   # Minimum word count                        
        num_workers = 4       # Number of threads to run in parallel
        context = 10          # Context window size                                                                                    
        downsampling = 1e-3   # Downsample setting for frequent words
        
        # Initialize and train the model (this will take some time)
        from gensim.models import word2vec
        print "Training model..."
        model = word2vec.Word2Vec(sentences, workers=num_workers, \
                    size=num_features, min_count = min_word_count, \
                    window = context, sample = downsampling)
        
        # If you don't plan to train the model any further, calling 
        # init_sims will make the model much more memory-efficient.
        model.init_sims(replace=True)
        
        # It can be helpful to create a meaningful model name and 
        # save the model for later use. You can load it later using Word2Vec.load()
        model_name = "500features_40minwords_10context"
        model.save(model_name)
    def word_of_cluster():
        from gensim import models
        model = models.Word2Vec.load('400features_40minwords_10context')  
        from sklearn.cluster import KMeans
        n_dig = model.syn0.shape[0]/5
        #print model.syn0.shape
        words_vectors = model.syn0#[0:100,:]
        kmeans  = KMeans(n_clusters= n_dig)
        #print words_vectors.shape
        idx = kmeans.fit_predict(words_vectors) # return a word index belongs to which cluster
        # where is the word, this is saved in model.xx
        words = model.index2word#[0:100]
        words_idx = dict(zip(words,idx))
        return words_idx,n_dig
    def show_cluster(words_idx):
        for clusterindex in range(0,10):
            wordlist_cluster =[]
            for word in [keys for keys in words_idx if clusterindex==words_idx[keys]]:
                wordlist_cluster.append(word)
            print wordlist_cluster
    #unlabeled_train = pd.read_csv("unlabeledTrainData.tsv", header=0, 
    # delimiter="\t", quoting=3)
    #get_word2vector_model(sentences())
    #print "finish building word2vec model"
    """
    import cPickle as pickle 
    
    words_idx,n_dig = word_of_cluster()
    with open('words_idx.pickle', 'wb') as f: 
        pickle.dump((words_idx,n_dig), f, -1)
    
    with open('words_idx.pickle', 'rb') as f: 
        words_idx,n_dig = pickle.load(f)
    """   
    def review_to_wordofcluster(review,words_idx):
        wordlist = review_to_words(review)
        word_cluster_vector = np.zeros(n_dig)
        for word in wordlist:
            if word in words_idx:
                index = words_idx[word]
                word_cluster_vector[index] +=1 
        return word_cluster_vector       
    def get_data_features(datapath):     
        dataset = pd.read_csv(datapath, header=0, 
         delimiter="\t", quoting=3)
        #n_dig = train["review"].size # this is for bag of word
        #features = np.zeros((dataset['review'].size,n_dig),dtype = 'float32')
        reviews = []
        cnt = 0
        for review in dataset['review'][0:5000]:
            if((cnt+1)%1000 == 0):
                print 'finished preprocessing %s review' %(cnt+1)
            reviews.append(review_to_words(review)) 
            cnt =cnt+1
        """
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer()
        feature = vectorizer.fit_transform(reviews)
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        feature = vectorizer.fit_transform(reviews)
        return dataset,feature.toarray()     
    
    #_,train_features = get_data_features('labeledTrainData.tsv')
    
    """
    import cPickle as pickle 
    with open('train_features.pickle','wb') as f:
        pickle.dump((train_features),f,-1)
    
    with open('train_features.pickle','rb') as f:
        train_features = pickle.load(f)    
    train_features = pd.DataFrame(train_features)
    train_features = train_features.iloc[:1000,:10]
    print train_features.dtypes
    """
    #train_features[:1,0:10]

    #test,test_features = get_data_features('testData.tsv')
    #train model by RandomForest
    #from sklearn.ensemble import RandomForestClassifier as RF_clf
    #from sklearn import svm
    """
    print 'training model.........'
    from sklearn.linear_model import LogisticRegression#,SGDClassifier
    #from sklearn.naive_bayes import GaussianNB
    from sklearn import cross_validation
    X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(train_features,train['sentiment'],test_size = 0.3,random_state = 7)
    #model = RF_clf(n_estimators = 10)
    #model = svm.SVC(kernel= ['rbf','linear'][1], probability=False)
    #model = GaussianNB()
    model = LogisticRegression(penalty='l2', C=1, tol=0.001)
    #model = SGDClassifier()
    model.fit(X_train,Y_train)
    results = model.score(X_test,Y_test)
    print results*100
    
    from sklearn.grid_search import GridSearchCV
    tuned_parameters =[{'penalty': ['l1'], 'tol': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'penalty': ['l2'], 'tol':[1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}]
    print 'testing...........'
    grid =GridSearchCV(LogisticRegression(), tuned_parameters)#cv=5, scoring=['precision','recall']
    X = train_features
    Y = train['sentiment']
    grid.fit(X, Y)
    print(
        'The best parameters are {} with a score of {:0.2f}.'.format(
            grid.best_params_, grid.best_score_)
        )
    """
    """
    # THis is for k-fold cross validation
    X = train_features
    Y = train['sentiment']
    num_folds = 10
    num_instances = len(X)
    seed = 7
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    model = RF_clf(n_estimators = 10)
    results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
    print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)
    target = model.predict(test_features)
    results = pd.DataFrame(data = {'id':test['id'],'sentiment':target})
    results.to_csv('Bagofcluster.csv',index=False,quoting=3)

    import matplotlib.pyplot as plt
    from sklearn.metrics import auc,roc_curve
    pred_pro = model.predict_proba(X_test)[:,1]
    fpr,tpr,_ = roc_curve(Y_test,pred_pro)
    roc_auc = auc(fpr,tpr)
    print roc_auc
    plt.plot(fpr,tpr,label = 'area = %s' %(roc_auc))
    plt.plot([0,1],[0,1],'k--')
    plt.show()  
    """ 
if __name__ == '__main__':
    import time
    t1 = time.clock()
    word2vec_test()
    t2 = time.clock()
    print t2 - t1
    