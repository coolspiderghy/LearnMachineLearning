　Python 代码

　　#Import Library

　　fromsklearn.cluster importKMeans

　　#Assumed you have, X (attributes) for training data set and x_test(attributes) of test_dataset

　　# Create KNeighbors classifier object model

　　k_means =KMeans(n_clusters=3,random_state=0)

　　# Train the model using the training sets and check score

　　model.fit(X)

　　#Predict Output

　　predicted=model.predict(x_test)

　　R 代码

　　library(cluster)

　　fit <-kmeans(X,3)# 5 cluster solution