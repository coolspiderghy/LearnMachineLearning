Python 代码

　　#Import Library

　　fromsklearn.neighbors importKNeighborsClassifier

　　#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset

　　# Create KNeighbors classifier object model

　　KNeighborsClassifier(n_neighbors=6)# default value for n_neighbors is 5

　　# Train the model using the training sets and check score

　　model.fit(X,y)

　　#Predict Output

　　predicted=model.predict(x_test)

　　R 代码

　　library(knn)

　　x <-cbind(x_train,y_train)

　　# Fitting model

　　fit <-knn(y_train ~.,data =x,k=5)

　　summary(fit)

　　#Predict Output

　　predicted=predict(fit,x_test)