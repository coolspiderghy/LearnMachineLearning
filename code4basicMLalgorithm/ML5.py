　Python 代码

　　#Import Library

　　fromsklearn.naive_bayes importGaussianNB

　　#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset

　　# Create SVM classification object model = GaussianNB() # there is other distribution for multinomial classes like Bernoulli Naive Bayes, Refer link

　　# Train the model using the training sets and check score

　　model.fit(X,y)

　　#Predict Output

　　predicted=model.predict(x_test)

　　R 代码

　　library(e1071)

　　x <-cbind(x_train,y_train)

　　# Fitting model

　　fit <-naiveBayes(y_train ~.,data =x)

　　summary(fit)

　　#Predict Output

　　predicted=predict(fit,x_test)