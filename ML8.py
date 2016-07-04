　#Import Library

　　fromsklearn.ensemble importRandomForestClassifier

　　#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset

　　# Create Random Forest object

　　model=RandomForestClassifier()

　　# Train the model using the training sets and check score

　　model.fit(X,y)

　　#Predict Output

　　predicted=model.predict(x_test)

　　R 代码

　　library(randomForest)

　　x <-cbind(x_train,y_train)

　　# Fitting model

　　fit <-randomForest(Species~.,x,ntree=500)

　　summary(fit)

　　#Predict Output

　　predicted=predict(fit,x_test)