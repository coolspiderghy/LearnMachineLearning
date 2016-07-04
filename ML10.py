#Import Library

　　fromsklearn.ensemble importGradientBoostingClassifier

　　#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset

　　# Create Gradient Boosting Classifier object

　　model=GradientBoostingClassifier(n_estimators=100,learning_rate=1.0,max_depth=1,random_state=0)

　　# Train the model using the training sets and check score

　　model.fit(X,y)

　　#Predict Output

　　predicted=model.predict(x_test)

　　R 代码

　　library(caret)

　　x <-cbind(x_train,y_train)

　　# Fitting model

　　fitControl <-trainControl(method ="repeatedcv",number =4,repeats =4)

　　fit <-train(y ~.,data =x,method ="gbm",trControl =fitControl,verbose =FALSE)

　　predicted=predict(fit,x_test,type="prob")[,2]