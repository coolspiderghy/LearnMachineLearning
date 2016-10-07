Python 代码





- #Import Library

- from sklearn.linear_model import LogisticRegression

- #Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset

- # Create logistic regression object

- model = LogisticRegression()

- # Train the model using the training sets and check score

- model.fit(X, y)

- model.score(X, y)

- #Equation coefficient and Intercept

- print('Coefficient: \n', model.coef_)

- print('Intercept: \n', model.intercept_)

- #Predict Output

- predicted= model.predict(x_test)

R 代码





- x <- cbind(x_train,y_train)

- # Train the model using the training sets and check score

- logistic <- glm(y_train ~ ., data = x,family='binomial')

- summary(logistic)

- #Predict Output

- predicted= predict(logistic,x_test)