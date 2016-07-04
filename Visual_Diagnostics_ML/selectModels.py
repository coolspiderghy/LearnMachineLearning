import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

concrete  = pd.read_excel(os.path.join('data','concrete.xls'))
concrete.columns = [
    'cement', 'slag', 'ash', 'water', 'splast',
    'coarse', 'fine', 'age', 'strength'
]

credit = os.path.join('data','credit.xls')
credit = pd.read_excel(credit, header=1)
credit.columns = [
    'id', 'limit', 'sex', 'edu', 'married', 'age', 'apr_delay', 'may_delay',
    'jun_delay', 'jul_delay', 'aug_delay', 'sep_delay', 'apr_bill', 'may_bill',
    'jun_bill', 'jul_bill', 'aug_bill', 'sep_bill', 'apr_pay', 'may_pay', 'jun_pay',
    'jul_pay', 'aug_pay', 'sep_pay', 'default'
]
occupancy = os.path.join('data','occupancy_data','datatraining.txt')
occupancy = pd.read_csv(occupancy, sep=',')
occupancy.columns = [
    'date', 'temp', 'humid', 'light', 'co2', 'hratio', 'occupied'
]
#print len(occupancy) # 8,143
#print len(credit)    # 30,000
#print len(concrete)  # 1,030

from sklearn.preprocessing import scale
from sklearn.svm import LinearSVC
from sklearn import cross_validation as cv
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
#from matplotlib import colors
from matplotlib.colors import ListedColormap
    
ddl_heat = ['#DBDBDB','#DCD5CC','#DCCEBE','#DDC8AF','#DEC2A0','#DEBB91',\
                       '#DFB583','#DFAE74','#E0A865','#E1A256','#E19B48','#E29539']
ddlheatmap = ListedColormap(ddl_heat)
def plot_classification_report(cr, title=None, cmap=ddlheatmap):     

    title = title or 'Classification report'
    lines = cr.split('\n')
    classes = []
    matrix = []

    for line in lines[2:(len(lines)-3)]:
        s = line.split()
        classes.append(s[0])
        value = [float(x) for x in s[1: len(s) - 1]]
        matrix.append(value)

    fig, ax = plt.subplots(1)

    for column in range(len(matrix)+1):
        for row in range(len(classes)):
            txt = matrix[row][column]
            ax.text(column,row,matrix[row][column],va='center',ha='center')

    fig = plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(len(classes)+1)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.ylabel('Classes')
    plt.xlabel('Measures')
    plt.show()
from sklearn.linear_model import Ridge, Lasso, ElasticNet
def get_preds(attributes, targets, model):
    splits = cv.train_test_split(attributes, targets, test_size=0.2)
    X_train, X_test, y_train, y_test = splits

    model.fit(X_train, y_train)
    y_true = y_test
    y_pred = model.predict(X_test)
    return y_true,y_pred

# Divide data frame into features and labels
features = occupancy[['temp', 'humid', 'light', 'co2', 'hratio']]
labels   = occupancy['occupied']

# Scale the features
stdfeatures = scale(features)

"""  

y_list  = [classify(stdfeatures, labels, LinearSVC()),classify(stdfeatures, labels, KNeighborsClassifier())]
for y_true,y_pred in y_list:
    print(metrics.confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))#, target_names=target_names
    cr = classification_report(y_true, y_pred)
    plot_classification_report(cr)
"""
"""
# classify credit
features = credit[[
    'limit', 'sex', 'edu', 'married', 'age', 'apr_delay', 'may_delay',
    'jun_delay', 'jul_delay', 'aug_delay', 'sep_delay', 'apr_bill', 'may_bill',
    'jun_bill', 'jul_bill', 'aug_bill', 'sep_bill', 'apr_pay', 'may_pay',
    'jun_pay', 'jul_pay', 'aug_pay', 'sep_pay'
]]
labels   = credit['default']
#print features.head(5)
stdfeatures = scale(features)

classify(stdfeatures, labels, LinearSVC())
classify(stdfeatures, labels, KNeighborsClassifier())
"""

"""
features = concrete[[
    'cement', 'slag', 'ash', 'water', 'splast', 'coarse', 'fine', 'age'
]]
labels   = concrete['strength']
"""
"""
y_list = [regress(features, labels, Ridge()),regress(features, labels, Lasso()),regress(features, labels, ElasticNet())]
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
for y_true,y_pred in y_list:
    print('Mean squared error = {:0.3f}'.format(mse(y_true, y_pred)))
    print('R2 score = {:0.3f}'.format(r2_score(y_true, y_pred)))
"""
def roc_compare_two(y, yhats, models):
    from sklearn.metrics import roc_curve,auc
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    for ys,yhat, m, ax in ((y[0],yhats[0], models[0], ax1), (y[1],yhats[1], models[1], ax2)):
        false_positive_rate, true_positive_rate, thresholds = roc_curve(ys,yhat)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        ax.set_title('ROC for %s' % m)
        ax.plot(false_positive_rate, true_positive_rate, \
                c='#2B94E9', label='AUC = %0.2f'% roc_auc)
        ax.legend(loc='lower right')
        ax.plot([0,1],[0,1],'m--',c='#666666')
    plt.xlim([0,1])
    plt.ylim([0,1.1])
    plt.show()
"""
#plot ROC curve to validate the performance of classifier 
y_true_svc, y_pred_svc = get_preds(stdfeatures, labels, LinearSVC())
y_true_knn, y_pred_knn = get_preds(stdfeatures, labels, KNeighborsClassifier())
actuals = np.array([y_true_svc,y_true_knn])
predictions = np.array([y_pred_svc,y_pred_knn])
models = ['LinearSVC','KNeighborsClassifier']
roc_compare_two(actuals, predictions, models)
"""
# using prediction error plot to evaluate performance
def error_compare_three(mods,X,y):
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
    for mod, ax in ((mods[0], ax1),(mods[1], ax2),(mods[2], ax3)):
        predicted = cv.cross_val_predict(mod[0], X, y, cv=12)
        ax.scatter(y, predicted, c='#F2BE2C')
        ax.set_title('Prediction Error for %s' % mod[1])
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4, c='#2B94E9')
        ax.set_ylabel('Predicted')
    plt.xlabel('Measured')
    plt.show()
#from sklearn.svm import SVR
#from sklearn.linear_model import RANSACRegressor
#models = np.array([(Ridge(),'Ridge'), (SVR(),'SVR'), (RANSACRegressor(),'RANSAC')])
#error_compare_three(models, features, labels)
def resids_compare_three(mods,X,y):
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
    plt.title('Plotting residuals using training (blue) and test (green) data')
    for m, ax in ((mods[0], ax1),(mods[1], ax2),(mods[2], ax3)):
        for feature in list(X):
            splits = cv.train_test_split(X[[feature]], y, test_size=0.2)
            X_tn, X_tt, y_tn, y_tt = splits
            m[0].fit(X_tn, y_tn)
            ax.scatter(m[0].predict(X_tn),m[0].predict(X_tn)-y_tn,c='#2B94E9',s=40,alpha=0.5)
            ax.scatter(m[0].predict(X_tt), m[0].predict(X_tt)-y_tt,c='#94BA65',s=40)
        ax.hlines(y=0, xmin=0, xmax=100)
        ax.set_title(m[1])
        ax.set_ylabel('Residuals')
    plt.xlim([20,70])        # Adjust according to your dataset
    plt.ylim([-50,50])  
    plt.show()
#from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.learning_curve import validation_curve 
#models = np.array([(Ridge(),'Ridge'), (LinearRegression(),'Linear Regression'), (SVR(),'SVR')])
#resids_compare_three(models, features, labels)
def plot_val_curve(features, labels, model):
    p_range = np.logspace(-5, 5, 5)

    train_scores, test_scores = validation_curve(
        model, features, labels, param_name='gamma', param_range=p_range,
        cv=6, scoring='accuracy', n_jobs=1
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title('Validation Curve')
    plt.xlabel('$\gamma$')
    plt.ylabel('Score')
    plt.semilogx(p_range, train_scores_mean, label='Training score', color='#E29539')
    plt.semilogx(p_range, test_scores_mean, label='Cross-validation score', color='#94BA65')
    plt.legend(loc='best')
    plt.show()

#X = scale(credit[['limit','sex','edu','married','age','apr_delay']])
#y = credit['default']
#X = stdfeatures
#y = labels
#plot_val_curve(X, y, SVC())

from sklearn.grid_search import GridSearchCV

def blind_gridsearch(model, X, y):
    C_range = np.logspace(-2, 10, 5)
    gamma_range = np.logspace(-5, 5, 5)
    param_grid = dict(gamma=gamma_range, C=C_range)
    grid = GridSearchCV(SVC(), param_grid=param_grid)
    grid.fit(X, y)

    print(
        'The best parameters are {} with a score of {:0.2f}.'.format(
            grid.best_params_, grid.best_score_
        )
    )
#features = credit[['limit','sex','edu','married','age','apr_delay']]
#labels   = credit['default']
#blind_gridsearch(SVC(), stdfeatures, labels)
def visual_gridsearch(model, X, y):
    C_range = np.logspace(-2, 10, 5)
    gamma_range = np.logspace(-5, 5, 5)
    param_grid = dict(gamma=gamma_range, C=C_range)
    grid = GridSearchCV(SVC(), param_grid=param_grid)
    grid.fit(X, y)

    scores = [x[1] for x in grid.grid_scores_]
    scores = np.array(scores).reshape(len(C_range), len(gamma_range))

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=ddlheatmap)
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title(
        "The best parameters are {} with a score of {:0.2f}.".format(
        grid.best_params_, grid.best_score_)
    )
    plt.show()
visual_gridsearch(SVC(), stdfeatures, labels)


