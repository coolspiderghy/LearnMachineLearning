import scipy.io as sio
import numpy as np
matFileName = ['roi22_beta','rsFC_deg_bw_sparsity30','conn_FC_deg_bw_s30_2b_0b']
matrixName = ['roi22_beta','rsFC_deg_bw_Y_dep4ML','deg_bw_Y_dep4ML']
# get X,y
metricIndex = 0
rulename = 'activation'
matfile  = '/Users/genghaiyang/ghy_works/projects/stress_wm_networkanalysis/classification/'+ matFileName[metricIndex]+'.mat'
data = sio.loadmat(matfile)
X=data[matrixName[metricIndex]]
y=np.array([0]*18+[1]*18)
# define calculate score function
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
score = 'recall'#'precision'
clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                       scoring='%s_weighted' % score)
clf.fit(X_train, y_train)
print(clf.best_params_)
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))