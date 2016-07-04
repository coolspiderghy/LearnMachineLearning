import scipy.io as sio
import numpy as np
matFileName = ['roi22_beta','rsFC_deg_bw_sparsity30','conn_FC_deg_bw_s30_2b_0b']
matrixName = ['roi22_beta','rsFC_deg_bw_Y_dep4ML','deg_bw_Y_dep4ML']
rulenames = ['activation','rsFC','conn_FC']
# get X,y
metricIndex = 0
rulename =rulenames[metricIndex]
matfile  = '/Users/genghaiyang/ghy_works/projects/stress_wm_networkanalysis/classification/'+ matFileName[metricIndex]+'.mat'
data = sio.loadmat(matfile)
X=data[matrixName[metricIndex]]
y=np.array([0]*18+[1]*18)
# define calculate score function
def calculate_accuracy_score(X):
    from sklearn import svm
    from sklearn.cross_validation import LeaveOneOut
    loo = LeaveOneOut(36)
    y_pred = []
    y_true = []
    for train, test in loo:
        trainSetX,trainSetY = X[train],y[train]
        clf = svm.SVC(kernel= ['rbf','linear'][1], probability=False)
        clf.fit(trainSetX, trainSetY) 
        testSetX,testSetY = X[test],y[test]
        y_pred.append(clf.predict(testSetX))
        y_true.append(testSetY)
    from sklearn.metrics import accuracy_score
    #print clf.coef_
    return accuracy_score(y_true, y_pred)
#get predict accuracy score
dimx,dimy = X.shape
predict_accuracy_score = np.zeros(dimy)
for i in range(0,dimy):
    predict_accuracy_score[i] = calculate_accuracy_score(X[:,i:i+1])
predict_accuracy_score_sorted = sorted(list(set(predict_accuracy_score)),reverse=True)
#print calculate_accuracy_score(X[:,6:7]),'\n',X[:,6:7]
#get score index list in the order of descent of score.
score_index_list = []
for i in predict_accuracy_score_sorted:
#if i>0: 
    for indexiterm in np.where(predict_accuracy_score==i):
        score_index_list.append(indexiterm.tolist())                  
# get accuracy score of combined individual features with highest predict score           
import itertools 
score_index_list = list(itertools.chain(*score_index_list))
predict_accuracy_score_ms = np.zeros(len(score_index_list))#np.zeros((len(score_index_list),len(score_index_list)))
for i in range(0,len(score_index_list)):
    predict_accuracy_score_ms[i] = calculate_accuracy_score(X[:,score_index_list[0:i+1]])
#get roiname and accuracy for individual of each one and combined highest.
acc_score_ind = sorted(predict_accuracy_score,reverse = True)
roi_index_ind = score_index_list
acc_score_com = max(predict_accuracy_score_ms)
end_index = list(np.where(predict_accuracy_score_ms==max(predict_accuracy_score_ms))[0])[0]
roi_index_com = score_index_list[0:end_index+1]
#print acc_score_ind,'\n',roi_index_ind,'\n',acc_score_com,'\n',roi_index_com
rule=['activation','rsFC','conn_FC']
def indexmappingroiname(index,rule):
    index=index+1
    roinamelistfile = open('/Users/genghaiyang/ghy_works/projects/stress_wm_networkanalysis/roinamelist/conn_FC_ROINames_22.txt',"r+")
    roinamelist=[]
    for j in roinamelistfile.readlines():
        roinamelist.append(j.split()[0])
    roinamelistfile.close() 
    if rule=='activation':
        if index%2==0:
            roiindex =  int(index/2)
            brain_metric = '_'
            condition = '2b'
        elif index%2==1:
            roiindex =  int(index/2)+1
            brain_metric= '_'
            condition= '0b'
    elif rule=='rsFC':
        if index<=22:
            roiindex = index
            brain_metric = 'deg'
            condition = '_'
        else:
            roiindex = index-22
            brain_metric = 'bw'
            condition = '_'
    elif rule=='conn_FC':
        #print index
        if index<=44:
            if index%2==0:
                roiindex =  int(index/2)
                brain_metric = 'deg'
                condition = '0b'
            elif index%2==1:
                roiindex =  int(index/2)+1
                brain_metric = 'deg'
                condition = '2b'
        elif index>44:
            index = index-44
            if index%2==0:
                roiindex =  int(index/2)
                brain_metric = 'bw'
                condition = '0b'
            elif index%2==1:
                roiindex =  int(index/2)+1
                brain_metric = 'bw'
                condition = '2b'
    roiname  = roinamelist[roiindex-1]
    return roiname,condition,brain_metric
def write2txt(rule):
    roi_score_metrics_file = open(rule+'_'+'roi_score_condition_metrics.txt','w+')
    roi_score_metrics_file.write('---------This is individual one:-------------'+'\n')
    roi_score_metrics_file.write('ROI'+'\t'+'Accuracy'+'\t'+'Condition'+'\t'+'Brain_Metrics'+'\n')
    for rii in roi_index_ind:
        roi_score_metrics_file.write(str(indexmappingroiname(rii,rule)[0])+'\t'+str(acc_score_ind[roi_index_ind.index(rii)])+'\t'+str(indexmappingroiname(rii,rule)[1])+'\t'+str(indexmappingroiname(rii,rule)[2])+'\n')
    roi_score_metrics_file.write('---------This is combined one:-------------'+'\n')
    roi_score_metrics_file.write('ROI'+'\t'+'Accuracy'+'\t'+'Condition'+'\t'+'Brain_Metrics'+'\n')
    for rii in roi_index_com:
        roi_score_metrics_file.write(str(indexmappingroiname(rii,rule)[0])+'\t'+str(acc_score_com)+'\t'+str(indexmappingroiname(rii,rule)[1])+'\t'+str(indexmappingroiname(rii,rule)[2]+'\n'))
write2txt(rulename) 

print 'all done!!!' 
import matplotlib.pyplot as plt
plt.plot(predict_accuracy_score_ms)
plt.show()
#print predict_accuracy_score_ms,roi_index_com
#plot score line followed feature combined.

"""


for i in range(0,len(score_index_list)):
    for j in range(0,len(score_index_list)):
        if j>=i:
            predict_accuracy_score_ms[i,j] = calculate_accuracy_score(X[:,score_index_list[i:j+1]])
print predict_accuracy_score_ms
sio.savemat('predict_accuracy_score_ms.mat',{'predict_accuracy_score_ms':predict_accuracy_score_ms})

for i in predict_accuracy_score_sorted:
    if i>0:
        print 'score: %s, index: %s' %(i,np.where(predict_accuracy_score==i))

for i in predict_accuracy_score_sorted:
    if i>0:
        FC_index=[]
        for j in np.where(predict_accuracy_score==i):
            print type(j)
            for m in j:
                FC_index.append((int(m/22),int(m%22)))
        print 'score: %s, FC: %s, index: %s' %(i,FC_index,np.where(predict_accuracy_score==i))
"""
#print accuracy_score(y_true, y_pred, normalize=False)