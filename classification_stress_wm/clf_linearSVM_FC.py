import scipy.io as sio
import numpy as np
# get X,y
def get_X_y(metricIndex):
    matrixName = ['roi22_beta','rsFC_deg_bw_Y_dep4ML','deg_bw_Y_dep4ML','networkFC_noGM_all']
    matFileName = ['roi22_beta','rsFC_deg_bw_sparsity30','conn_FC_deg_bw_s30_2b_0b','networkFC_noGM_all']
    matfile  = matFileName[metricIndex]+'.mat'
    data = sio.loadmat(matfile)
    X=data[matrixName[metricIndex]]
    y=np.array([0]*18+[1]*18)
    return X,y
# define calculate score function
def calculate_weight_accuracyScore(X):
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
    #get_confusionMatrix(y_pred,y_true):
    pred_true = [(y_pred[i],y_true[i]) for i in range(0,len(y_pred))]
    tp,fn,fp,tn=0,0,0,0
    for pred,true in pred_true:
        if pred==1 and true==1:
            tp = tp+1
        elif pred==1 and true==0:
            fn = fn+1  
        elif pred==0 and true==1:
            fp = fp+1
        elif pred==0 and true==0:
            tn = tn+1
    conMatrix = np.array([[tp,fn],[fp,tn]])
    import blancedACC_p as bap
    bacc_mean = bap.bacc_mean(conMatrix)
    from sklearn.metrics import accuracy_score
    return clf.coef_,accuracy_score(y_true, y_pred),conMatrix,bacc_mean

"""
def calculate_ind_wei_AccScore(X):
    'get predict accuracy score'
    _,dimy = X.shape
    weight = np.zeros(dimy)
    predict_accuracy_score = np.zeros(dimy)
    for i in range(0,dimy):
        weight[i],predict_accuracy_score[i] = calculate_weight_accuracyScore(X[:,i:i+1])
    predict_accuracy_score_sorted = sorted(list(set(predict_accuracy_score)),key=abs,reverse=True)
    weight_sorted = sorted(list(set(weight)),key=abs,reverse=True)
    return weight,predict_accuracy_score,weight_sorted,predict_accuracy_score_sorted
"""
def calculate_ind_wei_AccScore(X):
    'get predict accuracy score'
    weight,predict_accuracy_score,_,_ = calculate_weight_accuracyScore(X)
    weight = weight[0]
    weight_sorted = sorted(weight,key=abs,reverse=True)
    return weight,weight_sorted,predict_accuracy_score

def get_indexList_des(orginalList,sortedList):
    'get score index list in the order of des of score.'
    indexList_des = []
    for i in sortedList:
        for indexiterm in np.where(orginalList==i):
            indexList_des.append(indexiterm.tolist())                           
    import itertools 
    indexList_des = list(itertools.chain(*indexList_des))
    return indexList_des

def calculate_com_wei_AccScore(X,indexList_des):
    'get accuracy score of combined individual features with highest predict score'  
    weight_ms = [0]*len(indexList_des)#np.zeros(len(indexList_des))
    predict_accuracy_score_ms = np.zeros(len(indexList_des))#np.zeros((len(score_index_list),len(score_index_list)))
    conMatrix_ms = [0]*len(indexList_des)
    bacc_mean_ms = np.zeros(len(indexList_des))
    for i in range(0,len(indexList_des)):
        weight_ms[i],predict_accuracy_score_ms[i],conMatrix_ms[i] ,bacc_mean_ms[i]=calculate_weight_accuracyScore(X[:,indexList_des[0:i+1]])
    #end_index = list(np.where(predict_accuracy_score_ms==max(predict_accuracy_score_ms))[0])[0]
    #print conMatrix_ms
    #conMatrix_ms = conMatrix_ms[end_index][0]
    #print indexList_des,'\n',bacc_mean_ms
    return weight_ms,predict_accuracy_score_ms,conMatrix_ms,bacc_mean_ms
def index2FC(index,rule):
    index=index+1
    roinamelistfile = open('/Users/genghaiyang/ghy_works/projects/stress_wm_networkanalysis/roinamelist/conn_FC_ROINames_22.txt',"r+")
    roinamelist=[]
    for j in roinamelistfile.readlines():
        roinamelist.append(j.split()[0])
    roinamelistfile.close() 
    if rule=='activation':
        if index%2==0:
            roiindex,brain_metric,condition =  int(index/2),'_','2b'
        elif index%2==1:
            roiindex,brain_metric,condition =  int(index/2)+1,'_','0b'
    elif rule=='rsFC':
        if index<=22:
            roiindex,brain_metric,condition = index,'deg','_'
        else:
            roiindex,brain_metric,condition = index-22,'bw','_'
    elif rule=='conn_FC':
        roiindex = index
        condition = '_'
        brain_metric = '_'
        """
        if index<=44:
            if index%2==0:
                roiindex,brain_metric,condition =  int(index/2),'deg','0b'
            elif index%2==1:
                roiindex,brain_metric,condition =  int(index/2)+1,'deg','2b'
        elif index>44:
            index = index-44
            if index%2==0:
                roiindex,brain_metric,condition =  int(index/2),'bw','0b'
            elif index%2==1:
                roiindex,brain_metric,condition =  int(index/2)+1,'bw','2b'
        """
    elif rule =='networkFC':
        FCName = ['fpn','fsn','dmn','fpn_fsn','fpn_dmn','fsn_dmn']
        if index%2==0:
            roiindex,brain_metric,condition =  int(index/2),'networkFC','0b'
        elif index%2==1:
            roiindex,brain_metric,condition =  int(index/2)+1,'networkFC','2b'
    
    roiname  = FCName[roiindex-1]#roinamelist[roiindex-1]
    return roiname,condition,brain_metric
def get_ind_com_sum_table(X,rule):
    import blancedACC_p as bap
    #get roiname and accuracy for individual of each one and combined highest.
    weight_ind,sorted_weight_ind,predict_accuracy_score_ind= calculate_ind_wei_AccScore(X)
    weight_indexList_des_ind= get_indexList_des(weight_ind,sorted_weight_ind)
    weight_ms,_,conMatrix_ms,bacc_mean_ms = calculate_com_wei_AccScore(X,weight_indexList_des_ind)
    #print weight_ms
    roi_index_ind = weight_indexList_des_ind
    acc_score_ind = predict_accuracy_score_ind
    weight_ind = weight_ind

    def get_endindex_max(column):
        end_index = list(np.where(column==max(column))[0])[0]
        column_max_com = max(column)
        return end_index,column_max_com
    end_index,column_max_com = get_endindex_max(bacc_mean_ms)  
    weight_com = weight_ms = weight_ms[end_index][0]
    roi_index_com = roi_index_ind[:end_index+1]
    roi_score_metrics_file = open(rule+'_'+'roi_score_condition_metrics.txt','w+')
    bacc_mean_ms_file = open(rule+'_'+'bacc_mean_ms.txt','w+')
    roi_score_metrics_file.write('---------This is individual one:-------------'+'\n')
    roi_score_metrics_file.write('ROI'+'\t'+'Accuracy'+'\t'+'weight'+'\t'+'Condition'+'\t'+'Brain_Metrics'+'\n')
    for rii in roi_index_ind:
        roi_score_metrics_file.write(str(index2FC(rii,rule)[0])+'_'+str(index2FC(rii,rule)[1])+'\t'+str(acc_score_ind)+'\t'+str(weight_ind[rii])+'\t'+str(index2FC(rii,rule)[1])+'\t'+str(index2FC(rii,rule)[2])+'\n')
    roi_score_metrics_file.write('---------This is combined one:-------------'+'\n')
    roi_score_metrics_file.write('ROI'+'\t'+'Accuracy'+'\t'+'weight'+'\t'+'Condition'+'\t'+'Brain_Metrics'+'\n')
    for rii in roi_index_com:
        roi_score_metrics_file.write(str(index2FC(rii,rule)[0])+'\t'+str(column_max_com)+'\t'+str(weight_com[roi_index_com.index(rii)])+'\t'+str(conMatrix_ms[end_index])+'\t'+str(bap.bacc_p(conMatrix_ms[end_index]))+'\t'+str(bacc_mean_ms[end_index])+'\t'+str(index2FC(rii,rule)[1])+'\t'+str(index2FC(rii,rule)[2]+'\n'))
    roi_score_metrics_file.close()
    bacc_mean_ms_file.write(str(bacc_mean_ms))
    bacc_mean_ms_file.close()
    print bacc_mean_ms[end_index],len(roi_index_com),bacc_mean_ms[end_index]/len(roi_index_com)
    #print bacc_mean_ms
    import matplotlib.pyplot as plt
    plt.plot(bacc_mean_ms)
    plt.savefig(rulenames[metricIndex]+'.jpg')  
    #print bacc_mean_ms[0:end_index+1]
    #plt.hist(bacc_mean_ms[0:end_index+1], bins = len(bacc_mean_ms[0:end_index+1]), facecolor='blue', alpha=0.5)  
    #plt.show()
#initialize variables
if __name__ =='__main__':
    rulenames = ['activation','rsFC','conn_FC','networkFC']
    metricIndex = 3
    X,y=get_X_y(metricIndex)
    get_ind_com_sum_table(X,rulenames[metricIndex])
    print 'all done!!!' 