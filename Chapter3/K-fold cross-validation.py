import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# data load
df=pd.read_csv('Chapter3/transfusion.data')
data=np.array(df)

# model construct
 # DataSet
def getDataSet(dataSet):
    """
    get watermelon data set UCI.
    :return:(feature array, label array)
    """
    # insert number 1 before colummn 0.
    # e.g: dataSet[0]=[1,0.697,0.460,1]
    dataSet = np.insert(dataSet,0,
                        np.ones(dataSet.shape[0]),
                        axis=1)
    dataArr = dataSet[:,:-1]
    labelArr = dataSet[:,-1]
    return dataArr, labelArr

 # gradDescent
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def gradDescent(dataArr, labelArr,alpha,T):
    """
    calculate logistic parameters by gradient descent method.
    :param dataArr: input data set with shape(m,n)
    :param labelArr: the label of data set with shape(m,1)
    :param alpha: step length (learning rate)
    :param T: iteration
    :return: parameters of gradient descent method.
    """
    m,n = dataArr.shape
    labelArr = labelArr.reshape(-1,1)
    errList=[]

    beta=np.ones((n,1))
    for t in range(T):
        py1=sigmoid(np.dot(dataArr,beta))
        dBetaMat=-dataArr*(labelArr-py1)
        # shape (1,n)
        dBeta=np.sum(dBetaMat,axis=0,keepdims=True)
        beta-=alpha*dBeta.T

        #test code
        pre=predict(beta,dataArr)
        errorRate = cntErrRate(pre,labelArr)
        errList.append(errorRate)

    return beta,errList

 # predict & cntErrRate
def predict(beta,dataArr):
    preArr=sigmoid(np.dot(dataArr,beta))
    preArr[preArr>0.5]=1
    preArr[preArr<0.5]=0

    return preArr

def cntErrRate(preLabel,label):
    """
    calculate error rate of predicted label by cnt method.
    :param preLabel: predict label
    :param label: real label
    :return: error rate
    """
    m=len(preLabel)
    cnt=0.0

    for i in range(m):
        if preLabel[i]!=label[i]:
            cnt+=1.0
    return cnt/float(m)

# K-fold cross-validation
# 随机打乱数据
np.random.seed(123)
rand_data=np.random.permutation(data)

# 分离label
X_v,y_v=getDataSet(rand_data)
X_v = (X_v - X_v.mean()) / X_v.std()

# 定义accuracy
def accuracy(pre,y):
    sum=0
    for i in range(len(y)):
        if pre[i]==y[i]:
            sum+=1
    return sum/len(pre)

# 10折交叉验证
def data_split_test(seq,X,y,show=1):
    sum1=0
    sum2=0
    ran=int(X.shape[0]/seq)
    for i in range(seq):
        rest_X=X[i*ran:(i+1)*ran]
        rest_y=y[i*ran:(i+1)*ran]
        temp_X=X[list(range(0,i*ran))+list(range((i+1)*ran,X.shape[0]))]
        temp_y=y[list(range(0,i*ran))+list(range((i+1)*ran,y.shape[0]))]
        beta,errGrad=gradDescent(temp_X,temp_y,0.001,1000)
        pre_train=predict(beta,temp_X)
        sum1+=accuracy(pre_train,temp_y)
        pre_test=predict(beta,rest_X)
        sum2+=accuracy(pre_test,rest_y)
        if(show):
            print("第{}折".format(i+1))
            print("gradient descent error rate is:",errGrad[-1])
            print("Train accuracy: {}%".format(100*accuracy(pre_train,temp_y)))
            print("Test accuracy: {}%".format(100*accuracy(pre_test,rest_y)))
    return sum1/seq,sum2/seq

 # main
def main():
    # 10 fold
    acc_train,acc_test=data_split_test(10,X_v,y_v)
    print("\n十折交叉验证后")
    print("Average Train accuracy is {}%".format(100*acc_train))
    print("Average Test accuracy is {}%".format(100*acc_test))

    #leave one out
    acc_train1,acc_test1 = data_split_test(X_v.shape[0], X_v, y_v,show=0)
    print("\n留一法")
    print("Total train accuracy is {}%".format(acc_train1 * 100))
    print("Total test accuracy is {}%".format(acc_test1 * 100))

if __name__ == '__main__':
    main()