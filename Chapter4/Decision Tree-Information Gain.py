import numpy as np
import matplotlib.pyplot as plt
from pylab import *

# 定义特征字典
featureDic = {
    '色泽': ['浅白', '青绿', '乌黑'],
    '根蒂': ['硬挺', '蜷缩', '稍蜷'],
    '敲声': ['沉闷', '浊响', '清脆'],
    '纹理': ['清晰', '模糊', '稍糊'],
    '脐部': ['凹陷', '平坦', '稍凹'],
    '触感': ['硬滑', '软粘']
}


# 数据集获取
def getDataSet():
    """
    get watermelon data 3.0
    :return: 编码好的数据集以及特征字典
    """
    dataSet = [
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, 1],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, 1],
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, 1],
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, 1],
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, 1],
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, 1],
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, 1],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, 1],
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, 0],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, 0],
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, 0],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, 0],
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, 0],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, 0],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370, 0],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, 0],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, 0]
    ]

    features = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖量']

    # 生成列表，表示每个分类型变量的类别数
    numList=[]
    for i in range(len(features)-2):
        numList.append(len(featureDic[features[i]]))

    # encoding: ->1,2,3
    newDataSet=[]
    for dataVec in dataSet: # 每次选择一条数据
        dataNum = dataVec[-3:] #把数值型变量摘出
        newData=[] # 创建新的数据条
        for i in range(len(dataVec)-3): # 遍历这一个数据条中的所有类别型变量
            for j in range(numList[i]): # 判断第 i个特征是否是第 j+1类
                if dataVec[i]==featureDic[features[i]][j]:
                    newData.append(j+1)
        newData.extend(dataNum)
        newDataSet.append(newData)

    return np.array(newDataSet),features

def calEntropy(dataArr,classArr):
    """
    calculate information entropy
    :param dataArr:
    :param classArr:
    :return: entropy
    """

    n=dataArr.size
    data0=dataArr[classArr==0]
    data1=dataArr[classArr==1]
    p0=data0.size/float(n)
    p1=data1.size/float(n)
    # 约定 plogp=0,如果p=0
    if p0==0:
        ent=-(p1*np.log2(p1))
    elif p1==0:
        ent=-(p0*np.log2(p0))
    else:
        ent=-p0*np.log2(p0) - p1*np.log2(p1)

    return ent

def splitDataSet(dataSet,ax,value):
    """
    按照给定的属性ax和ax的取值value把数据划分
    当属性是分类型变量时，返回一个属性值都为value的数据集
    当属性是数值型变量时，以value的大小为标准返回二分数据集

    :param dataSet: 输入数据集，形状为(m,n)表示m个数据，前n-1列为属性，最后一列为类型
    :param ax: 第ax个属性
    :param value: 分类型变量取1，2，3；数值型变量取某一实数

    :return:
        * 分类型变量dataset返回第ax个属性中值为value组成的集合
        * 数值型变量返回根据value的值二分的集合
    """
    # 2个数值型属性，前边n-3列为分类型属性
    if ax<dataSet.shape[1]-3:
        dataS=np.delete(dataSet[dataSet[:,ax]==value],ax,axis=1)
        return dataS
    else:
        dataL=dataSet[dataSet[:,ax]<=value]
        dataR=dataSet[dataSet[:,ax]>value]
        return dataL,dataR

def  calInfGain(dataSet,labelList,ax,value=-1):
    """
    计算给定的数据集dataSet在属性ax上的信息增益

    :param dataSet: 输入数据集，形状为(m,n)，表示有m条数据，前n-1列为属性，最后一列是类型
    :param labelList:属性列表，表示dataSet中的属性都有哪些
    :param ax:选择用来计算信息增益的属性，从0开始计数，最后两个为数值型，其他为分类型
    :param value:用来划分数据的值，默认-1，表示不使用这个参数

    :return:信息增益
    """

    # 计算划分前信息熵
    baseEnt= calEntropy(dataSet[:,:-1],dataSet[:,-1])

    # 计算根据属性ax划分后的信息熵
    newEnt=0
    # 分类型和数值型属性分别计算
    if ax<dataSet.shape[1]-3:
        num=len(featureDic[labelList[ax]])
        for j in range(num):
            subDataSet=splitDataSet(dataSet,ax,j+1)
            prob=len(subDataSet)/float(len(dataSet))
            if prob!=0:
                newEnt+=prob*calEntropy(subDataSet[:,:-1],subDataSet[:,-1])
    else:
        # 将数据集二分
        dataL,dataR=splitDataSet(dataSet,ax,value)
        # 计算二分后的信息熵
        entL=calEntropy(dataL[:,:-1],dataL[:,-1])
        entR=calEntropy(dataR[:,:-1],dataR[:,-1])
        # 计算划分后总的信息熵
        newEnt=(dataL.size*entL+dataR.size*entR)/float(dataSet.size)

    # 计算信息增益
    gain=baseEnt-newEnt
    return gain

"""
check it

data,feat=getDataSet()
print(calInfGain(data,feat,2))
## 0.14078
"""

def chooseBestSplit(dataSet,labelList):
    """
    计算信息增益最大的划分方式，如果最佳划分方式不是根据数值型属性，划分值bestthresh=-1
    :param dataSet: 当前数据集
    :param labelList: 当前属性列表
    :return:
        bestFeature: 得到的最大增益对应的属性(0,1,,,,)取值为索引
        bestThresh: 使得到最大增益划分时所取得划分值value,若是分类型变量则取-1
        maxGain: 最大增益划分时的增益值
    """

    maxGain=0
    bestFeature=0
    bestThresh=-1
    m,n=dataSet.shape
    # 遍历每一个属性
    for i in range(n-1):
        if i < (n-3): #先考虑分类型属性
            gain=calInfGain(dataSet,labelList,i)
            if gain > maxGain:
                bestFeature=i
                maxGain=gain
        else: # 考虑数值型属性
            featvals=dataSet[:,i] #得到第i个特征的所有取值
            sortedFeat=np.sort(featvals)
            T=[]
            # 计算划分点
            for j in range(m-1):
                t=(sortedFeat[j]+sortedFeat[j+1])/2.0
                T.append(t)
            # 对每一个划分点计算信息增益
            for t in T:
                gain=calInfGain(dataSet,labelList,i,t)
                if gain > maxGain:
                    bestFeature=i
                    bestThresh=t
                    maxGain=gain

    return bestFeature,bestThresh,maxGain


"""
# check it

data, feat = getDataSet()
f, tv, g = chooseBestSplit(data, feat)
print(f"best feature is {list(featureDic.keys())[f]}\n" 
        f"best thresh value is {tv}\n"
        f"max information gain is {g}")
# best feature is 纹理
# best thresh value is -1
# max information gain is 0.3805918973682686
"""

def majorityCnt(classList):
    """
    投票确定某一属性某一取值最终对应的类别，0多则返回‘坏瓜’，1多则返回‘好瓜’

    :param classList: 该叶节点下数据集的类别列表
    :return: 分类
    """
    cnt0=len(classList[classList==0])
    cnt1=len(classList[classList==1])
    if cnt0>cnt1:
        return '坏瓜'
    else:
        return '好瓜'

def createTree(dataSet,labels):
    """
    创建决策树-base on Information Gain

    :param dataSet:
    :param labels:
    :return: 返回一个保存有树的字典
    """
    classList=dataSet[:,-1]
    # 如果剩余的类别全部相同，则返回
    if len(classList[classList==classList[0]])==len(classList):
        if classList[0]==0:
            return '坏瓜'
        else:
            return '好瓜'

    #如果只剩下类标签,投票返回
    if len(dataSet[0])==1:
        return majorityCnt(classList)

    # 得到信息增益最大划分的属性与取值
    bestFeat,bestVal,entGain=chooseBestSplit(dataSet,labels)
    bestFeatLabel=labels[bestFeat]

    if bestVal!=-1: #如果选出的是数值型
        txt=bestFeatLabel+"<="+str(bestVal)+"?"
    else: #如果是分类型
        txt=bestFeatLabel+"="+"?"

    # 创建字典
    myTree={txt:{}}
    if bestVal!=-1: # 数值型变量就是父节点下有两个子节点
        subDataL,subDataR=splitDataSet(dataSet,bestFeat,bestVal)
        myTree[txt]['是']=createTree(subDataL,labels)
        myTree[txt]['否']=createTree(subDataR,labels)
    else: # 如果父节点是分类型变量
        i=0
        # 子节点的属性中删去父节点属性
        del(labels[bestFeat])
        uniqueVals=featureDic[bestFeatLabel] # 父节点属性可取哪些类别
        for value in uniqueVals:
            subLabels=labels[:]
            i+=1
            subDataSet=splitDataSet(dataSet,bestFeat,i)
            myTree[txt][value]=createTree(subDataSet,subLabels)

    return myTree


# check it
data, feat = getDataSet()
Tree=createTree(data,feat)
print(Tree)


"""
根据保存的决策树进行绘图
"""

# 定义文本框和箭头格式
decisionNode=dict(boxstyle="sawtooth",fc="0.8")
leafNode=dict(boxstyle="round4",fc="0.8")
arrow_args=dict(arrowstyle="<-")
mpl.rcParams['font.sans-serif']=['SimHei']

def plotMidText(cntrPt,parentPt,txtString):
    """
    绘制文字在节点之间
    :param cntrPt: 当前节点坐标
    :param parentPt: 父节点坐标
    :param txtString: 需要绘制的文字内容
    :return:
    """
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, fontsize=20)

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    """
    在图中绘制节点，并用箭头指向父节点。
    :param nodeTxt: 节点显示的文字
    :param centerPt: 当前节点的位置
    :param parentPt: 父节点的位置
    :param nodeType: 节点的样式(decisionNode 或 leafNode)
    :return:
    """
    createPlot.ax1.annotate(nodeTxt,
                            xy=parentPt,
                            xycoords="axes fraction",
                            xytext=centerPt,
                            textcoords="axes fraction",
                            va="center",
                            ha="center",
                            bbox=nodeType,
                            arrowprops=arrow_args,
                            fontsize=20)


def getNumLeafs(myTree):
    """
    获取叶节点数,计算树的宽度，决定绘图时节点的横坐标分布
    :param myTree:
    :return: 叶节点的数量
    """
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    """
    获取树的最大深度(层数)
    :param myTree:
    :return:
    """
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

def plotTree(myTree,parentPt,nodeTxt):
    """
    绘制整棵树
    :param myTree:
    :param parentPt:
    :param nodeTxt:
    :return:
    """
    numLeafs = getNumLeafs(myTree)
    getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW,
              plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff),
                     cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD



def createPlot(inTree):
    fig = plt.figure(1, figsize=(600, 30), facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


def main():
    dataSet,labelList=getDataSet()
    myTree=createTree(dataSet,labelList)
    createPlot(myTree)

if __name__ == '__main__':
    main()
















