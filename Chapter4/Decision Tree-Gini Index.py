import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import operator

# 绘图函数
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

# 生成函数
# 特征字典
featureDic={
'色泽': ['浅白', '青绿', '乌黑'],
    '根蒂': ['硬挺', '蜷缩', '稍蜷'],
    '敲声': ['沉闷', '浊响', '清脆'],
    '纹理': ['清晰', '模糊', '稍糊'],
    '脐部': ['凹陷', '平坦', '稍凹'],
    '触感': ['硬滑', '软粘']
}

def getDataSet():
    """
    get watermelon dataset 2.0
    :return:
        newDataSet: 整体数据集
        trainDataSet: 训练集
        pruneDataSet: 剪枝集 
        features: 特征变量
    """
    dataSet=[
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '好瓜'],
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '好瓜'],
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '坏瓜'],
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '坏瓜'],
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '坏瓜'],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜']
    ]

    features=['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']

    # 每种特征的属性个数
    numList=[len(featureDic[key]) for key in features]

    newDataSet=np.array(dataSet)
    # 训练集
    trainIndex=[0,1,2,5,6,9,13,14,15,16]
    trainDataSet=newDataSet[trainIndex]
    # 剪枝集
    pruneIndex=[3,4,7,8,10,11,12]
    pruneDataSet=newDataSet[pruneIndex]

    return newDataSet,trainDataSet,pruneDataSet,features

def calGini(dataArr):
    """
    calculate information entropy.
    :param dataArr:
    :param classArr:
    :return: Gini
    """









