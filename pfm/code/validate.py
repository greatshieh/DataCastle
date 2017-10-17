
# coding: utf-8


#pandas, numpy, matplotlib, seaborn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
try:
    from jupyterthemes import jtplot
    #jtplot.style(theme='grade3')
except ModuleNotFoundError:
    pass
import numpy as np
import time


#sklearn API
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc

#定义交叉验证函数
def classification_func(model, X, Y, CV=5, method=1, scoring=None):
    '''
    参数解释:
    ====================
    model: 分类器
    X: 输入的feature
    Y: 输入的target
    CV: 交叉验证次数
    method: 交叉验证的方法
            1：cross_val_score,默认
            2：cross_validate
    scoring: 评估方式,准确率/精度/召回率/F1
    '''
    #用cross_validate
    #cross_validate和cross_val_score的区别是
    #1. 可以定义多个指标
    #2. 返回一个dict

    if method == 'cross_validate':
        score = cross_validate(model,X,Y,scoring=scoring, cv=CV)
        print('='*60)
        for x in score.keys():
            print('the %s is %.4f'%(x,score[x].mean()))
            print('='*60)

    #默认用cross_val_score
    else:
        #判断是否为list
        if isinstance(scoring, list):
            print('='*60)
            for x in range(len(scoring)):
                score = cross_val_score(model, X, Y,
                                        scoring=scoring[x], cv=CV)
                print('the %s score is %.4f(+/-%.4f)'\
                     %(scoring[x], score.mean(), score.std()*2))
                print('='*60)
        #判断是否为字符串
        elif isinstance(scoring, str):
            score = cross_val_score(model, X, Y, scoring=scoring, cv=CV)
            print('the %s score is %.4f(+/-%.4f)'\
                 %(scoring, score.mean(), score.std()*2))
        #默认用准确度评估
        else:
            score = cross_val_score(model, X, Y, scoring=scoring, cv=CV)
            print('the accuracy is %.4f(+/-%.4f)'%(score.mean(),score.std()*2))




def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 20),
                        verbose=0, ax=None):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，
         其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if ax == None:
        plt.figure(figsize=(15,7))
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std,
                         alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std,
                         alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        #plt.show()
    else:
        ax.set_title(title)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel("Training examples")
        ax.set_ylabel("Score")
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std,
                         alpha=0.1, color="r")
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std,
                         alpha=0.1, color="g")
        ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        ax.legend(loc="best")

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1])\
              + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1])\
         - (test_scores_mean[-1] - test_scores_std[-1])

    

    return midpoint, diff


def plot_pr_curve(estimator, X, y, test_size=0.3, ax=None):
    """
    画出data在某模型上的P-R曲线.
    参数解释
    ----------
    estimator : 你用的分类器。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=0)
    estimator.fit(X_train, Y_train)
    if hasattr(estimator, 'decision_function'):
        y_score = estimator.decision_function(X_test)
    else:
        y_score = estimator.predict_proba(X_test)[:,1]
    average_precision = average_precision_score(Y_test, y_score)
    precision, recall, threshold = precision_recall_curve(Y_test, y_score)

    if ax == None:
        plt.figure(figsize=(15,7))
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid()

        plt.step(recall, precision, color='b', alpha=0.5, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.title('Precision-Recall curve: AUC={0:0.2f}'\
                  .format(average_precision))
    else:
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.grid()

        ax.step(recall, precision, color='b', alpha=0.5, where='post')
        ax.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        ax.set_title('Precision-Recall curve: AUC={0:0.2f}'\
                      .format(average_precision))

    return


def plot_roc_curve(estimator, X, y, test_size=0.3, ax=None):
    """
    画出data在某模型上的ROC曲线.
    参数解释
    ----------
    estimator : 你用的分类器。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，
         其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """

    X_train, X_test, Y_train, Y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=0)
    estimator.fit(X_train, Y_train)
    if hasattr(estimator, 'decision_function'):
        y_score = estimator.decision_function(X_test)
    else:
        y_score = estimator.predict_proba(X_test)[:,1]
    roc_score = roc_auc_score(Y_test, y_score)
    fpr, tpr, threshold = roc_curve(Y_test, y_score)
    #roc_score = auc(fpr, tpr)

    if ax == None:
        plt.figure(figsize=(15,7))
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid()

        plt.step(fpr, tpr, color='b', alpha=0.5, where='post')
        plt.fill_between(fpr, tpr, step='post', alpha=0.2, color='b')
        #plt.ylim([0.0, 1.05])
        #plt.xlim([0.0, 1.0])
        plt.title('ROC curve: AUC={0:0.2f}'.format(roc_score))
    else:
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.grid()

        ax.step(fpr, tpr, color='b', alpha=0.5, where='post')
        ax.fill_between(fpr, tpr, step='post', alpha=0.2, color='b')
        #plt.ylim([0.0, 1.05])
        #plt.xlim([0.0, 1.0])
        ax.set_title('ROC curve: AUC={0:0.2f}'.format(roc_score))

    return

print('123')
