'''导入库'''
import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import pprint

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# import validate as va

df_train = pd.read_csv('./dataset/pfm_train.csv')
df_test = pd.read_csv('./dataset/pfm_test.csv')

# 前文分析过，两个变量方差为0，可以删除。
# EmployeeNumber是唯一识别号码，删除
df_train.drop(
    ['Over18', 'StandardHours', 'EmployeeNumber'], axis=1, inplace=True)
df_test.drop(
    ['Over18', 'StandardHours', 'EmployeeNumber'], axis=1, inplace=True)

# 预测变量
target_var = 'Attrition'

# 字符型
character_var = [
    x for x in df_train.dtypes.index if df_train.dtypes[x] == 'object'
]
numeric_var = [
    x for x in df_train.dtypes.index
    if x != target_var and x not in character_var
]
# 将数值型变量标准化
#scaler = MinMaxScaler()
#pattern = scaler.fit(df_train[numeric_var])
#df_train[numeric_var] = scaler.transform(df_train[numeric_var])
#df_test[numeric_var] = scaler.transform(df_test[numeric_var])

df_train = pd.get_dummies(df_train)
df_test = pd.get_dummies(df_test)

predictor = [x for x in df_train.columns if x != target_var]

'''
def select_kbest_clf(data_frame, target, k=2):
    """
    Selecting K-Best features for classification
    :param data_frame: A pandas dataFrame with the training data
    :param target: target variable name in DataFrame
    :param k: desired number of features from the data
    :returns feature_scores: scores for each feature in the data as 
    pandas DataFrame
    """
    feat_selector = GenericUnivariateSelect(f_classif, mode='fdr')
    _ = feat_selector.fit(data_frame.drop(target, axis=1), data_frame[target])

    feat_scores = pd.DataFrame()
    feat_scores["F Score"] = feat_selector.scores_
    feat_scores["P Value"] = feat_selector.pvalues_
    feat_scores["Support"] = feat_selector.get_support()
    feat_scores["Attribute"] = data_frame.drop(target, axis=1).columns

    return feat_scores


kbest_feat = select_kbest_clf(df_train, target_var, k=len(predictor))
kbest_feat = kbest_feat.sort_values(
    ["F Score", "P Value"], ascending=[False, False])
pprint.pprint(kbest_feat)

predictor = kbest_feat[kbest_feat['Support'] == True]['Attribute'].tolist()

pprint.pprint(predictor)
'''
validation_size = 0.3
seed = 7
scoring = 'accuracy'
X_train, X_test, y_train, y_test = train_test_split(
    df_train[predictor],
    df_train[target_var],
    test_size=validation_size,
    random_state=seed)
kfold = KFold(n_splits=10, random_state=seed)

model = LogisticRegression()
parameters = {
    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.1, 1],
    'tol': [1e-6, 1e-5, 1e-4],
    'random_state': [1, 2, 3, 4, 5]
}

cv_results = cross_val_score(
    model, X_train, y_train, cv=kfold, scoring=scoring)
msg = "原始模型交叉验证分数: %f (%f)" % (cv_results.mean(), cv_results.std())
print(msg)

grid_search = GridSearchCV(
    estimator=model, param_grid=parameters, scoring=scoring, cv=kfold).fit(
        X_train, y_train)
print("优化模型交叉验证分数: %f" % (grid_search.best_score_))
pred_result = grid_search.best_estimator_.predict(X_test)
pred_score = accuracy_score(y_test, pred_result)
print('优化模型测试集分数： %.4f' % pred_score)
cv_results = cross_val_score(
    grid_search.best_estimator_, X_train, y_train, cv=kfold, scoring=scoring)
msg = "优化模型交叉验证分数: %f (%f)" % (cv_results.mean(), cv_results.std())
print(msg)
print('最优化模型参数: ')
pprint.pprint(grid_search.best_params_)
sub_result = grid_search.best_estimator_.predict(df_test[predictor])
submission = pd.DataFrame({'result': sub_result})
submission.to_csv('result.csv', index=False)
