'''导入库'''
import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import pprint

from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import MinMaxScaler
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

# 连续变量
continuous_var = [
    'Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
]
# 分类变量
categorical_var = [
    x for x in df_train.columns if x not in continuous_var and x != target_var
]

# 数据类型
# 数值型
numeric_var = [
    x for x in df_train.dtypes.index
    if df_train.dtypes[x] != 'object' and x != target_var
]
# 字符型
character_var = [
    x for x in df_train.dtypes.index
    if x not in numeric_var and x != target_var
]


# 将字符变量数值化
le = LabelEncoder()
for col in character_var:
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col] = le.fit_transform(df_test[col])

# pprint.pprint(df_train.head(2))


# 将连续变量标准化
scaler = MinMaxScaler()

poly = PolynomialFeatures()
poly_features = poly.fit_transform(df_train[continuous_var])

poly_featuresdf = scaler.fit_transform(poly_features)
df = pd.DataFrame(poly_features)
pprint.pprint(df)
df_train[continuous_var] = scaler.fit_transform(df_train[continuous_var])
df_train = pd.concat([df_train, df], axis=1)


poly_features = poly.fit_transform(df_test[continuous_var])
poly_features = scaler.fit_transform(poly_features)
df = pd.DataFrame(poly_features)
df_test[continuous_var] = scaler.fit_transform(df_test[continuous_var])
df_test = pd.concat([df_test, df], axis=1)
'''
predictor_var = [x for x in df_train.columns if x != target_var]

clf = LogisticRegression()
rfe = RFE(clf)
pipeline = Pipeline([('rfe', rfe), ('clf', clf)])

parameters = {
    'rfe__n_features_to_select': range(1, len(predictor_var) + 1),
    'clf__penalty': ['l1', 'l2'],
    'clf__C': [0.1, 0.001],
    'clf__tol': [1e-6, 1e-5, 1e-4]
}

grid_search = GridSearchCV(
    pipeline, parameters, verbose=3, cv=5).fit(df_train[predictor_var],
                                               df_train[target_var])
# pprint.pprint(grid_search.best_score_)
# pprint.pprint(grid_search.cv_results_)

bool_index = grid_search.best_estimator_.named_steps['rfe'].support_.tolist()
predictor = [
    predictor_var[x] for x in range(len(bool_index)) if bool_index[x] is True
]
'''
predictor = [x for x in df_train.columns if x != target_var]
validation_size = 0.23
seed = 7
scoring = 'accuracy'
X_train, X_Test, Y_train, Y_Test = train_test_split(
    df_train[predictor],
    df_train[target_var],
    test_size=validation_size,
    random_state=seed)

# result = grid_search.predict(df_test[predictor_var])
# pd.DataFrame({'result':result}).to_csv('result.csv')

models = []
models.append(('LR', LogisticRegression()))
models.append(('RandomForest', RandomForestClassifier(
    n_estimators=100, max_features=3)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('GradientBoosting', GradientBoostingClassifier(
    n_estimators=100, random_state=seed)))
models.append(('AdaBoost', AdaBoostClassifier(
    n_estimators=100, random_state=seed)))
results = []
names = []
print("Loan Approval Model performance")
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(
        model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Predict values
ensemble = VotingClassifier(models).fit(X_train, Y_train)
finalpred = ensemble.predict(X_Test)
print("Combined Model Accuracy", accuracy_score(Y_Test, finalpred))
print(confusion_matrix(Y_Test, finalpred))
print(classification_report(Y_Test, finalpred))

# Submission
ensemble = VotingClassifier(models).fit(df_train[predictor],
                                        df_train[target_var])
predictions = ensemble.predict(df_test[predictor])
submission = pd.DataFrame({'result': predictions})
submission.to_csv("submission.csv", index=False)
# 为分类变量常见虚拟变量
# for col in categorical_var:
#    dummy = pd.get_dummies(df_train[col], prefix=col)
#    # 合并虚拟变量
#    df_train = pd.concat([df_train, dummy], axis=1)
#    # 删除原变量
#    df_train.drop([col], axis=1, inplace=True)
#
#    # 测试集进行同样操作
#    dummy = pd.get_dummies(df_test[col], prefix=col)
#    df_test = pd.concat([df_test, dummy], axis=1)
#    df_test.drop([col], axis=1, inplace=True)

# 为连续变量生成多项式
# poly = PolynomialFeatures()
# poly_features = poly.fit_transform(df_train[continuous_var])
# poly_df = pd.Dataframe(poly_features, index)
