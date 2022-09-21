from sklearn import model_selection
from collections import Counter
from sklearn import metrics
import time
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
from sklearn import svm

A = np.load('../../data/Hoorn_array.npy')  # 读入数据
# 划分数据（train:test=3:1),顺序划分
X = A[:, 0:-1]
Y = A[:, -1]
X_train = X[0:171]
X_test = X[171:]
Y_train = Y[0:171]
Y_test = Y[171:]
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# 准备空表
station = 'Hoorn'
with open('{}.csv'.format(station), 'w+') as out_file:
    out_file.write('methods,'
                   'runtime,'
                   'MRE,'
                   'MAE,'
                   'MSE,'
                   'MAPE,'
                   'SMAPE,'
                   'RMSE,'
                   'R2,'
                   '\n')


    def mre(y_true, y_pred):
        res_mre = np.average(np.abs((y_pred - y_true) / y_true))
        return res_mre


    def smape(y_true, y_pred):
        res_smape = 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

        return res_smape


    # SVM
    starttime_SVM = time.time()  # 开始计时
    clf = svm.SVR()
    clf.fit(X_train, Y_train)
    SVM_pred = clf.predict(X_test)
    endtime_SVM = time.time()  # 结束计时
    runtime_SVM = endtime_SVM - starttime_SVM
    print(runtime_SVM)
    out_file.write('SVM,')
    list_SVM = [runtime_SVM,
                mre(Y_test, SVM_pred),
                metrics.mean_absolute_error(Y_test, SVM_pred),
                metrics.mean_squared_error(Y_test, SVM_pred),
                metrics.mean_absolute_percentage_error(Y_test, SVM_pred),
                smape(Y_test, SVM_pred),
                np.sqrt(metrics.mean_squared_error(Y_test, SVM_pred)),
                metrics.r2_score(Y_test, SVM_pred)
                ]
    list_SVM = [str(i) for i in list_SVM]
    out_file.write(','.join(list_SVM))
    out_file.write('\n')
    # RF
    from sklearn.ensemble import RandomForestRegressor

    starttime_RF = time.time()  # 开始计时
    regressor = RandomForestRegressor(n_estimators=200, random_state=0)
    regressor.fit(X_train, Y_train)
    RF_Y_pred = regressor.predict(X_test)
    endtime_RF = time.time()  # 结束计时
    runtime_RF = endtime_RF - starttime_RF
    print(runtime_RF)
    out_file.write('RF,')
    list_RF = [runtime_RF,
               mre(Y_test, RF_Y_pred),
               metrics.mean_absolute_error(Y_test, RF_Y_pred),
               metrics.mean_squared_error(Y_test, RF_Y_pred),
               metrics.mean_absolute_percentage_error(Y_test, RF_Y_pred),
               smape(Y_test, RF_Y_pred),
               np.sqrt(metrics.mean_squared_error(Y_test, RF_Y_pred)),
               metrics.r2_score(Y_test, RF_Y_pred)
               ]
    list_RF = [str(i) for i in list_RF]
    out_file.write(','.join(list_RF))
    out_file.write('\n')

    # DT
    from sklearn.tree import DecisionTreeClassifier

    # 构建分类决策树模型
    starttime_DT = time.time()  # 开始计时
    tree = DecisionTreeClassifier(random_state=0)  # 默认将树完全展开，random_state=0，用于解决内部平局问题
    # 模型训练
    tree.fit(X_train, Y_train)
    DT_Y_pred = tree.predict(X_test)
    endtime_DT = time.time()  # 结束计时
    runtime_DT = endtime_DT - starttime_DT
    print(runtime_DT)
    out_file.write('DT,')
    list_DT = [runtime_DT,
               mre(Y_test, DT_Y_pred),
               metrics.mean_absolute_error(Y_test, DT_Y_pred),
               metrics.mean_squared_error(Y_test, DT_Y_pred),
               metrics.mean_absolute_percentage_error(Y_test, DT_Y_pred),
               smape(Y_test, DT_Y_pred),
               np.sqrt(metrics.mean_squared_error(Y_test, DT_Y_pred)),
               metrics.r2_score(Y_test, DT_Y_pred)
               ]
    list_DT = [str(i) for i in list_DT]
    out_file.write(','.join(list_DT))
    out_file.write('\n')

    # Bayes
    from sklearn import linear_model

    starttime_Bayes = time.time()  # 开始计时
    Bayes = linear_model.BayesianRidge()  # 贝叶斯岭回归
    Bayes.fit(X_train, Y_train)
    Bayes_Y_pred = Bayes.predict(X_test)
    endtime_Bayes = time.time()  # 结束计时
    runtime_Bayes = endtime_Bayes - starttime_Bayes
    print(runtime_Bayes)
    out_file.write('Bayes,')
    list_Bayes = [runtime_Bayes,
                  mre(Y_test, Bayes_Y_pred),
                  metrics.mean_absolute_error(Y_test, Bayes_Y_pred),
                  metrics.mean_squared_error(Y_test, Bayes_Y_pred),
                  metrics.mean_absolute_percentage_error(Y_test, Bayes_Y_pred),
                  smape(Y_test, Bayes_Y_pred),
                  np.sqrt(metrics.mean_squared_error(Y_test, Bayes_Y_pred)),
                  metrics.r2_score(Y_test, Bayes_Y_pred)
                  ]
    list_Bayes = [str(i) for i in list_Bayes]
    out_file.write(','.join(list_Bayes))
    out_file.write('\n')

    # 第二种贝叶斯回归器
    starttime_Bayes2 = time.time()  # 开始计时
    Bayes2 = linear_model.ARDRegression()  # 主动相关决策理论回归
    Bayes2.fit(X_train, Y_train)
    Bayes2_Y_pred = Bayes2.predict(X_test)
    endtime_Bayes2 = time.time()  # 结束计时
    runtime_Bayes2 = endtime_Bayes2 - starttime_Bayes2
    print(runtime_Bayes2)
    out_file.write('Bayes,')
    list_Bayes2 = [runtime_Bayes2,
                   mre(Y_test, Bayes2_Y_pred),
                   metrics.mean_absolute_error(Y_test, Bayes2_Y_pred),
                   metrics.mean_squared_error(Y_test, Bayes2_Y_pred),
                   metrics.mean_absolute_percentage_error(Y_test, Bayes2_Y_pred),
                   smape(Y_test, Bayes2_Y_pred),
                   np.sqrt(metrics.mean_squared_error(Y_test, Bayes2_Y_pred)),
                   metrics.r2_score(Y_test, Bayes2_Y_pred)
                   ]
    list_Bayes2 = [str(i) for i in list_Bayes2]
    out_file.write(','.join(list_Bayes2))
    out_file.write('\n')

    # Liner

    from sklearn.linear_model import LinearRegression

    starttime_liner = time.time()  # 开始计时
    liner = LinearRegression().fit(X_train, Y_train)
    liner_Y_pred = liner.predict(X_test)
    endtime_liner = time.time()  # 结束计时
    runtime_liner = endtime_liner - starttime_liner
    print(runtime_liner)
    out_file.write('Liner,')
    list_liner = [runtime_liner,
                  mre(Y_test, liner_Y_pred),
                  metrics.mean_absolute_error(Y_test, liner_Y_pred),
                  metrics.mean_squared_error(Y_test, liner_Y_pred),
                  metrics.mean_absolute_percentage_error(Y_test, liner_Y_pred),
                  smape(Y_test, liner_Y_pred),
                  np.sqrt(metrics.mean_squared_error(Y_test, liner_Y_pred)),
                  metrics.r2_score(Y_test, liner_Y_pred)
                  ]
    list_liner = [str(i) for i in list_liner]
    out_file.write(','.join(list_liner))
    out_file.write('\n')

    # Ridge
    from sklearn.linear_model import Ridge

    starttime_ridge = time.time()  # 开始计时
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, Y_train)
    ridge_Y_pred = ridge.predict(X_test)
    endtime_ridge = time.time()  # 结束计时
    runtime_ridge = endtime_ridge - starttime_ridge
    print(runtime_ridge)
    out_file.write('Ridge,')
    list_ridge = [runtime_ridge,
                  mre(Y_test, ridge_Y_pred),
                  metrics.mean_absolute_error(Y_test, ridge_Y_pred),
                  metrics.mean_squared_error(Y_test, ridge_Y_pred),
                  metrics.mean_absolute_percentage_error(Y_test, ridge_Y_pred),
                  smape(Y_test, ridge_Y_pred),
                  np.sqrt(metrics.mean_squared_error(Y_test, ridge_Y_pred)),
                  metrics.r2_score(Y_test, ridge_Y_pred)
                  ]
    list_ridge = [str(i) for i in list_ridge]
    out_file.write(','.join(list_ridge))
    out_file.write('\n')

    # Lasso
    # 基本Lasso
    starttime_Lasso = time.time()  # 开始计时
    Lasso = linear_model.Lasso(alpha=0.1)
    Lasso.fit(X_train, Y_train)
    Lasso_Y_pred = Lasso.predict(X_test)
    endtime_Lasso = time.time()  # 结束计时
    runtime_Lasso = endtime_Lasso - starttime_Lasso
    print(runtime_Lasso)
    out_file.write('Lasso,')
    list_Lasso = [runtime_Lasso,
                  mre(Y_test, Lasso_Y_pred),
                  metrics.mean_absolute_error(Y_test, Lasso_Y_pred),
                  metrics.mean_squared_error(Y_test, Lasso_Y_pred),
                  metrics.mean_absolute_percentage_error(Y_test, Lasso_Y_pred),
                  smape(Y_test, Lasso_Y_pred),
                  np.sqrt(metrics.mean_squared_error(Y_test, Lasso_Y_pred)),
                  metrics.r2_score(Y_test, Lasso_Y_pred)
                  ]
    list_Lasso = [str(i) for i in list_Lasso]
    out_file.write(','.join(list_Lasso))
    out_file.write('\n')

    # LassoLars
    starttime_LassoLars = time.time()  # 开始计时
    LassoLars = linear_model.LassoLars(alpha=0.01, normalize=False)
    LassoLars.fit(X_train, Y_train)
    LassoLars_Y_pred = LassoLars.predict(X_test)
    endtime_LassoLars = time.time()  # 结束计时
    runtime_LassoLars = endtime_LassoLars - starttime_LassoLars
    print(runtime_LassoLars)
    out_file.write('LassoLars,')
    list_LassoLars = [runtime_LassoLars,
                      mre(Y_test, LassoLars_Y_pred),
                      metrics.mean_absolute_error(Y_test, LassoLars_Y_pred),
                      metrics.mean_squared_error(Y_test, LassoLars_Y_pred),
                      metrics.mean_absolute_percentage_error(Y_test, LassoLars_Y_pred),
                      smape(Y_test, LassoLars_Y_pred),
                      np.sqrt(metrics.mean_squared_error(Y_test, LassoLars_Y_pred)),
                      metrics.r2_score(Y_test, LassoLars_Y_pred)
                      ]
    list_LassoLars = [str(i) for i in list_LassoLars]
    out_file.write(','.join(list_LassoLars))
    out_file.write('\n')

    # KNN
    from sklearn import neighbors

    starttime_knn = time.time()  # 开始计时
    knn = neighbors.KNeighborsRegressor(7, weights="uniform")  # 第一个参数neighbors可能需要调整
    knn.fit(X_train, Y_train)
    knn_Y_pred = knn.predict(X_test)
    endtime_knn = time.time()  # 结束计时
    runtime_knn = endtime_knn - starttime_knn
    print(runtime_knn)
    out_file.write('KNN,')
    list_knn = [runtime_knn,
                mre(Y_test, knn_Y_pred),
                metrics.mean_absolute_error(Y_test, knn_Y_pred),
                metrics.mean_squared_error(Y_test, knn_Y_pred),
                metrics.mean_absolute_percentage_error(Y_test, knn_Y_pred),
                smape(Y_test, knn_Y_pred),
                np.sqrt(metrics.mean_squared_error(Y_test, knn_Y_pred)),
                metrics.r2_score(Y_test, knn_Y_pred)
                ]
    list_knn = [str(i) for i in list_knn]
    out_file.write(','.join(list_knn))
    out_file.write('\n')

    # MLP
    from sklearn.neural_network import MLPRegressor

    starttime_MLP = time.time()  # 开始计时
    MLP = MLPRegressor(random_state=1, max_iter=10000).fit(X_train, Y_train)
    MLP_Y_pred = MLP.predict(X_test)
    endtime_MLP = time.time()  # 结束计时
    runtime_MLP = endtime_MLP - starttime_MLP
    print(runtime_MLP)
    out_file.write('MLP,')
    list_MLP = [runtime_MLP,
                mre(Y_test, MLP_Y_pred),
                metrics.mean_absolute_error(Y_test, MLP_Y_pred),
                metrics.mean_squared_error(Y_test, MLP_Y_pred),
                metrics.mean_absolute_percentage_error(Y_test, MLP_Y_pred),
                smape(Y_test, MLP_Y_pred),
                np.sqrt(metrics.mean_squared_error(Y_test, MLP_Y_pred)),
                metrics.r2_score(Y_test, MLP_Y_pred)
                ]
    list_MLP = [str(i) for i in list_MLP]
    out_file.write(','.join(list_MLP))
    out_file.write('\n')

    # Arima
    from statsmodels.tsa.arima.model import ARIMA

    starttime_ARIMA = time.time()  # 开始计时
    arima = ARIMA(Y_train, order=(1, 0, 0)).fit()  # order参数可能需要再调整
    arima_pred = arima.forecast(57)  # 测试集数量
    endtime_ARIMA = time.time()  # 结束计时
    runtime_ARIMA = endtime_ARIMA - starttime_ARIMA
    print(runtime_ARIMA)
    out_file.write('Arima,')
    list_ARIMA = [runtime_ARIMA,
                  mre(Y_test, arima_pred),
                  metrics.mean_absolute_error(Y_test, arima_pred),
                  metrics.mean_squared_error(Y_test, arima_pred),
                  metrics.mean_absolute_percentage_error(Y_test, arima_pred),
                  smape(Y_test, arima_pred),
                  np.sqrt(metrics.mean_squared_error(Y_test, arima_pred)),
                  metrics.r2_score(Y_test, arima_pred)
                  ]
    list_ARIMA = [str(i) for i in list_ARIMA]
    out_file.write(','.join(list_ARIMA))
    out_file.write('\n')
