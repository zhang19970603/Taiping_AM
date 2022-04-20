import pandas as pd
from pandas.plotting import autocorrelation_plot
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import VAR
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from xgboost import XGBRegressor, plot_importance

file_path_US = 'E:\\blp\\bqnt\\extra data\\US\\'
file_path_China = 'E:\\blp\\bqnt\\extra data\\China\\'
file_path_ex = 'E:\\blp\\bqnt\\extra data\\extra data\\'
file_path_micro = 'E:\\blp\\bqnt\\extra data\\micro\\'

macroFlag = False 
techFlag = True 
sentiFlag = True
microFlag = True 

start_date = '2019-01-01'
end_date = '2020-01-01'



def load_data(file_path_US, file_path_China, file_path_ex, file_path_micro):
    df_non_agri = pd.read_excel(file_path_US + '非农就业人数(月).xls', names=['Date','ADP non-agri']) 

    df_Cor_bond = pd.read_excel(file_path_US + 'US企业债收益率(月).xls', names=['Date','US_Bond_return'])
    df_Cor_bond = df_Cor_bond.iloc[191:-1,:].reset_index().drop(["index"], axis = 1)

    df_gov_asset = pd.read_excel(file_path_US + 'US储备资产(月).xls', names=['Date','US_gov_assets'])  
    df_gov_asset = df_gov_asset.iloc[14:,:].reset_index().drop(["index"], axis = 1)

    dates = df_gov_asset.Date.tolist()
    df_us_base_rate = pd.read_excel(file_path_US + 'US基准利率.xls', names=['Date','US_base_rate'])  
    df_us_base_rate['monthly_base_rate'] = df_us_base_rate['US_base_rate'].rolling(27, min_periods = 1).mean() 
    df_us_base_rate = df_us_base_rate.iloc[::27,:]
    df_us_base_rate = df_us_base_rate.iloc[15:-3,:].reset_index().drop(["index"],axis=1)
    df_us_base_rate = df_us_base_rate.drop(["US_base_rate"],axis=1)
    df_us_base_rate["Date"] = np.array(dates)

    df_us_forei_res = pd.read_excel(file_path_US + 'US外汇储备(月).xls', names=['Date','US_foreign_asset_rev'])  
    df_us_forei_res = df_us_forei_res.iloc[14:,:].reset_index().drop(["index"], axis = 1)

    dates = df_gov_asset.Date.tolist()
    df_vix = pd.read_excel(file_path_US + 'US市场波动率指数(VIX).xls', names=['Date','VIX'])  
    df_vix = df_vix.iloc[2524:,:].reset_index().drop(["index"], axis = 1)
    df_vix['monthly_vix'] = df_vix['VIX'].rolling(21, min_periods = 1).mean() 
    df_vix = df_vix.iloc[::21,:].reset_index().drop(["index"],axis=1)
    df_vix = df_vix.iloc[:-2,:].drop(["VIX"],axis=1)
    df_vix["Date"] = np.array(dates)

    df_us_com_cre = pd.read_excel(file_path_US + 'US消费信贷(月).xls', names=['Date','US_consumer_credit'])  
    df_us_com_cre = df_us_com_cre.iloc[683:,:].reset_index().drop(["index"], axis = 1)

    df_us_cpi = pd.read_excel(file_path_US + 'US消费者物价指数(CPI)同比(月).xls', names=['Date','US_CPI'])  
    df_us_cpi = df_us_cpi.iloc[1031:-1,:].reset_index().drop(["index"], axis = 1)

    df_us_deficit = pd.read_excel(file_path_US + 'US联邦政府财政赤字(月).xls', names=['Date','US_gov_deficit'])  
    df_us_deficit = df_us_deficit.iloc[230:-1,:].reset_index().drop(["index"], axis = 1)

    df_us_money_supply = pd.read_excel(file_path_US + 'US货币供应量(月).xls', names=['Date','US_money_supply'])  
    df_us_money_supply = df_us_money_supply.iloc[491:-1,:].reset_index().drop(["index"], axis = 1)

    df_us_import_price_index = pd.read_excel(file_path_US + 'US进口价格指数(月).xls', names=['Date','import_price_index'])  
    df_us_import_price_index = df_us_import_price_index.iloc[157:-1,:].reset_index().drop(["index"], axis = 1)

    df_us_pmi = pd.read_excel(file_path_US + 'US采购经理指数(PMI)(月).xls', names=['Date','US_pmi'])  
    df_us_pmi = df_us_pmi.iloc[623:-1,:].reset_index().drop(["index"], axis = 1)

    df_US = pd.concat([df_us_money_supply,df_Cor_bond, df_gov_asset,df_us_base_rate,
                    df_us_forei_res,df_vix,df_us_com_cre,df_us_cpi,df_us_deficit,
                    df_us_pmi,df_us_import_price_index, df_non_agri], 
                    axis=1).drop(["Date"], axis =1) 
    df_US.insert(loc=0, column='Date', value=dates)


    """### **China Data Extraction**"""
    df_M0 = pd.read_excel(file_path_China + 'M0 supply.xls', names=['Date','M0'])  
    df_M0 = df_M0[65:-1].reset_index().drop(["index"], axis = 1)

    df_Chinese_CPI = pd.read_excel(file_path_China + 'CPI monthly change.xls', names=['Date','Chinese CPI'])  
    df_Chinese_CPI = df_Chinese_CPI.iloc[155:-1,:].reset_index().drop(["index"], axis = 1)

    df_oversea_inv = pd.read_excel(file_path_China + 'Oversea security investment.xls', names=['Date','CHN_oversea_sec_inv'])  
    df_oversea_inv = df_oversea_inv.iloc[252:,:].reset_index().drop(["index"], axis = 1)

    df_industry_increment = pd.read_excel(file_path_China + 'monthly above-scale industrial increment.xls', names=['Date','increment from scaled industry'])  
    df_industry_increment = df_industry_increment.iloc[119:-1,:].reset_index().drop(["index"], axis = 1)

    df_mac_eco_index = pd.read_excel(file_path_China + 'macro eco index.xls', names=['Date','macro eco performance index'])  
    df_mac_eco_index = df_mac_eco_index.iloc[106:,:].reset_index().drop(["index"], axis = 1)

    df_CNY_import_index = pd.read_excel(file_path_China + 'export price index.xls', names=['Date','CNY HS2 import index'])  
    df_CNY_import_index = df_CNY_import_index.iloc[81:,:].reset_index().drop(["index"], axis = 1)
    df_CNY_import_index["Date"] = np.array(dates)


    df_CNY_saving_base = pd.read_excel(file_path_China + 'RMB base rate.xls', names=['Date','CNY Saving Account Base Rate'])  
    df_CNY_saving_base = df_CNY_saving_base.iloc[131:,:].reset_index().drop(["index"], axis = 1)

    df_fiscal = pd.read_excel(file_path_China + 'fiscal spending.xls', names=['Date','Government Fiscal Spending'])  
    df_fiscal = df_fiscal.iloc[117:,:].reset_index().drop(["index"], axis = 1)

    df_CNY_foreign_res = pd.read_excel(file_path_China + 'monthly official foriegn asset reserve.xls', names=['Date','Official Foreign Asset Reserve'])  
    df_CNY_foreign_res = df_CNY_foreign_res.iloc[126:-1,:].reset_index().drop(["index"], axis = 1)

    df_deposit_reserve_ratio = pd.read_excel(file_path_China + 'financial institute deposite funds.xls', names=['Date','Financial Institute Required Reserve Ratio'])  
    df_deposit_reserve_ratio = df_deposit_reserve_ratio.iloc[179:-1,:].reset_index().drop(["index"], axis = 1)

    df_consumer_confi_index = pd.read_excel(file_path_China + 'consumer confidence index.xls', names=['Date','Consumer Confidence Index'])  
    df_consumer_confi_index = df_consumer_confi_index.iloc[107:,:].reset_index().drop(["index"], axis = 1)

    df_China = pd.concat([df_M0,df_Chinese_CPI,df_oversea_inv,df_industry_increment,
                        df_mac_eco_index,df_CNY_import_index,df_CNY_saving_base,
                        df_fiscal,df_deposit_reserve_ratio,df_CNY_foreign_res,
                        df_consumer_confi_index],axis=1).drop(["Date"], axis =1) 
                        
    df_China.insert(loc=0, column='Date', value=dates)
    df_China = df_China.reset_index().drop(["index"], axis = 1)
    df_all = pd.concat([df_US, df_China], axis=1).reset_index().drop(["index"], axis=1)

    ########3
    # df_fx = pd.read_excel(file_path_ex + 'monthly USDCNY exchange rate.xls', names=['Date','USD/CNY'])  
    # df_fx['Date'] = pd.to_datetime(df_fx['Date'])

    # df_fx_actual = pd.read_excel(file_path_ex + 'monthly USDCNY exchange rate.xls', names=['Date','USD/CNY'])  
    # df_fx_actual['Date'] = pd.to_datetime(df_fx_actual['Date'])

    # moving average calculation
    # df_fx['daily_dif'] = df_fx['USD/CNY'].diff(periods=1) # daily increase
    # df_fx['SMA_7_ON'] = df_fx['USD/CNY'].rolling(7, min_periods = 1).mean() # weekly simple moving average
    # df_fx['SMA_30_ON'] = df_fx['USD/CNY'].rolling(30, min_periods = 1).mean() # monthly simple moving average
    # df_fx['CMA'] = df_fx['USD/CNY'].expanding().mean() # cumulative moving average
    # df_fx['EMA_0.3'] = df_fx['USD/CNY'].ewm(alpha = 0.3, adjust = False).mean() # exponential moving average, alpha of 0.3
    # df_fx['EMA_0.1'] = df_fx['USD/CNY'].ewm(alpha = 0.1, adjust = False).mean() # exponential moving average, alpha of 0.1

    # df_tech = df_fx.reset_index().drop(["index"], axis=1)

    df_senti = pd.read_csv(file_path_micro + 'tweet_sentiment.csv')
    df_senti['date'] = pd.to_datetime(df_senti['date'])
    df_senti = df_senti.fillna(0)

    df_CNY_1_yr_bond = pd.read_excel(file_path_micro + '1yr中债国债到期收益率(中债)(日).xls', names=['Date','1_yr_AAA_CNY_govern_bond'])  
    df_CNY_1_yr_bond = df_CNY_1_yr_bond.iloc[1995:,:].reset_index().drop(["index"], axis = 1)

    df_NASDAQ = pd.read_excel(file_path_micro + 'NASDAQ(日).xls', names=['Date','NASDAQ'])  
    df_NASDAQ = df_NASDAQ.iloc[9817:,:].reset_index().drop(["index"], axis = 1)

    df_sp500 = pd.read_excel(file_path_micro + 'sp500主要指数(日).xls', names=['Date','sp500'])  
    df_sp500 = df_sp500.iloc[20598:,:].reset_index().drop(["index"], axis = 1)

    df_usd_libor = pd.read_excel(file_path_micro + 'USD伦敦同业拆借利率(LIBOR)(日).xls', names=['Date','USD_LIBOR'])  
    df_usd_libor = df_usd_libor.iloc[2214:,:].reset_index().drop(["index"], axis = 1)

    df_shibor = pd.read_excel(file_path_micro + '上海银行间同业拆放利率(SHIBOR)(日).xls', names=['Date','Shibor'])  
    df_shibor = df_shibor.iloc[14:,:].reset_index().drop(["index"], axis = 1)

    df_CDB_bond = pd.read_excel(file_path_micro + '中债国开债到期收益率(中债)(日).xls', names=['Date','1_yr_CDB'])  
    df_CDB_bond = df_CDB_bond.iloc[1991:,:].reset_index().drop(["index"], axis = 1)

    df_sp500_vix = pd.read_excel(file_path_micro + '市场波动率指数(VIX).xls', names=['Date','sp500_vix'])  
    df_sp500_vix = df_sp500_vix.iloc[5041:,:].reset_index().drop(["index"], axis = 1)

    df_fx = pd.read_excel(file_path_micro + 'USDCNY人民币汇率(日).xls', names=['Date','USD/CNY'])  
    df_fx = df_fx.iloc[3797:,:].reset_index().drop(["index"], axis = 1)
    df_fx['daily_dif'] = df_fx['USD/CNY'].diff(periods=1) # daily increase
    df_fx['SMA_7_ON'] = df_fx['USD/CNY'].rolling(7, min_periods = 1).mean() # weekly simple moving average
    df_fx['SMA_30_ON'] = df_fx['USD/CNY'].rolling(30, min_periods = 1).mean() # monthly simple moving average
    df_fx['CMA'] = df_fx['USD/CNY'].expanding().mean() # cumulative moving average
    df_fx['EMA_0.3'] = df_fx['USD/CNY'].ewm(alpha = 0.3, adjust = False).mean() # exponential moving average, alpha of 0.3
    df_fx['EMA_0.1'] = df_fx['USD/CNY'].ewm(alpha = 0.1, adjust = False).mean() # exponential moving average, alpha of 0.1
    df_fx = df_fx.reset_index().drop(["index","USD/CNY"], axis=1)

    df_fx_actual = pd.read_excel(file_path_micro + 'USDCNY人民币汇率(日).xls', names=['Date','USD/CNY'])  
    df_fx_actual = df_fx_actual.iloc[3797:,:].reset_index().drop(["index"], axis = 1)

    df_micro = df_CNY_1_yr_bond.set_index('Date').join(
        df_NASDAQ.set_index('Date')).join(df_sp500.set_index('Date')).join(
            df_usd_libor.set_index('Date')).join(df_shibor.set_index('Date')).join(
                df_CDB_bond.set_index('Date')).join(df_sp500_vix.set_index('Date'))   #join(df_USDCNY_spot.set_index('Date'))

    df_micro=  df_micro.fillna(method='ffill')
    df_micro = df_micro.reset_index()

    return df_all, df_fx, df_senti, df_micro, df_fx_actual


def get_input_df(macroFlag, techFlag, sentiFlag, microFlag, df_fx_actual):
    #start_date =  datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
    if macroFlag:
        df_all ## 填补30日空缺
        df_fx_actual = df_fx_actual.set_index('Date').join(df_all.set_index('Date')).reset_index()

    if techFlag:
        df_fx_actual = df_fx_actual.set_index('Date').join(df_fx.set_index('Date')).reset_index()
    
    if sentiFlag:
        df_fx_actual = df_fx_actual.set_index('Date').join(df_senti.set_index('date')).reset_index()
    
    if microFlag:
        df_fx_actual = df_fx_actual.set_index('Date').join(df_micro.set_index('Date')).reset_index()

    df_fx_actual = df_fx_actual[(df_fx_actual['Date'] >= datetime.datetime.strptime('2015-01-01', '%Y-%m-%d').date())]

    return df_fx_actual


def get_pred_result(model, start_date, end_date, df_input):
    start_date =  datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date =  datetime.datetime.strptime(end_date, '%Y-%m-%d').date()

    df_train = df_input[(df_input['Date'] <= start_date)].drop(['Date'],axis = 1)
    df_test = df_input[(df_input['Date'] >= start_date) & (df_input['Date'] <= end_date)].drop(['Date'],axis = 1)

    x_train, y_train = np.array(df_train.iloc[:,1:]), np.array(df_train.iloc[:,0])
    x_test, y_test = np.array(df_test.iloc[:,1:]), np.array(df_test.iloc[:,0])
    
    if model == "Random Forest" :
        y_pred, SMAPE, feature_importance = get_preds_RF(x_train, y_train, x_test, y_test)
    elif model == "Decision Tree" :
        y_pred, SMAPE, feature_importance = get_preds_DT(x_train, y_train, x_test, y_test)
    elif model == "XGBoost" :
        y_pred, SMAPE, feature_importance = get_preds_XGB(x_train, y_train, x_test, y_test)
    else :
        y_pred, SMAPE, feature_importance = get_preds_LR(x_train, y_train, x_test, y_test)
    
    df_final = df_input[(df_input['Date'] >= start_date) & (df_input['Date'] <= end_date)][['Date', 'USD/CNY']]
    df_final['y_pred'] = y_pred
    df_final.rename(columns={'USD/CNY':'y'}, inplace = True)

    return df_final, SMAPE, feature_importance

def get_preds_DT(x_train, y_train, x_test, y_test):
    final_DS = DecisionTreeRegressor(criterion = "mae", max_depth = 25, splitter='best').fit(x_train,y_train)
    y_valid = y_test
    preds = final_DS.predict(x_test)
    SMAPE = (100/len(y_valid)) * np.sum(2 * np.abs(preds - y_valid) / (np.abs(y_valid) + np.abs(preds)))

    X = np.vstack((x_train, x_test))
    y = np.hstack((y_train, y_test))
    X_std = MinMaxScaler().fit_transform(X) 
    # clf = DecisionTreeRegressor(criterion = "mae", max_depth = 25, splitter='best')
    clf = final_DS
    predictor = clf.fit(X_std, y)
    feature_importance = predictor.feature_importances_

    return preds, SMAPE, feature_importance
    # Feature_name = X.columns
    # indices = feature_importance.argsort()[::-1][0:30]
    # feature_importance [[indices]]
    # Feature_name[[indices]]

def get_preds_RF(x_train, y_train, x_test, y_test):
    final_RF = RandomForestRegressor(criterion = "mae", max_depth = 200, n_estimators = 10).fit(x_train,y_train)
    preds = final_RF.predict(x_test)

    y_valid = y_test
    SMAPE = (100/len(y_valid)) * np.sum(2 * np.abs(preds - y_valid) / (np.abs(y_valid) + np.abs(preds)))

    X = pd.concat(x_train, x_test)
    y = pd.concat(y_train, y_test)
    X_std = MinMaxScaler().fit_transform(X) 
    clf = final_RF
    predictor = clf.fit(X_std, y)
    feature_importance = predictor.feature_importances_

    return preds, SMAPE, feature_importance

def get_preds_XGB(x_train, y_train, x_test, y_test):
    final_xgb = XGBRegressor(n_estimators=300, learning_rate=0.2, max_depth=100, 
                            min_child_weight=10,gamma=0.2,reg_alpha=0.1, reg_lambda=1,
                            colsample_bytree= 0.2, subsample=0.3).fit(x_train,y_train) 

    preds = final_xgb.predict(x_test)
    y_valid = y_test
    SMAPE = (100/len(y_valid)) * np.sum(2 * np.abs(preds - y_valid) / (np.abs(y_valid) + np.abs(preds)))

    #scaling based on range for different features 
    X = pd.concat(x_train, x_test)
    y = pd.concat(y_train, y_test)
    X_std = MinMaxScaler().fit_transform(X) 
    clf = final_xgb
    predictor = clf.fit(X_std, y)
    feature_importance = predictor.feature_importances_

    return preds, SMAPE, feature_importance

def get_preds_LR(x_train, y_train, x_test, y_test):
    final_LR = LinearRegression(normalize = True).fit(x_train,y_train) 

    preds = final_LR.predict(x_test)
    y_valid = y_test
    SMAPE = (100/len(y_valid)) * np.sum(2 * np.abs(preds - y_valid) / (np.abs(y_valid) + np.abs(preds)))

    #scaling based on range for different features 
    X = pd.concat(x_train, x_test)
    y = pd.concat(y_train, y_test)
    predictor = final_LR.fit(X_std, y)
    feature_importance = [abs(i) for i in predictor.coef_]

    return preds, SMAPE, feature_importance

# def generate_comparison(df_output, model):
#     df = df_output.set_index(['Date'], inplace = true)
#     comparisonViz = bqviz.PlotComparison(df = df, title = "{} forecast results VS actual FX")
#     return comparisonViz

# # Feature_name = df_input.columns
# def generate_bar(Feature_name, importance):
#     indices = importance.argsort()[::-1][0:30]
#     scores = importance[[indices]]
#     names = Feature_name[[indices]]
#     return bqviz.barPlot(scores, names)

   

if __name__ == '__main__':
    # df_all, df_fx, df_senti, df_micro, df_fx_actual = load_data(file_path_US, file_path_China, file_path_ex, file_path_micro)
    
    # get_input_df(macroFlag, techFlag, sentiFlag, microFlag, df_fx_actual)











