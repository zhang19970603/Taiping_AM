from datetime import datetime
#from pandas_datareader import data
import pandas as pd
import numpy as np
from numpy import log, sqrt
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

import itertools as it

file_path_ex = 'E:\\blp\\bqnt\\extra data\\extra data\\'

def concat_df(fd_final):
    USDHKD = pd.read_csv(file_path_ex + '美元对港元汇率(日).csv')
    USDHKD['Date'] = pd.to_datetime(USDHKD['Date'])
    df1 =  df_final.set_index('Date').join(USDHKD.set_index('Date')).reset_index()
    df1 = df1.fillna(method = 'ffill')
    df1 = df1.assign(x_DETREND =  detrendPrice(df1.x).values)
    df1 = df1.assign(y_DETREND =  detrendPrice(df1.y).values)
    df1 = df1.assign(TIME = pd.Series(np.arange(df1.shape[0])).values) 

    return df1

def detrendPrice(series):
    # fit linear model
    length = len(series)
    x = np.arange(length)
    y = np.array(series.values)
    x_const = sm.add_constant(x) #need to add intercept constant
    model = sm.OLS(y,x_const)
    result = model.fit()
    #intercept = result.params[0]
    #beta = result.params[1]
    #print(result.summary())
    df = pd.DataFrame(result.params*x_const)
    y_hat = df[0] + df[1]
    #the residuals are the detrended prices
    resid = y-y_hat
    #add minimum necessary to residuals to avoid negative detrended parices
    resid = resid + abs(resid.min() + 1/10*resid.min())
    return resid 


def backtest_benchmark(df1, start_date, end_date):
    entryZscore = 1
    exitZscore = -0
    window =  7
    regression = 1
    residuals_model = 0

    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
    df1 = df1[(df1['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    window_hr_reg = 58 #smallest window for regression when using y_hat
    
    a = np.array([np.nan] * len(df1))
    b = [np.nan] * len(df1)  # If betas required.
    y_ = df1["y"].values
    x_ = df1[['x']].assign(constant=0).values #if constant=0, intercept is forced to zero; if constant=1 intercept is as usual
    for n in range(window_hr_reg, len(df1)):
        y = y_[(n - window_hr_reg):n]
        X = x_[(n - window_hr_reg):n]
        # betas = Inverse(X'.X).X'.y
        betas = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
        y_hat = betas.dot(x_[n, :])
        a[n] = y_hat
        b[n] = betas.tolist()  # If betas required. b[n][0] is the slope, b[n][1] is the intercept
    
    if residuals_model:
        myList = []
        for e in range(len(b)):
            if e < window_hr_reg:
                 myList.append(0)
            else:
                myList.append(b[e][0])
        df1["rolling_hedge_ratio"] = myList
    else:
        df1 = df1.assign(rolling_hedge_ratio = pd.Series(np.ones(df1.shape[0])).values)

    #repeat for detrended prices

    a = np.array([np.nan] * len(df1))
    b = [np.nan] * len(df1)  # If betas required.
    y_ = df1["y_DETREND"].values
    x_ = df1[['x_DETREND']].assign(constant=0).values #if constant=0, intercept is forced to zero; if constant=1 intercept is as usual
    for n in range(window_hr_reg, len(df1)):
        y = y_[(n - window_hr_reg):n]
        X = x_[(n - window_hr_reg):n]
        # betas = Inverse(X'.X).X'.y
        betas = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
        y_hat = betas.dot(x_[n, :])
        a[n] = y_hat
        b[n] = betas.tolist()  # If betas required. b[n][0] is the slope, b[n][1] is the intercept

    if residuals_model:
        myList = []
        for e in range(len(b)):
            if e < window_hr_reg:
                myList.append(0)
            else:
                myList.append(b[e][0])
        df1["rolling_hedge_ratio_DETREND"] = myList
    else:
        df1 = df1.assign(rolling_hedge_ratio_DETREND = pd.Series(np.ones(df1.shape[0])).values)
    
    #calculate the spread
    if residuals_model == 1:
        df1['spread'] = df1.y - df1.rolling_hedge_ratio*df1.x
    else:
        df1['spread'] = log(df1.y) - log(df1.x)
    
    #rolling regression instead of moving average
    a = np.array([np.nan] * len(df1))
    b = [np.nan] * len(df1)  # If betas required.
    y_ = df1['spread'].values
    x_ = df1[['TIME']].assign(constant=0).values #if constant=0, intercept is forced to zero; if constant=1 intercept is as usual
    for n in range(window, len(df1)):
        y = y_[(n - window):n]
        X = x_[(n - window):n]
        # betas = Inverse(X'.X).X'.y
        betas = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
        y_hat = betas.dot(x_[n, :])
        a[n] = y_hat
        b[n] = betas.tolist()  # If betas required. b[n][0] is the slope, b[n][1] is the intercept
    
    df1 = df1.assign(y_hat = pd.Series(a).values)
        
    if regression == 1:
        mean = df1['y_hat']
    else:
        mean = df1['spread'].rolling(window=window).mean()
    
    #calculate the zScore indicator
    df1 = df1.assign(meanSpread = pd.Series(a).values)
    stdSpread = df1.spread.rolling(window=window).std()
    df1['zScore'] = (df1.spread-mean)/stdSpread
    
    TotaAnnReturn, TotaAnnReturn_trading, CAGRdbl, CAGRdbl_trading, sharpe = get_return_rates(df1, entryZscore, exitZscore)
    
    return df1, TotaAnnReturn*100, CAGRdbl*100, round(sharpe,2)

def backtest_prediction(entryZ, exitZ, window, regression, resiMode, df1, start_date, end_date):
    entryZscore = entryZ #1
    exitZscore = exitZ #-0
    window =  window #7
    regression = regression #1
    residuals_model = resiMode #0

    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
    df1 = df1[(df1['Date'] >= start_date) & (df['Date'] <= end_date)]

    #find the hedge ratio and the spread
    #regress the y variable against the x variable
    #the slope of the rolling linear univariate regression=the rolling hedge ratio
    
    window_hr_reg = 58 #smallest window for regression when using y_hat
    
    a = np.array([np.nan] * len(df1))
    b = [np.nan] * len(df1)  # If betas required.
    y_ = df1["y"].values
    x_ = df1[['x']].assign(constant=0).values #if constant=0, intercept is forced to zero; if constant=1 intercept is as usual
    for n in range(window_hr_reg, len(df1)):
        y = y_[(n - window_hr_reg):n]
        X = x_[(n - window_hr_reg):n]
        # betas = Inverse(X'.X).X'.y
        betas = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
        y_hat = betas.dot(x_[n, :])
        a[n] = y_hat
        b[n] = betas.tolist()  # If betas required. b[n][0] is the slope, b[n][1] is the intercept
    
    if residuals_model:
        myList = []
        for e in range(len(b)):
            if e < window_hr_reg:
                 myList.append(0)
            else:
                myList.append(b[e][0])
        df1["rolling_hedge_ratio"] = myList
    else:
        df1 = df1.assign(rolling_hedge_ratio = pd.Series(np.ones(df1.shape[0])).values)
    
    #calculate the spread 
    if residuals_model == 1:
        df1['spread'] = df1.y - df1.rolling_hedge_ratio*df1.x
        df1['spread_pred'] = df1.y_pred - df1.rolling_hedge_ratio*df1.x
    else:
        df1['spread'] = log(df1.y) - log(df1.x)
        df1['spread_pred'] = log(df1.y_pred) - log(df1.x)
    
    #rolling regression instead of moving average
    a = np.array([np.nan] * len(df1))
    b = [np.nan] * len(df1)  # If betas required.
    y_ = df1['spread'].values
    x_ = df1[['TIME']].assign(constant=0).values #if constant=0, intercept is forced to zero; if constant=1 intercept is as usual
    for n in range(window, len(df1)):
        y = y_[(n - window):n]
        X = x_[(n - window):n]
        # betas = Inverse(X'.X).X'.y
        betas = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
        y_hat = betas.dot(x_[n, :])
        a[n] = y_hat
        b[n] = betas.tolist()  # If betas required. b[n][0] is the slope, b[n][1] is the intercept
    
    df1 = df1.assign(y_hat = pd.Series(a).values)
        
    if regression == 1:
        mean = df1['y_hat']
    else:
        mean = df1['spread'].rolling(window=window).mean()
    
    #calculate the zScore indicator
    df1 = df1.assign(meanSpread = pd.Series(a).values)
    stdSpread = df1.spread.rolling(window=window).std()
    df1['zScore'] = (df1.spread_pred-mean)/stdSpread
    
    TotaAnnReturn, TotaAnnReturn_trading, CAGRdbl, CAGRdbl_trading, sharpe = get_return_rates(df1, entryZscore, exitZscore)
    
    return df1, TotaAnnReturn*100, CAGRdbl*100, round(sharpe,2)


def backtest_prediction_NoLoop(entryZ, exitZ, window, regression, resiMode, df1, start_date, end_date):
    entryZscore = entryZ #1
    exitZscore = exitZ #-0
    window =  window #7
    regression = regression #1
    residuals_model = resiMode #0

    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
    df1 = df1[(df1['Date'] >= start_date) & (df['Date'] <= end_date)]

    #find the hedge ratio and the spread
    #regress the y variable against the x variable
    #the slope of the rolling linear univariate regression=the rolling hedge ratio
    
    window_hr_reg = 58 #smallest window for regression when using y_hat
    
    a = np.array([np.nan] * len(df1))
    b = [np.nan] * len(df1)  # If betas required.
    y_ = df1["y"].values
    x_ = df1[['x']].assign(constant=0).values #if constant=0, intercept is forced to zero; if constant=1 intercept is as usual
    for n in range(window_hr_reg, len(df1)):
        y = y_[(n - window_hr_reg):n]
        X = x_[(n - window_hr_reg):n]
        # betas = Inverse(X'.X).X'.y
        betas = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
        y_hat = betas.dot(x_[n, :])
        a[n] = y_hat
        b[n] = betas.tolist()  # If betas required. b[n][0] is the slope, b[n][1] is the intercept
    
    if residuals_model:
        myList = []
        for e in range(len(b)):
            if e < window_hr_reg:
                 myList.append(0)
            else:
                myList.append(b[e][0])
        df1["rolling_hedge_ratio"] = myList
    else:
        df1 = df1.assign(rolling_hedge_ratio = pd.Series(np.ones(df1.shape[0])).values)

        #repeat for detrended prices

    a = np.array([np.nan] * len(df1))
    b = [np.nan] * len(df1)  # If betas required.
    y_ = df1["y_DETREND"].values
    x_ = df1[['x_DETREND']].assign(constant=0).values #if constant=0, intercept is forced to zero; if constant=1 intercept is as usual
    for n in range(window_hr_reg, len(df1)):
        y = y_[(n - window_hr_reg):n]
        X = x_[(n - window_hr_reg):n]
        # betas = Inverse(X'.X).X'.y
        betas = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
        y_hat = betas.dot(x_[n, :])
        a[n] = y_hat
        b[n] = betas.tolist()  # If betas required. b[n][0] is the slope, b[n][1] is the intercept

    if residuals_model:
        myList = []
        for e in range(len(b)):
            if e < window_hr_reg:
                myList.append(0)
            else:
                myList.append(b[e][0])
        df1["rolling_hedge_ratio_DETREND"] = myList
    else:
        df1 = df1.assign(rolling_hedge_ratio_DETREND = pd.Series(np.ones(df1.shape[0])).values)
    
    #calculate the spread 
    if residuals_model == 1:
        df1['spread'] = df1.y - df1.rolling_hedge_ratio*df1.x
        df1['spread_pred'] = df1.y_pred - df1.rolling_hedge_ratio*df1.x
    else:
        df1['spread'] = log(df1.y) - log(df1.x)
        df1['spread_pred'] = log(df1.y_pred) - log(df1.x)
    
    #rolling regression instead of moving average
    a = np.array([np.nan] * len(df1))
    b = [np.nan] * len(df1)  # If betas required.
    y_ = df1['spread'].values
    x_ = df1[['TIME']].assign(constant=0).values #if constant=0, intercept is forced to zero; if constant=1 intercept is as usual
    for n in range(window, len(df1)):
        y = y_[(n - window):n]
        X = x_[(n - window):n]
        # betas = Inverse(X'.X).X'.y
        betas = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
        y_hat = betas.dot(x_[n, :])
        a[n] = y_hat
        b[n] = betas.tolist()  # If betas required. b[n][0] is the slope, b[n][1] is the intercept
    
    df1 = df1.assign(y_hat = pd.Series(a).values)
        
    if regression == 1:
        mean = df1['y_hat']
    else:
        mean = df1['spread'].rolling(window=window).mean()
    
    #calculate the zScore indicator
    df1 = df1.assign(meanSpread = pd.Series(a).values)
    stdSpread = df1.spread.rolling(window=window).std()
    df1['zScore'] = (df1.spread_pred-mean)/stdSpread
    
    TotaAnnReturn, TotaAnnReturn_trading, CAGRdbl, CAGRdbl_trading, sharpe = get_return_rates(df1, entryZscore, exitZscore)
    
    return df1, TotaAnnReturn*100, CAGRdbl*100, round(sharpe,2)


def get_return_rates(df1, entryZscore, exitZscore):
    df1['long entry'] = ((df1.zScore < - entryZscore))
    df1['long exit'] = ((df1.zScore > - exitZscore)) 
    df1['num units long'] = np.nan 
    df1.loc[df1['long entry'],'num units long'] = 1 
    df1.loc[df1['long exit'],'num units long'] = 0 
    df1.iat[0,df1.columns.get_loc("num units long")]= 0
    
    df1['num units long'] = df1['num units long'].fillna(method='pad') 
    
    #set up num units short 
    df1['short entry'] = ((df1.zScore >  entryZscore))
    df1['short exit'] = ((df1.zScore < exitZscore))
    df1['num units short'] = np.nan
    df1.loc[df1['short entry'],'num units short'] = -1 
    df1.loc[df1['short exit'],'num units short'] = 0
    df1.iat[0,df1.columns.get_loc("num units short")]= 0
    df1['num units short'] = df1['num units short'].fillna(method='pad')
###############################################################################################################################
    df1['numUnits'] = df1['num units long'] + df1['num units short']
    
    #positions_ = dollar capital allocation in each ETF
    df1["positions_x"] =-1*df1["rolling_hedge_ratio"]*df1["x"]*df1["numUnits"]
    df1["positions_y"] =df1["y"]*df1["numUnits"]
    df1["price_change_x"] = df1["x"] - df1["x"].shift(1)
    df1["price_change_y"] = df1["y"] - df1["y"].shift(1)
    df1["pnl_x"] = df1["price_change_x"]*df1["positions_x"].shift(1)/df1["x"].shift(1)
    df1["pnl_y"] = df1["price_change_y"]*df1["positions_y"].shift(1)/df1["y"].shift(1)
    df1["pnl"] = df1["pnl_x"] + df1["pnl_y"] 
    df1["portfolio_cost"] = np.abs(df1["positions_x"])+np.abs(df1["positions_y"])
    df1["port_rets"]= df1["pnl"]/df1["portfolio_cost"].shift(1)
    df1["port_rets"].fillna(0, inplace=True)

    df1 = df1.assign(I =np.cumprod(1+df1['port_rets'])) #this is good for pct return or log return
    df1.iat[0,df1.columns.get_loc('I')]= 1
    
    start_val = 1
    end_val = df1['I'].iat[-1]
    
    start_date = df1.iloc[0].Date
    end_date = df1.iloc[-1].Date
    days = (end_date - start_date).days
    
    TotaAnnReturn = (end_val-start_val)/start_val/(days/360)
    TotaAnnReturn_trading = (end_val-start_val)/start_val/(days/252)
    
    CAGRdbl_trading = round(((float(end_val) / float(start_val)) ** (1/(days/252.0))).real - 1,4) #when raised to an exponent I am getting a complex number, I need only the real part
    CAGRdbl = round(((float(end_val) / float(start_val)) ** (1/(days/360))).real - 1,4) #when raised to an exponent I am getting a complex number, I need only the real part
    
    try:
        sharpe =  (df1['port_rets'].mean()/ (df1['port_rets'].std()) * np.sqrt(252))
    except ZeroDivisionError:
        sharpe = 0.0

    return TotaAnnReturn, TotaAnnReturn_trading, CAGRdbl, CAGRdbl_trading, sharpe

    def get_return_rates_detrended(df1, entryZscore, exitZscore):
        df1['long entry'] = ((df1.zScore < - entryZscore))
        df1['long exit'] = ((df1.zScore > - exitZscore)) 
        df1['num units long'] = np.nan 
        df1.loc[df1['long entry'],'num units long'] = 1 
        df1.loc[df1['long exit'],'num units long'] = 0 
        df1.iat[0,df1.columns.get_loc("num units long")]= 0

        df1['num units long'] = df1['num units long'].fillna(method='pad') 

        #set up num units short 
        df1['short entry'] = ((df1.zScore >  entryZscore))
        df1['short exit'] = ((df1.zScore < exitZscore))
        df1['num units short'] = np.nan
        df1.loc[df1['short entry'],'num units short'] = -1 
        df1.loc[df1['short exit'],'num units short'] = 0
        df1.iat[0,df1.columns.get_loc("num units short")]= 0

        df1['num units short'] = df1['num units short'].fillna(method='pad')
        ############################################################################################################################
        df1['numUnits'] = df1['num units long'] + df1['num units short']
        #positions_ = dollar capital allocation in each ETF
        df1["positions_x"] =-1*df1["rolling_hedge_ratio"]*df1["x"]*df1["numUnits"]
        df1["positions_y"] =df1["y"]*df1["numUnits"]
        df1["price_change_x"] = df1["x"] - df1["x"].shift(1)
        df1["price_change_y"] = df1["y"] - df1["y"].shift(1)
        df1["pnl_x"] = df1["price_change_x"]*df1["positions_x"].shift(1)/df1["x"].shift(1)
        df1["pnl_y"] = df1["price_change_y"]*df1["positions_y"].shift(1)/df1["y"].shift(1)
        df1["pnl"] = df1["pnl_x"] + df1["pnl_y"] 
        df1["portfolio_cost"] = np.abs(df1["positions_x"])+np.abs(df1["positions_y"])
        df1["port_rets"]= df1["pnl"]/df1["portfolio_cost"].shift(1)
        df1["port_rets"].fillna(0, inplace=True)

        #repeat for detrended prices
        df1["positions_x_DETREND"] =-1*df1["rolling_hedge_ratio_DETREND"]*df1["x_DETREND"]*df1["numUnits"]
        df1["positions_y_DETREND"] =df1["y_DETREND"]*df1["numUnits"]
        df1["price_change_x_DETREND"] = df1["x_DETREND"] - df1["x_DETREND"].shift(1)
        df1["price_change_y_DETREND"] = df1["y"] - df1["y"].shift(1)
        df1["pnl_x_DETREND"] = df1["price_change_x_DETREND"]*df1["positions_x_DETREND"].shift(1)/df1["x_DETREND"].shift(1)
        df1["pnl_y_DETREND"] = df1["price_change_y_DETREND"]*df1["positions_y_DETREND"].shift(1)/df1["y_DETREND"].shift(1)
        df1["pnl_DETREND"] = df1["pnl_x_DETREND"] + df1["pnl_y_DETREND"] 
        df1["portfolio_cost_DETREND"] = np.abs(df1["positions_x_DETREND"])+np.abs(df1["positions_y_DETREND"])
        df1["port_rets_DETREND"]= df1["pnl_DETREND"]/df1["portfolio_cost_DETREND"].shift(1)
        df1["port_rets_DETREND"].fillna(0, inplace=True)


        df1 = df1.assign(I =np.cumprod(1+df1['port_rets'])) #this is good for pct return or log return
        df1.iat[0,df1.columns.get_loc('I')]= 1
        start_val = 1
        end_val = df1['I'].iat[-1]

        start_date = df1.iloc[0].Date
        end_date = df1.iloc[-1].Date
        days = (end_date - start_date).days

        periods = 360
        trading_periods = 254

        TotaAnnReturn = (end_val-start_val)/start_val/(days/periods)
        TotaAnnReturn_trading = (end_val-start_val)/start_val/(days/trading_periods)

        CAGRdbl_trading = round(((float(end_val) / float(start_val)) ** (1/(days/trading_periods))).real - 1,4) #when raised to an exponent I am getting a complex number, I need only the real part
        CAGRdbl = round(((float(end_val) / float(start_val)) ** (1/(days/periods))).real - 1,4) #when raised to an exponent I am getting a complex number, I need only the real part

        try:
            sharpe =  (df1['port_rets'].mean()/ (df1['port_rets'].std()) * np.sqrt(trading_periods))
        except ZeroDivisionError:
            sharpe = 0.0

        return TotaAnnReturn, TotaAnnReturn_trading, CAGRdbl, CAGRdbl_trading, sharpe