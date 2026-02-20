import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

def ewa_volatility(returns, EWA):
    """mean_returns = returns.ewm(alpha=EWA).mean()
    var_ewa = (returns - mean_returns) ** 2
    vol_ewa = np.sqrt(var_ewa.ewm(alpha=EWA).mean()) * np.sqrt(252)"""

    r=returns**2
    vol_ewa=np.sqrt(r.ewm(alpha=EWA).mean()*252)
    return vol_ewa


def index_SP_avec_risque(ticker:str,period:str,risk:int,EWA:float,plot:bool):
    risk_free_rate=0.043
    daily_risk_free_rate=np.exp(np.log(1+risk_free_rate)/252)-1
    stock=yf.Ticker(ticker)
    history=stock.history(period)
    returns=history.pct_change()["Close"]
    
    df=returns.to_frame(name="Returns")
    df.insert(0, "Close", history["Close"])

    df["Volatility(per year)"]=ewa_volatility(df["Returns"],EWA=EWA)             ####returns.rolling(window).std().ffill()*np.sqrt(252)
    df["Controled returns"]=df.apply(lambda row: min(1,(risk*0.01)/row["Volatility(per year)"])*row["Returns"]+(1-min(1,(risk*0.01)/row["Volatility(per year)"]))*daily_risk_free_rate , axis=1)#if row["Annualized std"]>1e-4 else row["Returns"]
    df["Controled Volatility"]=ewa_volatility(df["Controled returns"],EWA=EWA)
    df["Spot"]=100*((1+df["Returns"].fillna(0)).cumprod())
    df["Controled Spot"]=100*((1+df["Controled returns"].fillna(0)).cumprod())

    df["Cash returns"]=1+daily_risk_free_rate
    df["Cash returns"]=100*(df["Cash returns"].cumprod())

    if plot:
        df["Spot"].plot()
        df["Controled Spot"].plot()
        df["Cash returns"].plot()

        plt.figure()
        
        df["Volatility(per year)"].plot()
        df["Controled Volatility"].plot()
        plt.show()
        
        print(df)
    else:
        print(df)

index_SP_avec_risque("^GSPC","6mo",10,20,0.4,plot=True)
