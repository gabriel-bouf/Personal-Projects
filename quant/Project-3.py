import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from math import exp,floor
from scipy.optimize import minimize
from scipy.interpolate import griddata
import QuantLib as ql

def skeww(N_call,K,ticker,start,end,vol):

    stock=yf.Ticker(ticker)
    history=stock.history(start=start,end=end)
    spot_price=history['Close'].iloc[0]
    

    today= ql.Date(int(start[8:]),int(start[5:7]),int(start[0:4])) #DD/MM/YYYY
    ql.Settings.instance().evaluationDate = today
    expiry =ql.Date(int(end[8:]),int(end[5:7]),int(end[0:4]))
    option_type= ql.Option.Call
    payoff = ql.PlainVanillaPayoff(option_type, K)
    exercise =ql.EuropeanExercise(expiry)
    european_option= ql.VanillaOption(payoff,exercise)
 
    dividend_rate=0
    risk_free_rate=0.043
    daily_risk_free_rate=(1+risk_free_rate)**(1/365)-1
    commission_rate = 0.0005
    bid_ask_spread = 0.0002  
    
    spot_quote=ql.SimpleQuote(spot_price)
    spot_handle= ql.QuoteHandle(spot_quote)
    volatility_handle= ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today,ql.NullCalendar(),ql.QuoteHandle(ql.SimpleQuote(vol)),ql.Actual365Fixed()))
    dividend_handle= ql.YieldTermStructureHandle(ql.FlatForward(today,ql.QuoteHandle(ql.SimpleQuote(dividend_rate)),ql.Actual365Fixed()))
    risk_free_handle= ql.YieldTermStructureHandle(ql.FlatForward(today,ql.QuoteHandle(ql.SimpleQuote(risk_free_rate)),ql.Actual365Fixed()))
    bsm_process =ql.BlackScholesMertonProcess(spot_handle,dividend_handle,risk_free_handle,volatility_handle)
    european_option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))
    
    option_price=european_option.NPV()
    delta=european_option.delta()
    P_and_L=N_call*option_price-delta*N_call*spot_price

    for i in range(1,history.shape[0]):
        last_delta=delta
        spot_price =history['Close'].iloc[i]
        spot_quote.setValue(spot_price)
        option_price=european_option.NPV()
        delta=european_option.delta()
        P_and_L-=(delta-last_delta)*N_call*spot_price
        
        P_and_L-=abs(delta-last_delta)*N_call*spot_price*(commission_rate+bid_ask_spread)#transaction costs

        if P_and_L<0:
            P_and_L*=exp(daily_risk_free_rate)#loan interest

    if K<spot_price:
        P_and_L+=K*N_call

    if K>spot_price:
        P_and_L += delta * N_call * spot_price

    return P_and_L


def find_break_even_vol(N_call, K, ticker, start, end):
    def loss_function(vol):
        return abs(skeww(N_call, K, ticker, start, end,float(vol[0])))
    
    result = minimize(loss_function, x0=0.2, bounds=[(0.01, 1.0)])
    return result.x[0] if result.success else None


def volatility_skew(ticker, start, N_call, n):
    spot_price = yf.Ticker(ticker).history(start=start, period="1mo")['Close'].iloc[0]
    strikes = np.linspace(floor(spot_price * 0.8), floor(spot_price * 1.2), n)
    start_date = pd.Timestamp(start)
    maturities = [i for i in range(1,24)]
    end_dates = [start_date + pd.DateOffset(months=m) for m in maturities]

    strikes_mesh, maturities_mesh, bevs_mesh = [], [], []
    for strike in strikes:
        for end in end_dates:
            bev = find_break_even_vol(N_call, strike, ticker, start, end.strftime('%Y-%m-%d'))
            if bev != None:
                strikes_mesh.append(strike)
                maturities_mesh.append((end - start_date).days)  # Convertir en jours
                bevs_mesh.append(bev)

    grid_x, grid_y = np.meshgrid(np.linspace(min(strikes_mesh), max(strikes_mesh), n),
                                 np.linspace(min(maturities_mesh), max(maturities_mesh), n))
    grid_z = griddata((strikes_mesh, maturities_mesh), bevs_mesh, (grid_x, grid_y), method='cubic')

    # Affichage des résultats
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', alpha=0.5)
    ax.scatter(strikes_mesh, maturities_mesh, bevs_mesh, c=bevs_mesh, cmap='viridis', marker='o')
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Time to Maturity (days)')
    ax.set_zlabel('Break-even Volatility')
    ax.set_title('Break-even Volatility Surface')
    plt.show()


start = '2023-01-01'#yyyy-mm-dd
end = '2024-04-01'
bev = find_break_even_vol(100, 110, 'AAPL', start, end)
#print("Break-even volatility:", round(100*bev,3),"%")
#print(" P&L associate:",skeww(100, 110, 'AAPL', start, end,bev))


volatility_skew("AAPL", '2023-01-01', 100, 20)
