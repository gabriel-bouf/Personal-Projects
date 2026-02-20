import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from math import sqrt, exp,log
from scipy.optimize import minimize
from mistralai import Mistral
#from mistralai.models import ChatMessage
import json

def constraints(weights):
    return np.sum(weights)-1 #contrainte=0 dans minimise; ici on veut 100% du portefeuille en actions

def average_daily_portfolio_infos(weights:list,returns:pd.DataFrame):
    # [portfolio_return,portfolio_risk]
    return [np.dot(weights, returns.mean()),sqrt(np.dot(weights, np.dot(returns.cov(), weights)))]

def sharpe_ratio(weights:list,returns: pd.DataFrame):
    daily_risk_free_rate=exp(log(1+0.03)/252)-1
    # constant = 3% over a year (252 days)
    p_return=average_daily_portfolio_infos(weights,returns)[0]
    p_risk=average_daily_portfolio_infos(weights,returns)[1]
    return (p_return-daily_risk_free_rate)/p_risk

def minimizer_sharpe_ratio(weights:list,returns:pd.DataFrame):
    return -sharpe_ratio(weights,returns)
    
def optimizator(tickers:list,start,end):
    returns=yf.Tickers(tickers).history(start=start,end=end).pct_change(fill_method=None)['Close']
    initial_weights=[1/len(tickers) for elt in tickers]
    bounds = [(0, 1) for elt in tickers]
    c={'type': 'eq', 'fun': constraints}
    x0=np.array(initial_weights)
    opti=minimize(minimizer_sharpe_ratio,x0,args=returns,bounds=bounds,constraints=c)
    #opti.x is a np tab
    return opti.x.tolist(), returns

def random_portfolio(returns,n):
    n_actions = returns.shape[1]
    couples=[]
    for i in range(n):
        weight = np.random.random(n_actions)
        weight /= np.sum(weight)
        couples.append(average_daily_portfolio_infos(weight,returns))
    return couples

def print_portfolios(couples,optimal_portfolio):
    returns, risks = zip(*couples)#* pour décompresser
    returns = np.array(returns)
    risks = np.array(risks)
    daily_risk_free_rate=exp(log(1+0.03)/252)-1
    # constant = 3% over a year (252 days)

    plt.figure(figsize=(10, 10))
    plt.scatter(risks, returns, c=(returns-daily_risk_free_rate)/risks, marker='o',s=5)
    plt.scatter(optimal_portfolio[1], optimal_portfolio[0], color='red', marker='o', s=30)
    plt.title('Efficient portfolio frontier\nOptimal portfolio in red')
    plt.xlabel('Risk')
    plt.ylabel('Daily expected return')
    plt.colorbar(label='Sharpe ratio')
    plt.show()    

def global_print(tickers,start,end,n_portfolio):
    opti_weights,returns=optimizator(tickers,start,end)
    opti_weights= [round(x,3) for x in opti_weights]
    filtered_opti_set =[(tickers[i], opti_weights[i]) for i in range(len(opti_weights)) if opti_weights[i] > 0]
    deleted_opti_set=[tickers[i] for i in range(len(opti_weights)) if opti_weights[i] == 0]
    
    opti_couple=average_daily_portfolio_infos(opti_weights,returns)
    print("Optimal set :\n",filtered_opti_set)
    print("\n Deleted tickers : \n",deleted_opti_set)
    print(f'This portfolio is supposed to have a daily return of {round(opti_couple[0]*100,4)}% and a daily risk of {round(opti_couple[1]*100,4)}% based on {start} to {end}')
    couples = random_portfolio(returns, n_portfolio)
    print_portfolios(couples,opti_couple)
    return opti_weights,opti_couple[0]

def mistral_tickers_selection(n_tickers):
    #Mistral AI selects n tickers of diverse firms which had performed between start and end
    #but you need an api key
    api_key = ""
    if api_key == "":
        print("You need an api key to use Mistral")
        #If you don't have an API key, tickers are randomly choose based on this database :
        tickers_database= [
        "AAPL", "MSFT", "GOOGL", "ORCL", "IBM",     # Tech
        "JNJ", "PFE", "MRK", "UNH", "ABT",           # Santé
        "JPM", "BAC", "GS", "C", "MS",               # Services financiers
        "TSLA", "AMZN", "HD", "NKE", "MCD",          # Consommation cyclique
        "PG", "KO", "PEP", "WMT", "COST",            # Consommation de base
        "NEE", "DUK", "SO", "AEP", "EXC",            # Services publics
        "XOM", "CVX", "COP", "BP", "SLB",            # Énergie
        "BHP", "RIO", "DD", "VALE", "FCX",           # Matériaux
        "SPG", "PLD", "AMT", "PSA", "CBRE",          # Immobilier
        "BA", "CAT", "GE", "DE", "LMT",              # Industriels
        "NFLX", "DIS", "T", "VZ", "CHTR"             # Communication
        ]
        tickers=[]
        for i in range(n_tickers):
            tickers.append(tickers_database[np.random.randint(0,len(tickers_database))])
            tickers_database.remove(tickers[i])            
    model ="open-mistral-nemo-2407"
    client=Mistral(api_key=api_key)
    
    #the prompt :
    messages=[
        {
            "role": "user",
            "content": "donne moi au format JSON une liste de "+ str(n_tickers) +
            " tickers valide sur Yahoo Finance, d'entreprises françaises, européennes et internationales dans des secteurs d'activité varié, qui ont performées entre " 
            + start + " et " + end + 
            ". Tu peux aussi donner des cryptomonnaies mais pas trop ! "
            "Attention à donner des tickers valide (par exemple 'FB' n'existe pas) car certains ont récemment changer de nom et surtout donne juste le nom des tickers dans une liste, sans rien ajouter",
        }
    ]
    response=client.chat.complete(model=model,messages=messages,response_format ={"type": "json_object",})

    tickers_str=response.choices[0].message.content
    tickers = json.loads(tickers_str)
    print("\nMistral proposition :\n",tickers,"\n\n")
    #tickers=tickers_str["tickers"]
    if (type(tickers))!=list:
        print("Error during tickers recuperation")
        return []
    
    returns=yf.Tickers(tickers).history(start=start,end=end).pct_change(fill_method=None)['Close']
    #Sometimes MistralAI goes crazy 
    valid_tickers=[]
    for ticker in tickers:
        if ticker not in returns.columns: 
            print(f"\nError during the download of {ticker} (It might not exist)\n")
        elif returns[ticker].isna().all():
            print(f"\nError during the download of {ticker} (It might not exist)\n")
        else:
            valid_tickers.append(ticker)
    return valid_tickers

def backtest(weights:list,tickers:list,expected_return,test_start,test_end):
    returns=yf.Tickers(tickers).history(start=test_start,end=test_end)['Close']
    returns.index=pd.to_datetime(returns.index)#convert returns' index into a Datetime
    backtest=100*returns/returns.iloc[0] #normalize
    backtest=backtest.ffill() #different timezone leads to missing data
    weights=np.array(weights)
    #print(backtest)
    backtest["Portfolio Close"]=(weights*backtest).sum(axis=1)
    print("Return of the portfolio on the tested period : \n", round(100*(backtest["Portfolio Close"].iloc[-1]/100-1),3),"%")
    expected_return_over_period=((1+expected_return)**(returns.shape[0])-1)*100
    print("Expected return of the portfolio over the period :",round(expected_return_over_period,3),"%")
    ax=backtest.drop(columns="Portfolio Close").plot(title="Portfolio Backtesting")
    backtest["Portfolio Close"].plot(ax=ax,color="black", label="Portfolio Strike")
    plt.show()
    return

n=10000
start = '2024-01-01'#yyyy-mm-dd
end = '2024-08-01'
test_start=end
test_end='2024-12-01'

tickers=mistral_tickers_selection(5)
rounded_opti_set,expected_return=global_print(tickers,start,end,n)
backtest(rounded_opti_set,tickers,expected_return,test_start,test_end)
