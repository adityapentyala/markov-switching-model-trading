import yfinance as yf
import pandas as pd


def get_security_info(filepath):
    df = pd.read_csv(filepath)
    pe_ratios = []
    ev_ebitdas = []
    roes = []
    to_drop = []
    iter = 0
    for symbol in df['SYMBOL']:
        symbol_data = yf.Ticker(f'{symbol}.NS').info
        pe_ratio, ev_ebitda, roe = symbol_data.get('trailingPE'), symbol_data.get('enterpriseToEbitda'), symbol_data.get('returnOnEquity')
        if pe_ratio is not None and ev_ebitda is not None and roe is not None:
            print(f"{iter} {symbol}: {pe_ratio} {ev_ebitda} {roe}")
            pe_ratios.append(pe_ratio)
            ev_ebitdas.append(ev_ebitda)
            roes.append(roe)
        else: 
            to_drop.append(iter)
            print(f"{symbol}: None")
        iter+=1

    new_df =  df.drop(to_drop)
    new_df['pe_ratio'] = pe_ratios
    new_df['roe'] = roes
    new_df['ev_ebitda'] = ev_ebitdas

    new_df.to_csv(f'../data/{filepath}_ratios.csv', index=False)

get_security_info('../data/nifty_smallcap_250.csv')