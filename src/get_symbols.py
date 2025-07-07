import json
import pandas as pd

symbols = pd.read_csv("../data/MW-NIFTY-SMALLCAP-250-05-Jul-2025.csv")["SYMBOL"]

def get_instr_keys(symbols_list, json_file):
    instr_keys = {}
    with open(json_file, "r") as file: 
        data = json.load(file)
    file.close()

    count = 0
    for symbol in symbols_list:
        count+=1
        found = False
        for object in data:
            if object["trading_symbol"] == symbol:
                instr_keys[symbol] = object["instrument_key"]
                print(f"{count}: {symbol} instrument key found")
                found=True
                break
        if found==False:
            print("{count}: {symbol} instrument key found")
    
    return instr_keys

instr_keys = get_instr_keys(symbols, "NSE.json")

instr_df = pd.DataFrame(instr_keys, index=[0])

instr_df.T.to_csv("../data/instrumentkeys.csv")