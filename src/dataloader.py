import pandas as pd
import requests
import json

def get_candle_data(instrument, interval, to_date, from_date, frequency=1):
    url = f"https://api.upstox.com/v3/historical-candle/{instrument}/{interval}/{frequency}/{to_date}/{from_date}"
    print(url)
    headers = {
    'Accept': 'application/json'
    }
    response = requests.get(url, headers=headers)
    # Check the response status
    if response.status_code == 200:
        # Do something with the response data (e.g., print it)
        print(response.json())
    else:
        # Print an error message if the request was not successful
        print(f"Error: {response.status_code} - {response.text}")

    response_dict = response.json()
    df = pd.DataFrame(response_dict['data']['candles'], columns=['date', 'open', 'high', 'low', 'close', 'vol', 'misc'])
    df.set_index(pd.DatetimeIndex(df['date']), inplace=True)

    return df