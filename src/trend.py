from utils import indicator_signal 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

def generate_signals(data, ar_weights):
    signals = []
    for i in range(0, len(data)-20):
        print("\n\n this is signal", i, "\n\n")
        signal = indicator_signal(ar_weights, np.asarray(data['returns'][i:i+20].reset_index(drop=True)), p=20, a=25)
        signals.append(signal)
    return signals


def get_action_dates(data, signals):
    buy_dates = []
    sell_dates = []
    for i in range(0, len(signals)):
        if signals[i] > 0:
            buy_dates.append(data['date'][20+i])
        elif signals[i] < 0:
            sell_dates.append(data['date'][20+i])
    return buy_dates, sell_dates

def plot_actions(data, buy_dates, sell_dates, signals, n_dates=40, n_actions=10):
    plt.figure()
    plt.plot(data['date'][20:n_dates], data['returns'][20:n_dates])
    plt.vlines(buy_dates[:n_actions], -0.05, 0.05, color='green')
    plt.vlines(sell_dates[:n_actions], -0.05, 0.05, color='red')
    plt.scatter(data['date'][20:n_dates], signals[:20], color='orange')
    plt.xticks(rotation=70)
    plt.show()

