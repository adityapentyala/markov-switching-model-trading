from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
import pandas as pd

def get_msm(data, order=1, n_regimes=2):
    msm = MarkovAutoregression(data, n_regimes, order).fit()
    return msm

def identify_bull_bear_states(df, threshold=0.15):
    #df = df.sort_values('Date').reset_index(drop=True)
    df['Cumulative returns'] = (1 + df['returns']).cumprod() - 1

    states = []
    turning_points = []
    current_state = 0 if df.loc[0, 'returns'] >= 0 else 1
    states.append(current_state)
    turning_points.append(True)
    peak = df.loc[0, 'Cumulative returns']
    trough = df.loc[0, 'Cumulative returns']

    for i in range(1, len(df)):
        cum_returns = df.loc[i, 'Cumulative returns']
        if current_state == 0:
            if cum_returns <= peak - threshold:
                current_state = 1
                turning_points.append(True)
                trough = cum_returns
            else:
                turning_points.append(False)
                if cum_returns > peak:
                    peak = cum_returns
        else:
            if cum_returns >= trough + threshold:
                current_state = 0
                turning_points.append(True)
                peak = cum_returns
            else:
                turning_points.append(False)
                if cum_returns < trough:
                    trough = cum_returns
        states.append(current_state)

    df['State'] = states
    df['Turning Point'] = turning_points

    # Calculate mean and std deviation for each state
    stats = df.groupby('State')['returns'].agg(['mean', 'std']).reset_index()
    return df, stats

def get_transition_counts(df):
    a_a = 0
    a_b = 0
    b_a = 0
    b_b = 0

    for i in range(0, len(df)-1):
        if df.loc[i, 'State'] == 0 and df.loc[i+1, 'State'] == 0:
            a_a += 1
        elif df.loc[i, 'State'] == 0 and df.loc[i+1, 'State'] == 1:
            a_b += 1
        elif df.loc[i, 'State'] == 1 and df.loc[i+1, 'State'] == 0:
            b_a += 1
        elif df.loc[i, 'State'] == 1 and df.loc[i+1, 'State'] == 1:
            b_b += 1
    
    total = a_a + a_b + b_a + b_b
    
    transition_matrix = [[a_a/(a_a+a_b), a_b/(a_a+a_b)], [b_a/(b_a+b_b), b_b/(b_a+b_b)]]
    return transition_matrix, total