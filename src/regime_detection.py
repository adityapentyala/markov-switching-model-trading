from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
import pandas as pd
from scipy.signal import find_peaks, peak_prominences
import numpy as np
import matplotlib.pyplot as plt

def get_msm(data, order=1, n_regimes=2):
    msm = MarkovAutoregression(data, n_regimes, order).fit()
    return msm

def identify_bull_bear_states(df, threshold=0.2):
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
    print(sum(turning_points), len(turning_points))

    # Calculate mean and std deviation for each state
    stats = df.groupby('State')['returns'].agg(['mean', 'std', 'count']).reset_index()
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

'''def bry_boschan_algorithm(df, threshold = 0.2, min_cycle = 12, min_duration = 5):
    series = df['returns']

    peaks = find_peaks(series, distance=min_duration*2)
    _peak_prominences = peak_prominences(series, peaks)

    troughs = find_peaks(-1*series, distance=min_duration*2)
    _trough_prominences = peak_prominences(-1*series, troughs)

    prominence_dictionary = {}

    for i in range(0, len(peaks)):
        prominence_dictionary[peaks[i]] = _peak_prominences[i]
        prominence_dictionary[troughs[i]] = _trough_prominences[i]*-1

    data_peaks = [1 if i in peaks else 0 for i in range(0, len(series))]
    data_troughs = [-1 if i in troughs else 0 for i in range(0, len(series))]

    data_extrema = [data_peaks[i]+data_troughs[i] for i in range(0, len(data_peaks))]
    regimes = [' ' for i in range(0, len(series))]
    p1 = min(peaks[0], troughs[0])
    p2 = p1+1
    curr_regime = 'bull' if prominence_dictionary[p1]>0 else 'bear'

    cycle_duration = 1
    regime_duration = 1

    running = True
    while running:
        if data_extrema[p2] == 0:
            cycle_duration+=1
            regime_duration+=1
            regimes[p2] = curr_regime
            p2+=1
'''

class TurningPoint(object):
    def __init__(self, dti, sta, val):
        """Init turning point object

        :param dti: datetime of the turning point
        :type dti: datetime object
        :param sta: status of turning point: P / T
        :type sta: str
        :param val: value of turning point
        :type val: int
        """
        self.dti = dti
        self.sta = sta
        self.val = val

    def __repr__(self):
        return "<%s: %s: %.2f>" % (self.sta, self.dti, self.val)

    def __gt__(self, other):
        return self.val > other.val

    def __lt__(self, other):
        return self.val < other.val

    def __eq__(self, other):
        return self.val == other.val

    def __ge__(self, other):
        return self.val >= other.val

    def __le__(self, other):
        return self.val <= other.val

    def time_diff(self, other):
        return abs((self.dti - other.dti).n)


def duration_check(turnings, min_dur, verbose=False):
    """Check for minimum duration of a cycle

    :param turnings: list of TurningPoint objects
    :type turnings: list
    :param min_dur: minimum cycle duration
    :type min_dur: int
    :return: list of evaluated TurningPoint objects
    :rtype: list
    """
    i = 0

    while i < len(turnings) - 2:
        printed = None
        if TurningPoint.time_diff(turnings[i], turnings[i + 2]) < min_dur:
            if turnings[i].sta == "P":
                if turnings[i + 2] >= turnings[i]:
                    printed = turnings.pop(i)
                else:
                    printed = turnings.pop(i + 2)
            else:
                if turnings[i] >= turnings[i + 1]:
                    printed = turnings.pop(i)
                else:
                    printed = turnings.pop(i + 1)
            if verbose:
                print("remove %s: failed duration check" % printed)

            turnings = alternation_check(turnings, verbose)
            turnings = duration_check(turnings, min_dur, verbose)

        else:
            i += 1
    return turnings


def alternation_check(turnings, verbose=False):
    """Check for Alternation of Peaks and Troughs

    :param turnings: list of TurningPoint objects
    :type turnings: list
    """
    i = 0
    while i < len(turnings) - 1:
        printed = None
        if turnings[i].sta == turnings[i + 1].sta:
            if turnings[i].sta == "P":
                if turnings[i] <= turnings[i + 1]:
                    printed = turnings.pop(i)
                else:
                    printed = turnings.pop(i + 1)
                    i += 1
            elif turnings[i].sta == "T":
                if turnings[i] >= turnings[i + 1]:
                    printed = turnings.pop(i)
                else:
                    printed = turnings.pop(i + 1)
                    i += 1
            if verbose:
                print("remove %s: failed alternation check" % printed)
        else:
            i += 1
    return turnings


def phase_check(turnings, min_pha, verbose=False):
    """
    Check for minimum duration of phase

    :param turnings: list of TurningPoint objects
    :type turnings: list
    :param min_pha: minimum phase duration
    :type min_pha: int
    :return: list of evaluated TurningPoint objects
    :rtype: list
    """
    i = 0
    while i < len(turnings) - 1:
        printed = None
        if TurningPoint.time_diff(turnings[i], turnings[i + 1]) < min_pha:
            printed = turnings.pop(i + 1)
            if verbose:
                print("remove %s: failed phase_check" % printed)
            turnings = alternation_check(turnings)
            turnings = phase_check(turnings, min_pha)
        else:
            i += 1
    return turnings


def start_end_check(turnings, curve, min_boudary, verbose=False):
    """
    Remove turns that too close to the begining and end of the series

    :param turnings: list of TurningPoint objects
    :type turnings: list
    :param curve: The curve to check
    :type curve: BBSeries object
    :param min_boudary: minimum duration in the start and end of the series
    :type min_boudary: int
    :return: list of evaluated TurningPoint objects
    :rtype: list
    """
    printed = None
    if turnings[0].dti < curve.beginDate + min_boudary:
        printed = turnings.pop(0)
    if turnings[-1].dti > curve.endDate - min_boudary:
        printed = turnings.pop(-1)
    if verbose and printed:
        print("remove %s: falied start_end_check" % printed)
    return turnings

seplen = 60
sepchr = "-"

class BBSeries(object):
    def __init__(self, series, start, freq, name="original"):
        """Init BB object

        :param series: time series for dating
        :type series: pandas series object
        """
        self.series = series
        self.start = start
        self.freq = freq
        self.idx = pd.period_range(
            start=self.start, periods=len(self.series), freq=self.freq
        )
        self.series.index = self.idx
        self.name = name
        self.beginDate = series.index[0]
        self.endDate = series.index[-1]

    def rm_outliers(self, threshold, verbose=False):
        spcer = self.draw_spencer()
        stdv = spcer.series.std()
        mean = spcer.series.mean()
        upper = mean + threshold * stdv
        lower = mean - threshold * stdv
        outliers = self.series.loc[(self.series > upper) | (self.series < lower)]
        if verbose and len(outliers) > 0:
            print("Outliers:\n%s" % outliers)
        return self.series.clip(lower, upper)

    def draw_ma(self, *parg, window):
        """draw the moving average curve

        :param window: number of time-interval to calculate moving average
        :type window: int
        :return: BBSeries object of moving average curve
        :rtype: BBSeries object
        """
        series = parg[0] if parg else self.series
        ma = series.rolling(window=window).mean().dropna()
        return BBSeries(ma, self.start, self.freq, "ma-%s" % window)

    def draw_spencer(self, *parg, window=5, weights=[-3, 12, 17, 12, -3]):
        """draw spencer curve

        :param window: number of time interval for spencer curve, defaults to 5
        :type window: int, optional
        :param weights: weight of spencer curve, defaults to [-3, 12, 17, 12, -3]
        :type weights: list, optional
        :return: BBSeries object of spencer curve
        :rtype: BBSeries object
        """
        series = parg[0] if parg else self.series
        spencer = series.rolling(window=window, center=True).apply(
            lambda seq: np.average(seq, weights=weights)
        )
        return BBSeries(
            spencer, self.start, self.freq, "spencer-curve-%s-%s" % (window, weights)
        )

    def _maxima(self, width):
        """find local peak

        :param width: area on either side
        :type width: int
        :return: list of peaks
        :rtype: list
        """
        peaks = []
        for i in self.series.index[width:-width]:
            if max(self.series[i - width : i + width]) == self.series[i]:
                #print(i)
                peaks.append(TurningPoint(i, "P", self.series[i]))
        return peaks

    def _minima(self, width):
        """find local troughs

        :param width: area on either side
        :type width: int
        :return: list of peaks
        :rtype: list
        """
        troughs = []
        for i in self.series.index[width:-width]:
            if min(self.series[i - width : i + width]) == self.series[i]:
                #print(i)
                troughs.append(TurningPoint(i, "T", self.series[i]))
        return troughs

    def get_turnings(self, width):

        """Identify turning points

        :param width: area on either side of checking point
        :type width: int
        :return: list of Turning Points sorted by time
        :rtype: list
        """
        peaks = self._maxima(width)
        troughs = self._minima(width)
        return sorted((peaks + troughs), key=lambda x: x.dti)

    def re_apply(self, turnings, width):
        """apply set of turning points to second curve

        :param turnings: list of turning points from previous curve
        :type turnings: list
        :param width: area on either side
        :type width: int
        """
        new = []
        for p in turnings:
            temp = self.series[p.dti - width : p.dti + width]
            id = temp.idxmax() if p.sta == "P" else temp.idxmin()
            new.append(TurningPoint(id, p.sta, temp[id]))
        return new

    def plot_turns(self, turns):
        """plot the curve and turning points

        :param curve: BBSeries object
        :type curve: BBseries
        :param turns: list of TurningPoint objects
        :type turns: list
        """
        fig, ax = plt.subplots()
        self.series.plot()
        for turn in turns:
            ax.annotate("x", xy=(turn.dti, turn.val))
        plt.show()

    def dating(
        self,
        ma=[],
        width=3,
        min_dur=6,
        min_pha=3,
        min_boudary=3,
        threshold=3.5,
        spencer=False,
        rm_outliers=False,
        verbose=False,
    ):
        """
        Bry and Boschan dating process for identifying turning points

        :param ma: list of ma curves to draw
        :type ma: list
        :param width: area on either side, defaults to 3
        :type width: int, optional
        :param min_dur: minimum duration of a cycle, defaults to 6
        :type min_dur: int, optional
        :param min_pha: minimum duration of a phase, defaults to 3
        :type min_pha: int, optional
        :param min_boudary: minimum start and end of the series, defaults to 3
        :type min_boudary: int, optional
        :param verbose: to print out result and plot the line, defaults to True
        :type verbose: bool, optional
        :return: list of final TurningPoint objects
        :rtype: list
        """

        sepline = sepchr * seplen

        series = (
            self.rm_outliers(threshold=threshold, verbose=verbose)
            if rm_outliers
            else self.series
        )

        curves = [
            self.draw_ma(series, window=window) for window in sorted(ma, reverse=True)
        ]
        curves.append(self)

        if spencer and ma:
            curves.insert(1, self.draw_spencer(series))

        for idx, curve in enumerate(curves):
            if idx == 0:
                turns = curve.get_turnings(width=width)
            else:
                turns = curve.re_apply(turns, width=width)

            if verbose:
                # print initial information
                print(sepline)
                print(
                    "dating: %s | width: %s | min_dur: %s | min_pha: %s"
                    % (curve.name, width, min_dur, min_pha)
                )

            alternation_check(turns, verbose=verbose)
            duration_check(turns, min_dur=min_dur, verbose=verbose)
            phase_check(turns, min_pha=min_pha, verbose=verbose)

            if curve.name == "original":
                start_end_check(
                    turns, self, min_boudary=min_boudary, verbose=verbose
                )

            if verbose:
                # print result cycle
                for num, turn in enumerate(turns):
                    print("(%02d) %s" % (num, turn))
                # plot turns on the series
                curve.plot_turns(turns)
        return turns


def dating(
    series,
    ma=[],
    width=5,
    min_dur=16,
    min_pha=5,
    min_boudary=5,
    threshold=3.5,
    spencer=False,
    rm_outliers=False,
    verbose=True,
):
    """
    Bry and Boschan dating process for identifying turning points

    :param s: detrend seasonal adjusted original series for dating process
    :type s: pandas time series
    :param width: area on either side, defaults to 3
    :type width: int, optional
    :param min_dur: minimum duration of a cycle, defaults to 6
    :type min_dur: int, optional
    :param min_pha: minimum duration of a phase, defaults to 3
    :type min_pha: int, optional
    :param min_boudary: minimum start and end of the series, defaults to 3
    :type min_boudary: int, optional
    :param verbose: to print out result and plot the line, defaults to True
    :type verbose: bool, optional
    :return: list of final TurningPoint objects
    :rtype: list
    """
    original = BBSeries(series, series.index[0], 'M')
    turns = original.dating(
        ma=ma,
        width=width,
        min_dur=min_dur,
        min_pha=min_pha,
        min_boudary=min_boudary,
        threshold=threshold,
        spencer=spencer,
        rm_outliers=rm_outliers,
        verbose=verbose,
    )

    return turns

def assign_regimes(df, turningpoints):
    curr_regime = 0 if turningpoints[0].sta=='P' else 1
    extrema_count=0
    states = []
    for i in range(0, len(df)):
        if pd.to_datetime(df.index[i], utc=False)==pd.to_datetime(turningpoints[extrema_count].dti.to_timestamp()).tz_localize('Asia/Kolkata') and extrema_count<len(turningpoints)-1:
            print(pd.to_datetime(df.index[i], utc=False), pd.to_datetime(turningpoints[extrema_count].dti.to_timestamp()).tz_localize('Asia/Kolkata'))
            curr_regime= 0 if turningpoints[extrema_count].sta=='P' else 1
            extrema_count+=1 if extrema_count!=len(turningpoints)-1 else extrema_count
        states.append(curr_regime)
    df['State'] = states
    print(extrema_count)