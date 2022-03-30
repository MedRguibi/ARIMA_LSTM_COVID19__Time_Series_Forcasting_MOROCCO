import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

from scipy import stats as sps
from scipy.interpolate import interp1d

from IPython.display import clear_output
from datetime import timedelta

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

mpl.rc('font', **font)

mpl.rcParams['axes.grid']=True
plt.rcParams.update({'figure.figsize':(8, 5), 'figure.dpi':120})

mpl.rcParams['axes.grid']=True
pd.options.display.max_rows = 999

########################################## Import Actual Data ##################################

confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
death_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recovered_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

def prepare(df, name):
    cols_to_melt = df.columns.values[4:]
    res = pd.melt(df, id_vars='Country/Region', var_name='date', value_vars=cols_to_melt, value_name=name)
    res['date'] = pd.to_datetime(res['date'])
    res = res.sort_values(by = ['Country/Region', 'date'])
    res = res.set_index('date')
    res.columns = ['country', 'target']
    return res

confirmed_df = prepare(confirmed_df, 'target')
death_df = prepare(death_df, 'target')
recovered_df = prepare(recovered_df, 'target')

######################################## Data visualisation ###############################################################

#print('confirmed:',confirmed_df.shape)
#print('deaths:', death_df.shape)
#print('recovered:', recovered_df.shape)

# print('confirmed dates:')
# print(confirmed_df.index.min())
# print(confirmed_df.index.max())
# print(confirmed_df.index.max() - confirmed_df.index.min())

# print('Recovered dates:')
# print(recovered_df.index.min())
# print(recovered_df.index.max())
# print(recovered_df.index.max() - recovered_df.index.min())

# print('deaths dates:')
# print(death_df.index.min())
# print(death_df.index.max())
# print(death_df.index.max() - death_df.index.min())

########################################### Plot Set-up functions ####################################################

def plot2series(df, split, title = '', col = 'target', save_name = 'img', color_line = 'red', label_name = 'Actual data'):
    df1 = df.iloc[0:split, :]
    df2 = df.iloc[split -1:,:]
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    plt.plot(df1[col], color = color_line, linestyle='dashed', marker='o', linewidth=2, markersize=6, label = label_name)
    #plt.plot(df2[col], color = 'blue', linewidth=2, markersize=12, label = 'Predicted')
    x_ticks = np.linspace(df1.index.min().value, df2.index.max().value, 5)
    x_ticks = pd.to_datetime(x_ticks)
    plt.xticks(x_ticks, rotation = 0)
    y_ticks = np.linspace(df1[col].min(), df2[col].max(), 10)
    plt.yticks(y_ticks)
    plt.legend(loc='best')
    # plt.title(title)
    ax.grid(True)
    #plt.savefig(save_name + '.png')
    plt.show()

######################################### Plot Rt ##################################################
def plot_rt(result, ax, state_name):
    # ax.set_title(f"{state_name}")
    # Colors
    ABOVE = [1,0,0]
    MIDDLE = [1,1,1]
    BELOW = [0,0,0]
    cmap = ListedColormap(np.r_[
        np.linspace(BELOW,MIDDLE,25),
        np.linspace(MIDDLE,ABOVE,25)
    ])
    color_mapped = lambda y: np.clip(y, .5, 1.5)-.5
    
    index = result['ML'].index.get_level_values('date')
    values = result['ML'].values
    
    # Plot dots and line
    ax.plot(index, values, c='k', zorder=1, alpha=.25)
    ax.scatter(index,
               values,
               s=40,
               lw=.5,
               c=cmap(color_mapped(values)),
               edgecolors='k', zorder=2)
    
    # Aesthetically, extrapolate credible interval by 1 day either side
    lowfn = interp1d(date2num(index),
                     result['Low'].values,
                     bounds_error=False,
                     fill_value='extrapolate')
    
    highfn = interp1d(date2num(index),
                      result['High'].values,
                      bounds_error=False,
                      fill_value='extrapolate')
    
    '''Edit'''
    extended = pd.date_range(start=pd.Timestamp('2020-02-22'),
                             end=index[-1]+pd.Timedelta(days=1))
    '''Edit'''
    
    ax.fill_between(extended,
                    lowfn(date2num(extended)),
                    highfn(date2num(extended)),
                    color='k',
                    alpha=.1,
                    lw=0,
                    zorder=3)

    ax.axhline(1.0, c='k', lw=1, label='$R_t=1.0$', alpha=.25);
    
    # Formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    ax.yaxis.tick_right()
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.margins(0)
    ax.grid(which='major', axis='y', c='k', alpha=.1, zorder=-2)
    ax.margins(0)
    ax.set_ylim(0.0,3.5)

    '''Edit'''
    ax.set_xlim(pd.Timestamp('2020-02-22'), result.index.get_level_values('date')[-1]+pd.Timedelta(days=1))
    '''Edit'''

    fig.set_facecolor('w')
#########################################################################################################################

#forecasting Confirmed Cases in Morocco
country =  'Morocco'  ## you can use any other country name For Example : Brazil, Italy, Spain
title = 'Covid-19 Confirmed Cases in ' + country  ## Title of the plot 
country_df = confirmed_df[confirmed_df['country'] == country] ## Choose the right columns for the choosing country
country_df = country_df.drop(columns=['country']) ## Drop the columns name, keeping only data
split = len(country_df) ## Size of data
forcast = country_df.copy() ## Copy the data in forcast variable
res_Confirmed = forcast
plot2series(res_Confirmed, split, title, color_line = 'blue', label_name = 'Actual Confirmed Cases') ## Plot results 
# forecasting Recoveries Cases in Morocco
title = 'Covid-19 Recovered Cases in ' + country
country_df = recovered_df[recovered_df['country'] == country]
country_df = country_df.drop(columns=['country'])
split = len(country_df)
res_Recovred = country_df 
plot2series(res_Recovred, split, title, color_line = 'green', label_name = 'Actual Recovered Cases')
## forecasting Death Cases in Morocco
title = 'Covid-19 Death Cases in ' + country
country_df = death_df[death_df['country'] == country]
country_df = country_df.drop(columns=['country'])
split = len(country_df)
forcast = country_df.copy()
res_Death = forcast
plot2series(res_Death, split, title, color_line = 'red', label_name = 'Actual Death Cases')
############################################################# Save data to csv ##########################################
# res_Confirmed.to_csv('ConfirmedCases.csv')
# res_Recovred.to_csv('RecovredCases.csv')
# res_Death.to_csv('DeathCases.csv')
############################################################# descriptive statistics ##########################################
print(res_Confirmed.describe())
print('standard error = ',res_Confirmed.sem())
print('skew = ',res_Confirmed.skew())
print('kurt = ',res_Confirmed.kurt())

print(res_Death.describe())
print('standard error = ',res_Death.sem())
print('skew = ',res_Death.skew())
print('kurt = ',res_Death.kurt())
############################################################# Rt ###############################################################

# Column vector of k
k = np.arange(0, 70)[:, None]
# Different values of Lambda
lambdas = [10, 20, 30, 40]
# Evaluated the Probability Mass Function (remember: poisson is discrete)
y = sps.poisson.pmf(k, lambdas)
k = np.array([20, 40, 55, 90])
# We create an array for every possible value of Rt
R_T_MAX = 12
r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)
# Gamma is 1/serial interval
# https://wwwnc.cdc.gov/eid/article/26/7/20-0282_article
# https://www.nejm.org/doi/full/10.1056/NEJMoa2001316
GAMMA = 1/4
# Map Rt into lambda so we can substitute it into the equation below
# Note that we have N-1 lambdas because on the first day of an outbreak
# you do not know what to expect.
lam = k[:-1] * np.exp(GAMMA * (r_t_range[:, None] - 1))
# Evaluate the likelihood on each day and normalize sum of each day to 1.0
likelihood_r_t = sps.poisson.pmf(k[1:], lam)
likelihood_r_t /= np.sum(likelihood_r_t, axis=0)
# Start formating data produce daily COVID New cases 
test = res_Confirmed.copy()
split = len(test)

morocco = pd.Series(dtype='float64')
morocco = test 

def prepare_cases(cases):
    new_cases = cases.diff()

    smoothed = new_cases.rolling(10,
        win_type='gaussian',
        min_periods=1,
        center=True).mean(std=2).round()
    
    zeros = smoothed.index[smoothed.eq(0)]
    if len(zeros) == 0:
        idx_start = 0
    else:
        last_zero = zeros.max()
        idx_start = smoothed.index.get_loc(last_zero) + 1
    smoothed = smoothed.iloc[idx_start:]
    original = new_cases.loc[smoothed.index]
    
    return original, smoothed

original, smoothed = prepare_cases(morocco.target)


def highest_density_interval(pmf, p=.95):
    # If we pass a DataFrame, just call this recursively on the columns
    if(isinstance(pmf, pd.DataFrame)):
        return pd.DataFrame([highest_density_interval(pmf[col]) for col in pmf],
                            index=pmf.columns)
    
    cumsum = np.cumsum(pmf.values)
    best = None
    for i, value in enumerate(cumsum):
        for j, high_value in enumerate(cumsum[i+1:]):
            if (high_value-value > p) and (not best or j<best[1]-best[0]):
                best = (i, i+j+1)
                break
            
    low = pmf.index[best[0]]
    high = pmf.index[best[1]]
    return pd.Series([low, high], index=['Low', 'High'])

def get_posteriors(sr, window=7, min_periods=1):
    lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1))

    # Note: if you want to have a Uniform prior you can use the following line instead.
    # I chose the gamma distribution because of our prior knowledge of the likely value
    # of R_t.
    
    # prior0 = np.full(len(r_t_range), np.log(1/len(r_t_range)))
    prior0 = np.log(sps.gamma(a=3).pdf(r_t_range) + 1e-14)

    likelihoods = pd.DataFrame(
        # Short-hand way of concatenating the prior and likelihoods
        data = np.c_[prior0, sps.poisson.logpmf(sr[1:].values, lam)],
        index = r_t_range,
        columns = sr.index)

    # Perform a rolling sum of log likelihoods. This is the equivalent
    # of multiplying the original distributions. Exponentiate to move
    # out of log.
    posteriors = likelihoods.rolling(window,
                                     axis=1,
                                     min_periods=min_periods).sum()
    posteriors = np.exp(posteriors)

    # Normalize to 1.0
    posteriors = posteriors.div(posteriors.sum(axis=0), axis=1)
    
    return posteriors

# Note that we're fixing sigma to a value just for the example
posteriors = get_posteriors(smoothed)

# Note that this takes a while to execute - it's not the most efficient algorithm
hdis = highest_density_interval(posteriors)
most_likely = posteriors.idxmax().rename('ML')
# Look into why you shift -1
result = pd.concat([most_likely, hdis], axis=1)
# print(result.tail(10))

fig, ax = plt.subplots(figsize=(600/72,400/72))

plot_rt(result, ax, country)
# ax.set_title(f'Real-time $R_t$ for {country}')
ax.set_ylim(0.0,3.5)
# ax.xaxis.set_major_locator(mdates.WeekdayLocator())
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.show()

############################################################# Mortality and Recovery rate ###############################################################

mortality_rate = res_Death / res_Confirmed
mortality_rate = mortality_rate['target'].rename('Mortality rate')

recovery_rate = res_Recovred / res_Confirmed
recovery_rate = recovery_rate['target'].rename('Recovery rate')

ax = mortality_rate.plot(y='Mortality rate', grid=True, figsize=(12, 6), color='red', marker='o')
plt.gca().set_yticklabels(['{:.1f}%'.format(x*100) for x in plt.gca().get_yticks()]) 
plt.legend()
plt.show()

ax = recovery_rate.plot(y='Recovery rate', grid=True, figsize=(12, 6), color='green', marker='o')
plt.gca().set_yticklabels(['{:.1f}%'.format(x*100) for x in plt.gca().get_yticks()]) 
plt.legend()
plt.show()

############################################################# ARIMA ACF PCAF ###############################################################

def showDifferencingPlots_D(df, Category = 'Confirmed Cases'):
    data = df.copy()
    data.reset_index(drop=True, inplace=True)
    # Original Series
    fig, axes = plt.subplots(3, 2, sharex=False)
    axes[0, 0].plot(data['target']); axes[0, 0].set_title('Original Series')
    plot_acf(data['target'], ax=axes[0, 1])
    # 1st Differencing
    axes[1, 0].plot(data['target'].diff()); axes[1, 0].set_title('1st Order Differencing')
    plot_acf(data['target'].diff().dropna(), ax=axes[1, 1])
    # 2nd Differencing
    axes[2, 0].plot(data['target'].diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
    plot_acf(data['target'].diff().diff().dropna(), ax=axes[2, 1])
    plt.show()

def showDifferencingPlots_AR(df, Category = 'Confirmed Cases'):
    data = df.copy()
    data.reset_index(drop=True, inplace=True)
    # Original Series
    fig, axes = plt.subplots(3, 2, sharex=False)
    axes[0, 0].plot(data['target']); axes[0, 0].set_title('Original Series')
    plot_pacf(data['target'], ax=axes[0, 1])
    # 1st Differencing
    axes[1, 0].plot(data['target'].diff()); axes[1, 0].set_title('1st Order Differencing')
    plot_pacf(data['target'].diff().dropna(), ax=axes[1, 1])
    # 2nd Differencing
    axes[2, 0].plot(data['target'].diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
    plot_pacf(data['target'].diff().diff().dropna(), ax=axes[2, 1])
    plt.show()

def showPCAF_ACF(data, Category = 'Confirmed Cases'):
    plot_acf(data['target'], title= Category + ' Autocorrelation Original Series')
    plot_acf(data['target'].diff().dropna(), title= Category + ' Autocorrelation First Order')
    plot_acf(data['target'].diff().diff().dropna(), title= Category + ' Autocorrelation Second Order')
    plot_acf(data['target'].diff().diff().diff().dropna(), title= Category + ' Autocorrelation Third Order')
    plot_pacf(data['target'], title= Category + ' Partial Autocorrelation Original Series')
    plt.show()

############################################################# Show ARIMA ACF PCAF ###############################################################
# Show ARIMA Confirmed Cases ACF PCAF Results 
showDifferencingPlots_D(res_Confirmed, 'Confirmed Cases')
showDifferencingPlots_AR(res_Confirmed, 'Confirmed Cases')
showPCAF_ACF(res_Confirmed, 'Confirmed Cases')

# Show ARIMA Death Cases Results 
showDifferencingPlots_D(res_Death,'Death Cases')
showDifferencingPlots_AR(res_Death,'Death Cases')
showPCAF_ACF(res_Death,'Death Cases')


