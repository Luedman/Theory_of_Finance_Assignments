
# coding: utf-8

# # Theory of Finance Assignment 1 Due: 18.Okt
# Lukas Schreiner, 10 614 782 <br>
# Max Tragl, 13 613 419 <br>
# Asse, 14 604 201

# In[1]:


#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize



# ### Exercise 1.1 & 1.2

# In[2]:


#Load Data
SP500 = pd.read_csv("SP500daily.csv",
                sep         = ",",
                dtype       = {"Date"       : str,
                               "Open"       : np.float32,
                               "High"       : np.float32,
                               "Low"        : np.float32,
                               "Close"      : np.float32,
                               "Adj Close"  : np.float32,
                               "Volume"     : np.int32,},
                decimal     = ".",
                engine      = "python")


# In[3]:


#Adjust Date format
SP500['Date']  = pd.to_datetime(SP500['Date'])
SP500['Week']  = SP500['Date'].dt.week
SP500['Month'] = SP500['Date'].dt.month
SP500['Year']  = SP500['Date'].dt.year

#Calculate Returns and Log returns
SP500['Return'] = SP500['Adj Close'].pct_change()
SP500['Log Return'] = np.log(SP500['Adj Close']) - np.log(SP500['Adj Close'].shift(1))

#Rearrage Columns
cols = ['Date', 'Week', 'Month', 'Year',
        'Open', 'High', 'Low', 'Close',
        'Adj Close', 'Volume',  'Return',
        'Log Return']

SP500 = SP500[cols].set_index('Date')

SP500.head()


# In[4]:


def calculate_return_over_period(time_window, time_horizon = ['1950-01-03','2017-12-27']):

    value_endofperiod = []
    log_returns_period = []
    period = 0

    for index, row in SP500[time_horizon[0]:time_horizon[1]].iterrows():
        if row[time_window] != period:
            value_endofperiod.append(row['Close'])
            period = row[time_window]

    #The closing value of a period is the opening value of the next period
    for i in range(1,len(value_endofperiod)):
        log_returns_period.append(np.log(value_endofperiod[i] / value_endofperiod[i-1]))

    return log_returns_period


# In[5]:


log_mean_daily = np.mean(SP500['Log Return'])

log_returns_weekly  = calculate_return_over_period('Week')

log_returns_monthly = calculate_return_over_period('Month')

log_returns_yearly  = calculate_return_over_period('Year')


# In[6]:


output = {'Daily':   [(np.mean(SP500['Log Return']) *100),(np.std(SP500['Log Return']) *100), (np.var(SP500['Log Return']) *100)],
          'Weekly':  [(np.mean(log_returns_weekly)  *100),(np.std(log_returns_weekly)  *100), (np.var(log_returns_weekly)  *100)],
          'Monthly': [(np.mean(log_returns_monthly) *100),(np.std(log_returns_monthly) *100), (np.var(log_returns_monthly) *100)],
          'Yearly':  [(np.mean(log_returns_yearly)  *100),(np.std(log_returns_yearly)  *100), (np.var(log_returns_yearly)  *100)]}

ouput_table = pd.DataFrame(data = output, index = ['Mean', 'Std','Variance' ])
ouput_table.columns.name = 'in %'
ouput_table


# ### Exercise 1.3

# In[7]:


#Initial Investment = 1
init_inv = 1

value_inv_today = init_inv * (SP500['Return'] + 1).cumprod().iloc[-1]
print("Value of 1 USD Investment today: %.2f USD" % round(value_inv_today,2))


# ### Exercise 1.4

# In[8]:


#Import Data
TBill = pd.read_csv("DTB3.csv",
                sep         = ",",
                dtype       = {"DATE" : str,
                               "DTB3" : str,
                              },
                decimal     = ".",
                engine      = "python")

# Correct for annulization
TBill['Log TB3'] = np.log(pd.to_numeric(TBill['DTB3'], errors = 'coerce')/100/365 + 1)
TBill['DATE'] = pd.to_datetime(TBill['DATE'])

#Handle NAN values through interpolation
TBill['Log TB3'] = TBill['Log TB3'].interpolate()
print("Number of NAN Values in TB3: ", TBill['Log TB3'].isna().sum())

#Merge Datasets
SP500 = SP500.join(TBill.set_index("DATE"))
SP500[["Close", "Log Return" ,"Log TB3"]].tail(5)

#writer = pd.ExcelWriter('output.xlsx')
#df.to_excel(writer,'Sheet1')
#writer.save()


# In[9]:


risk_free = np.mean(SP500["Log TB3"]['2000-01-03':'2017-12-27'])

cum_log_returns     = (SP500['Log Return']['2000-01-03':'2017-12-27'] + 1).cumprod().iloc[-1]
cum_log_rf_returns  = (SP500['Log TB3']['2000-01-03':'2017-12-27'] + 1).cumprod().iloc[-1]

cum_excess_return = cum_log_returns - cum_log_rf_returns

print("The excess return over the period is %.4f percent" % (cum_excess_return*100))


# ### Exercise 1.5

# In[10]:


def sharpe_ratio(return_data, risk_free):

    excess_return  = [r - risk_free for r in return_data]

    avg_excess_ret = np.sum(excess_return) / len(excess_return)

    sharpe_ratio   = avg_excess_ret / np.var(excess_return)

    return sharpe_ratio

log_ret_weekly2    = calculate_return_over_period("Week",['2000-01-03','2017-12-27'])

log_ret_monthly2   = calculate_return_over_period("Month",['2000-01-03','2017-12-27'])

log_ret_yearly2    = calculate_return_over_period("Year",['2000-01-03','2017-12-27'])


# In[11]:


shr_daily     = sharpe_ratio(SP500["Log Return"]['2000-01-03':'2017-12-27'], risk_free)

shr_weekly    = sharpe_ratio(log_ret_weekly2, risk_free)

shr_monthly   = sharpe_ratio(log_ret_monthly2, risk_free)

shr_yearly    = sharpe_ratio(log_ret_yearly2, risk_free)

sharpe = pd.DataFrame({'Daily'   : [shr_daily],
                       'Weekly'  : [shr_weekly],
                       'Monthly' : [shr_monthly],
                       'Yearly'  : [shr_yearly]},
                       index = ['Sharpe Ratio'])

sharpe.columns.name = '.'
sharpe


# ### Exercise 2.1

# In[12]:


#Import Data
stock_data = pd.read_excel('PS1-2_Studynet.xlsx')

#Clean Header
stock_data.columns = stock_data.columns.str.replace("Equity", "")
stock_data.columns = stock_data.columns.str.replace("SE", "")
stock_data.columns = stock_data.columns.str.replace("SW", "")
stock_data.columns = stock_data.columns.str.replace(" ", "")


# In[13]:


stock_cols = stock_data.columns
stock_data = stock_data.set_index('Date')
stock_data.head()


# In[14]:


figure_1 = plt.figure(figsize=(20,6))
chart_1  = figure_1.add_subplot(121)
chart_2  = figure_1.add_subplot(122)

for i in range(1,len(stock_data.columns)):
    if stock_cols[i] != 'LISN':
        chart_1.plot(stock_data.index, stock_data[stock_cols[i]], label = stock_cols[i])

for i in range(1,len(stock_data.columns)):
    if stock_cols[i] != 'LISN':
        chart_2.plot(stock_data['LISN'], stock_data[stock_cols[i]], label = stock_cols[i])

chart_1.set_title('Prices / Time')
chart_2.set_title('Prices / LISN')


# In[15]:


# log_returns calclates the log return of a given dataset and returns a dataframe with only its log returns
def log_returns(dataset):

    cols = dataset.columns

    for i in range(0,len(cols)):

        #Calculate Log returns
        dataset['LG_' + cols[i]] = np.log(dataset.loc[:,(cols[i])]) - np.log(dataset.loc[:,(cols[i])].shift(1))

        #Delete Stocks Prices
        dataset = dataset.drop(cols[i],1)

        #Delete first row (contains 'NaN')
        lg_dataset = dataset.iloc[1:]

    return lg_dataset


# In[16]:


lg_stock_data = log_returns(stock_data)
lg_stock_data['Risk_Free'] = risk_free
lg_stock_data.head()


# In[17]:


figure_2 = plt.figure(figsize=(20,6))
chart_1  = figure_2.add_subplot(121)
chart_2  = figure_2.add_subplot(122)

for i in range(1,len(lg_stock_data.columns)):
    if stock_cols[i] != 'LG_LISN':
        chart_1.plot(lg_stock_data.index, lg_stock_data[lg_stock_data.columns[i]], label = lg_stock_data.columns[i])

for i in range(1,len(lg_stock_data.columns)):
    if stock_cols[i] != 'LG_LISN':
        chart_2.scatter(lg_stock_data['LG_LISN'], lg_stock_data[lg_stock_data.columns[i]], label = lg_stock_data.columns[i], s = 10)

chart_1.set_title('Returns / Time')
chart_2.set_title('Returns / LISN')


# ### Exercise 2.2

# In[18]:


# The risk free rate
risk_free = np.log(0.27/ 360 + 1)
print(risk_free)


# In[19]:


#Selected Stocks
selected_stocks_cols = ['CSGN', 'UBSG', 'ABBN', 'NESN', 'NOVN', 'CFR']

selected_stocks = stock_data.loc[:,(selected_stocks_cols)]

lg_selected_stocks = log_returns(selected_stocks)

lg_selected_stocks['Risk_Free'] = risk_free
lg_selected_stocks.head()


# In[20]:


np.mean(lg_selected_stocks)


# In[39]:


#Expected Log Returns for selected stocks with and without rf
sel_expretrf  = np.mean(lg_selected_stocks)
sel_expret    = np.mean(lg_selected_stocks.iloc[:,0:6])
#print(expected_returns)

#Covariance matrix rounded to 6 decimals for selected stocks with and without rf
sel_covmatrf = np.round(np.cov(lg_selected_stocks.values.T[:]),6)
sel_covmat   = np.round(np.cov(lg_selected_stocks.iloc[:,0:6].values.T[:]),6)
#print(sel_covmatrf)


# In[22]:


def portfolio_ret(w, expected_returns):
    return (np.sum(expected_returns * w)*100)

def portfolio_vol(w, cov_mat):
    return ((np.dot(np.dot(w.T,cov_mat), w)*100)**2)

def neg_sharpe_ratio(w, expected_returns, cov_mat, risk_free):
    pstd = np.sqrt(portfolio_vol(w, cov_mat))
    preturn = portfolio_ret(w, expected_returns)

    return -(preturn - risk_free) / pstd


# In[23]:


#Calculate the minimum variance portfolio without riskless asset
def minvar(ret, expected_returns, cov_mat):

    num_assets = len(expected_returns)

    args = (cov_mat)

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                   {'type': 'eq', 'fun': lambda w: portfolio_ret(w , expected_returns) - ret})

    result = minimize(portfolio_vol, num_assets*[1./num_assets,], args = args,
                        method = 'SLSQP', constraints = constraints)

    mu  = portfolio_ret(result.x, expected_returns)
    var = portfolio_vol(result.x, cov_mat)

    return mu, var, result.x


# In[24]:


#Calculate random portfolios
def random_portfolios(number, exp_ret, cov_mat):
    #Random Portfolios (rp)
    rp_ret = []
    rp_std = []

    for i in range(0,number):

        #Choose Random weights and calculate repective mean return (rp_ret) and standard deviations (rp_std)
        rn = np.random.normal(0,1,len(exp_ret))
        rp_w = rn / np.sum(rn)

        rp_ret.append(portfolio_ret(rp_w, exp_ret))
        rp_std.append(np.sqrt(portfolio_vol(rp_w, cov_mat)))

    return rp_ret, rp_std


# In[25]:


#Calculate the mean variance frontier without riskless asset
def mv_frontier(exp_ret, cov_mat):

    minvar_ret = []
    minvar_std = []
    minvar_wg  = []
    mu_star = 0

    for i in range(0,60):

        mu, var, w = minvar(mu_star, exp_ret, cov_mat)

        minvar_ret.append(round(mu,6))
        minvar_std.append(np.sqrt(var))
        minvar_wg.append(w)

        mu_star += 0.01

    return minvar_ret, minvar_std, minvar_wg


# In[26]:


#Calculate the tanget portfolio for a given set of portfolios
def tangency_portfolio(expected_returns, cov_mat, risk_free):

    num_assets = len(expected_returns)

    args = (expected_returns, cov_mat, risk_free)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    result = minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method = 'SLSQP', constraints=constraints)

    tang_mu  = portfolio_ret(result.x, expected_returns)
    tang_std = np.sqrt(portfolio_vol(result.x, cov_mat))

    return tang_mu, tang_std, result.x


# ### <font color='red'> Excersise 2.2 </font>

# In[27]:


#Calculate the MV Frontier for selected stocks
sel_minvar_ret, sel_minvar_std, minvar_wg = mv_frontier(sel_expret, sel_covmat)

#Calculate the Tangency Portfolio for selected stocks
sel_tang_mu, sel_tang_std, _ = tangency_portfolio(sel_expret, sel_covmat, risk_free)

#Calculate Radom Portfolios
sel_rp_ret, sel_rp_std = random_portfolios(1000, sel_expret, sel_covmat)

#Min Var Frontier w/ risk free asset
sel_minvar_stdrf   = np.linspace(0, sel_minvar_std[59], len(sel_minvar_std[:60]))
sel_minvar_retrf   = np.divide(np.dot(sel_minvar_stdrf,sel_tang_mu),sel_tang_std) + risk_free
sel_minvar_wgrf    = 1 - sel_minvar_retrf / sel_tang_mu


# In[28]:


portfolios = pd.DataFrame({'1%'    : [sel_minvar_retrf[0] ,sel_minvar_stdrf[0], sel_minvar_wgrf[0]],
                           '10%'   : [sel_minvar_retrf[1] ,sel_minvar_stdrf[1], sel_minvar_wgrf[1]],
                           '30%'   : [sel_minvar_retrf[30],sel_minvar_stdrf[30],sel_minvar_wgrf[30]],
                           '39%'   : [sel_minvar_retrf[40],sel_minvar_stdrf[40],sel_minvar_wgrf[40]]},
                            index = ['Exp Ret', 'Std', 'Weight rf'])

portfolios.columns.name = 'Expected Return'
portfolios


# ### Excersise 2.3

# In[29]:


#Plot the MV Frontier for selected stocks
chart_3 = plt.scatter(sel_rp_std, sel_rp_ret, marker = '.', color = 'grey', label = 'Random Portfolios')

chart_3 = plt.plot(sel_minvar_std, sel_minvar_ret, color = 'red', label = 'Effiecient Frontier w/o rf')
chart_4 = plt.plot(sel_minvar_stdrf, sel_minvar_retrf, color = 'blue', label = 'Efficient Frontier w/ risk free' , linestyle = ':')

chart_3 = plt.scatter([sel_tang_std], [sel_tang_mu], color = 'red',marker = 'x', label = 'Tangency Portfolio',  s=50)

chart_3 = plt.legend()

axes = plt.gca()
axes.set_xlim([0,0.15])
axes.set_ylim([-1,1])

chart_3 = plt.xlabel('Standard Deviation')
chart_3 = plt.ylabel('Expected Return')

chart_3 = plt.show()


# ### Exercise 2.4

# In[30]:


#Expected Log Returns for 48 Stocks
all_expret   = np.mean(lg_stock_data.iloc[:,0:48])
all_expretrf = np.mean(lg_stock_data)

#Covariance matrix for 48 stocks rounded to 6 decimals
all_covmat   = np.round(np.cov(lg_stock_data.iloc[:,0:48].values.T[:]),6)
all_covmatrf = np.round(np.cov(lg_stock_data.values.T[:]),6)


# In[31]:


#Calculate the MV Frontier for all stocks
all_minvar_ret, all_minvar_std, all_minvar_wg = mv_frontier(all_expret, all_covmat)

all_minvar_retrf, all_minvar_stdrf, all_minvar_wgrf = mv_frontier(all_expretrf, all_covmatrf)

#Calculate the Tangency Portfolio for selected stocks
all_tang_mu, all_tang_std, all_tang_wg = tangency_portfolio(all_expret, all_covmat, risk_free)

#Calculate Radom Portfolios
#all_rp_ret, all_rp_std = random_portfolios(3000, all_expret, all_covmat)

#Min Var Frontier w/ risk free asset
all_minvar_stdrf   = np.linspace(0, all_minvar_std[59], len(all_minvar_std[:60]))
all_minvar_retrf   = np.divide(np.dot(all_minvar_stdrf,all_tang_mu),all_tang_std)
all_minvar_wgrf    = 1 - sel_minvar_retrf / sel_tang_mu


# In[32]:


#Plot the MV Frontier for selected stocks
#chart_4 = plt.scatter(all_rp_std, all_rp_ret, marker = '.', color = 'grey',  label = 'Random Portfolios w/ all Assets' )

chart_4 = plt.scatter([all_tang_std], [all_tang_mu], color = 'red', marker = 'x', label = 'Tangency Portfolio')

chart_4 = plt.plot(all_minvar_stdrf, all_minvar_retrf, color = 'blue', label = 'Efficient Frontier w/ risk free' , linestyle = ':')
chart_4 = plt.plot(all_minvar_std, all_minvar_ret, color = 'red', label = 'Efficient Frontier w/o risk free')

chart_4 = plt.plot(sel_minvar_std, sel_minvar_ret, color = 'grey', label = 'Efficient Frontier (selected Assets)')

chart_4 = plt.legend()

axes = plt.gca()
axes.set_xlim([0,0.05])
axes.set_ylim([-1,1])

chart_4 = plt.xlabel('Standard Deviation')
chart_4 = plt.ylabel('Expected Return')

chart_4 = plt.show()


# ### Exercise 2.5

# In[33]:


wg_CSGN = round(all_tang_wg[lg_stock_data.columns.get_loc('LG_CSGN')]*100,2)
wg_NOVN = round(all_tang_wg[lg_stock_data.columns.get_loc('LG_NOVN')]*100,2)

print("The weight of Credit Suisse in the Tangency Portfolio is", wg_CSGN, "Percent")
print("The weight of Novartis in the Tangency Portfolio is", wg_NOVN, "Percent")


# ### <font color='red'> Excersise 2.6 </font>

# ### Exercise 2.7

# In[34]:


minvar_port_std = all_minvar_std[all_minvar_std.index(np.min(all_minvar_std))]
minvar_port_ret = all_minvar_ret[all_minvar_std.index(np.min(all_minvar_std))]
minvar_wg = all_minvar_wg[all_minvar_std.index(np.min(all_minvar_std))]

'''
print(np.round(minvar_port_std,4))
print(minvar_port_ret)
print(minvar_wg)
'''


# In[35]:


chart_5 = plt.scatter([minvar_port_std], [minvar_port_ret], color = 'blue', marker = 'x', label = 'Min Var Portfolio')
chart_5 = plt.plot(all_minvar_std, all_minvar_ret, color = 'red', label = 'Efficient Frontier w/o risk free')
chart_5 = plt.legend()

axes = plt.gca()
axes.set_xlim([0,0.05])
axes.set_ylim([-1,1])

chart_5 = plt.show()


# In[36]:


wg_UBSG = round(minvar_wg[lg_stock_data.columns.get_loc('LG_UBSG')]*100,2)
wg_ABBN = round(minvar_wg[lg_stock_data.columns.get_loc('LG_ABBN')]*100,2)
wg_ROG = round(minvar_wg[lg_stock_data.columns.get_loc('LG_ROG')]*100,2)

print("The weight of UBS in the Minvar Portfolio is", wg_UBSG, "Percent")
print("The weight of ABB in the Tangency Portfolio is", wg_ABBN, "Percent")
print("The weight of Roche in the Tangency Portfolio is", wg_ROG, "Percent")
