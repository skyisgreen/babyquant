import jqdatasdk as jq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# 设置起止时间及股票池
start_date = '2018-01-01'
end_date = '2018-1-16'
security_list = ['513100.XSHG', '518800.XSHG', '163407.XSHE', '159926.XSHE']

# 获取数据
stocks_price = jq.get_price(security_list, start_date=start_date, end_date=end_date, fields=['close'])['close']
stocks_price.head()

# 通过Monte Carlo模拟产生有效前沿组合

returns_daily = stocks_price.pct_change()
returns_annual = returns_daily.mean() * 250

cov_daily = returns_daily.cov()
cov_annual = cov_daily * 250

port_returns = []
port_volatility = []
sharpe_ratio = []
stock_weights = []

num_assets = len(security_list)
num_portfolios = 50000

np.random.seed(101)

for single_portfolio in range(num_portfolios):
   weights = np.random.random(num_assets)
   weights /= np.sum(weights)
   returns = np.dot(weights, returns_annual)
   volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
   sharpe = returns / volatility
   sharpe_ratio.append(sharpe)
   port_returns.append(returns)
   port_volatility.append(volatility)
   stock_weights.append(weights)

portfolio = {'Returns': port_returns,
            'Volatility': port_volatility,
            'Sharpe Ratio': sharpe_ratio}

for counter,symbol in enumerate(security_list):
   portfolio[symbol+' Weight'] = [Weight[counter] for Weight in stock_weights]

df = pd.DataFrame(portfolio)
column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock+' Weight' for stock in security_list]
df = df[column_order]
df.head()


plt.style.use('ggplot')
df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
               cmap='autumn', edgecolors='black', figsize=(15, 9), grid=True)
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier')
plt.show()