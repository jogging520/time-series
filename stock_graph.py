import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
#matplotlib inline
#存在日期转换的问题----------------------------------
def parser(x):
	return datetime.strptime(x, '%m/%d/%Y')

data = pd.read_csv('sp500_data.csv',header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
print(data.head())
# data.index = data['time'].values
plt.figure(figsize=(10,5))

# data.index = data['time'].values
plt.ylabel('S&P500_index')
plt.xlabel('Trade_date')
plt.figure(figsize=(10,5))
plt.plot(data)
plt.title('S&P 500 index series')
plt.show()


# series = pd.read_csv('sp500.csv')
# plt.ylabel('price')
# plt.title('time')
# series.plot()
# plt.show()