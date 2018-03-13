import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
#matplotlib inline
#存在日期转换的问题----------------------------------
def parser(x):
	return datetime.strptime(x, '%m/%d/%Y')

data = pd.read_csv('sp500_data.csv',header=0, parse_dates=[0], squeeze=True, date_parser=parser)
#print(data['trade_date'])
# data.index = data['time'].values
#
# plt.figure(figsize=(10,5))
# data.index = data['time'].values
plt.ylabel('S&P500_index')
plt.xlabel('Trade_date')
plt.title('S&P 500 index series')
plt.plot(data['value'])
plt.grid()
plt.annotate(u"This is a zhushi", xy = (0, 1), xytext = (1, 50),\
             arrowprops = dict(facecolor = "r", headlength = 10, headwidth = 30, width = 20))
plt.vlines(2983, 600, 2250, colors="r", linestyles="dashed")
plt.show()