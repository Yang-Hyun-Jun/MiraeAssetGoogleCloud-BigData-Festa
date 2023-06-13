from abc import ABCMeta, abstractmethod

#import libararies
import pandas as pd
import matplotlib
import pandas as pd
import json
import matplotlib.pyplot as plt

# stock api 
import FinanceDataReader as fdr

# crawling 
import requests
from bs4 import BeautifulSoup
import html5lib


import re
import os, sys


# set options for plot
pd.set_option('display.float_format', lambda x: '%.2f' % x)
#%matplotlib inline

plt.rcParams["figure.figsize"] = (14,8)
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2
plt.rcParams["axes.grid"] = True
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["axes.formatter.limits"] = -10000, 10000
