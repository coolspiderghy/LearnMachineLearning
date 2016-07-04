import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

concrete  = pd.read_excel(os.path.join('data','concrete.xls'))
concrete.columns = [
    'cement', 'slag', 'ash', 'water', 'splast',
    'coarse', 'fine', 'age', 'strength'
]

credit = os.path.join('data','credit.xls')
credit = pd.read_excel(credit, header=1)
credit.columns = [
    'id', 'limit', 'sex', 'edu', 'married', 'age', 'apr_delay', 'may_delay',
    'jun_delay', 'jul_delay', 'aug_delay', 'sep_delay', 'apr_bill', 'may_bill',
    'jun_bill', 'jul_bill', 'aug_bill', 'sep_bill', 'apr_pay', 'may_pay', 'jun_pay',
    'jul_pay', 'aug_pay', 'sep_pay', 'default'
]
occupancy = os.path.join('data','occupancy_data','datatraining.txt')
occupancy = pd.read_csv(occupancy, sep=',')
occupancy.columns = [
    'date', 'temp', 'humid', 'light', 'co2', 'hratio', 'occupied'
]

sns.set_style('whitegrid')

def box_viz(df):
    ax = sns.boxplot(df)
    #ax = sns.violinplot(df)
    plt.xticks(rotation=60)
    plt.show()
#box_viz(concrete)
def hist_viz(df,feature):
    ax = sns.distplot(df[feature])
    plt.xlabel(feature)
    plt.show()

#hist_viz(credit,'age') # We need to specify a feature vector

def splom_viz(df, labels=None):
    ax = sns.pairplot(df, hue=labels, diag_kind='kde', size=2)
    plt.show()

#splom_viz(concrete)

from pandas.tools.plotting import radviz

def rad_viz(df,labels):
    fig = radviz(df, labels, color=sns.color_palette())
    plt.show()

#rad_viz(occupancy.ix[:,1:],'occupied') # Specify which column contains the labels

from pandas.tools.plotting import parallel_coordinates

def pcoord_viz(df, labels):
    fig = parallel_coordinates(df, labels, color=sns.color_palette())
    plt.show()

#pcoord_viz(occupancy.ix[:,1:],'occupied') # Specify which column contains the labels

def joint_viz(feat1,feat2,df):
    ax = sns.jointplot(feat1, feat2, data=df, kind='reg', size=5)
    plt.xticks(rotation=60)
    plt.show()

#joint_viz('apr_bill','sep_bill',credit)