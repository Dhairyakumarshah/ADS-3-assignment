# -*- coding: utf-8 -*-
"""
Created on Thu May 07 16:47:13 2023

@author: Dhairya Shah
"""
#Start with importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import errors as err
import sklearn.metrics as skmet
from sklearn import cluster

#start taking data
def get_data_frames(filename,countries,indicator):
    '''
    functions returns as two dataframes one with countries as column 
    and other one year as also column.
    It is also traspose the dataframe rows into column and column into rows
    which have three arguments as below. 

    Parameters
    ----------
    filename : Text
        Name of the files for data.
    countries : List
        List of countries 
    indicator : Text
        Indicator Code 

    Returns
    -------
    df_countries : DATAFRAME
         which contains countries = rows 
                        years = columns
    df_years : DATAFRAME
         which contains countries = colomns 
                        years = rows
    '''
    # Reading data 
    df = pd.read_csv(filename, skiprows=(4), index_col=False)
    # Get dataframe information.
    df.info()
    # To clean data we need to remove unnamed column.
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # To filter data by countries
    df = df.loc[df['Country Name'].isin(countries)]
    # To filter data by indicator code.
    df = df.loc[df['Indicator Code'].eq(indicator)]
    
    # Using melt function to convert all the years column into rows as 1 column
    df2 = df.melt(id_vars=['Country Name','Country Code','Indicator Name'
                           ,'Indicator Code'], var_name='Years')
    # Deleting country code column.
    del df2['Country Code']
    # Using pivot table function to convert countries from rows to separate 
    # column for each country.   
    df2 = df2.pivot_table('value',['Years','Indicator Name','Indicator Code']
                          ,'Country Name').reset_index()
    
    df_countries = df
    df_years = df2
    
    # Cleaning data droping nan values.
    df_countries.dropna()
    df_years.dropna()
    
    return df_countries, df_years

def get_data_frames1(filename,indicator):
    '''
    This function returns two dataframes one with countries as column and other 
    one years as column.
    It tanspose the dataframe and converts rows into column and column into 
    rows of specific column and rows.
    It takes three arguments defined as below. 

    Parameters
    ----------
    filename : Text
        Name of the file to read data.
    countries : List
        List of countries to filter the data.
    indicator : Text
        Indicator Code to filter the data.

    Returns
    -------
    df_countries : DATAFRAME
        This dataframe contains countries in rows and years as column.
    df_years : DATAFRAME
        This dataframe contains years in rows and countries as column..

    '''
    # Read data using pandas in a dataframe.
    df = pd.read_csv(filename, skiprows=(4), index_col=False)
    # Get datafarme information.
    df.info()
    # To clean data we need to remove unnamed column.
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # To filter data by indicator codes.
    df = df.loc[df['Indicator Code'].isin(indicator)]
    
    # Using melt function to convert all the years column into rows as 1 column
    df2 = df.melt(id_vars=['Country Name','Country Code','Indicator Name'
                           ,'Indicator Code'], var_name='Years')
    # Deleting country code column.
    del df2['Indicator Name']
    # Using pivot table function to convert countries from rows to separate 
    # column for each country.   
    df2 = df2.pivot_table('value',['Years','Country Name','Country Code']
                          ,'Indicator Code').reset_index()
    
    df_countries = df
    df_indticators = df2
    
    # Cleaning data droping nan values.
    df_countries.dropna()
    df_indticators.dropna()
    
    return df_countries, df_indticators


def poly(x, a, b, c, d):
    '''
    Cubic polynominal for the fitting
    '''
    y = a*x**3 + b*x**2 + c*x + d
    return y

def exp_growth(t, scale, growth):
    ''' 
    Computes exponential function with scale and growth as free parameters
    '''
    f = scale * np.exp(growth * (t-1960))
    return f

def logistics(t, scale, growth, t0):
    ''' 
    Computes logistics function with scale, growth raat
    and time of the turning point as free parameters
    '''
    f = scale / (1.0 + np.exp(-growth * (t - t0)))
    return f

def norm(array):
    '''
    Returns array normalised to [0,1]. Array can be a numpy array
    or a column of a dataframe
    '''
    min_val = np.min(array)
    max_val = np.max(array)
    scaled = (array-min_val) / (max_val-min_val)
    return scaled

def norm_df(df, first=0, last=None):
    '''
    Returns all columns of the dataframe normalised to [0,1] with the
    exception of the first (containing the names)
    Calls function norm to do the normalisation of one column, but
    doing all in one function is also fine.
    First, last: columns from first to last (including) are normalised.
    Defaulted to all. None is the empty entry. The default corresponds
    '''
    # iterate over all numerical columns
    for col in df.columns[first:last]: # excluding the first column
        df[col] = norm(df[col])
    return df


def map_corr(df, size=10):
    """Function creates heatmap of correlation matrix for each pair of columns␣
    ↪→in the dataframe.
    Input:
    df: pandas DataFrame
    size: vertical and horizontal size of the plot (in inch)
    """
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr, cmap='RdBu')
    # setting ticks to column names
    plt.xticks(range(len(corr.columns)), ['Population Growth', 'Total Population', 'Urban Growth', 'Total urban'], rotation=90)
    plt.yticks(range(len(corr.columns)), ['Population Growth', 'Total Population', 'Urban Growth', 'Total urban'])



# Data fitting for India Population with prediction


countries = ['Germany','Australia','United States','India','United Kingdom']
# calling functions to get dataframes and use for plotting graphs.
df_c, df_y = get_data_frames("D:/40marks/small.csv",countries,
                             'SP.POP.TOTL')

df_y['Years'] = df_y['Years'].astype(int)

popt, covar = curve_fit(exp_growth, df_y['Years'], df_y['India'])
print("Fit parameter", popt)
# use *popt to pass on the fit parameters
df_y['ind_exp'] = exp_growth(df_y['Years'], *popt)
plt.figure()
plt.plot(df_y['Years'], df_y["India"], label='data')
plt.plot(df_y['Years'], df_y['ind_exp'], label='baseline',color='black')
plt.legend()
plt.title("Exponential Growth Model with Baseline",color = "red",fontsize = 15)
plt.xlabel("Year")
plt.ylabel("Indian Population")
plt.show()

# find a feasible start value the pedestrian way
# the scale factor is way very small. The exponential factor is very large.
# going down or rising exponential factor upto rough agreement 
popt = [7e8, 0.01]
df_y['ind_exp'] = exp_growth(df_y['Years'], *popt)
plt.figure()
plt.plot(df_y['Years'], df_y['India'], label='data')
plt.plot(df_y['Years'], df_y['ind_exp'], label='baseline',color='black')

#lables and legends for graph
plt.legend()
plt.xlabel("Year")
plt.ylabel("Indian Population")
plt.title("Improved start value", color = "red",fontsize = 15)
plt.show()

# fit exponential growth
popt, covar = curve_fit(exp_growth, df_y['Years'],df_y['India'], p0=[7e8, 0.02])
# much better
print("Fit parameter", popt)
df_y['ind_exp'] = exp_growth(df_y['Years'], *popt)
plt.figure()
plt.plot(df_y['Years'], df_y['India'], label='data')
plt.plot(df_y['Years'], df_y['ind_exp'], label='baseline',color='black')

#lables and legends for graph
plt.legend()
plt.xlabel("Year")
plt.ylabel("Indian Population")
plt.title("Exponential growth", color = "red",fontsize = 15)
plt.show()



# taking growth value as compare before
popt = [1135185000, 0.02, 1990]
df_y['ind_log'] = logistics(df_y['Years'], *popt)
plt.figure()
plt.plot(df_y['Years'], df_y['India'], label='data')
plt.plot(df_y['Years'], df_y['ind_log'], label='baseline',color='black')
plt.legend()

#lables and legends for graph
plt.xlabel("Year")
plt.ylabel("Indian Population")
plt.title("Improved start value",fontsize = 15)
plt.show()

popt, covar = curve_fit(logistics,  df_y['Years'],df_y['India'],
p0=(6e9, 0.05, 1990.0))
print("Fit parameter", popt)
df_y['ind_log'] = logistics(df_y['Years'], *popt)
plt.figure()
plt.plot(df_y['Years'], df_y['India'], label='data')
plt.plot(df_y['Years'], df_y['ind_log'], label='baseline',color='black')

#lables and legends for graph
plt.legend()
plt.xlabel("Year")
plt.ylabel("Indian Population")
plt.title("Logistic Function",color = "red",fontsize = 15)


# extract the sigmas from the diagonal of the covariance matrix
sigma = np.sqrt(np.diag(covar))
print(sigma)

low, up = err.err_ranges(df_y['Years'], logistics, popt, sigma)
plt.figure()
plt.plot(df_y['Years'], df_y['India'], label='data')
plt.plot(df_y['Years'], df_y['ind_log'], label='baseline',color='black')
plt.fill_between(df_y['Years'], low, up, alpha=0.7, label='error range')
plt.legend()
#addisional text below graph
text = "Logistic values of Indian poppulation with Error range"
plt.text(0.5, -0.4, text, ha='center', va='center',color = "blue" ,transform=plt.gca().transAxes)

#lables and legends for graph
plt.title("logistics functions", color = "red",fontsize = 15)
plt.xlabel("Year")
plt.ylabel("Indian Population")
plt.show()

print("Forecasted population")
low, up = err.err_ranges(2030, logistics, popt, sigma)
print("2030 between ", low, "and", up)
low, up = err.err_ranges(2040, logistics, popt, sigma)
print("2040 between ", low, "and", up)
low, up = err.err_ranges(2050, logistics, popt, sigma)
print("2050 between ", low, "and", up)

print("Forecasted population with uncertainties")
low, up = err.err_ranges(2030, logistics, popt, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2030:", mean, "+/-", pm)
low, up = err.err_ranges(2040, logistics, popt, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2040:", mean, "+/-", pm)
low, up = err.err_ranges(2050, logistics, popt, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2050:", mean, "+/-", pm)


# Bar graph for Urban population growth in %

df_c, df_y = get_data_frames("D:/40marks/small.csv",countries
                             ,'SP.POP.GROW')
num= np.arange(4)
width= 0.1
#selectig years 
df_y = df_y.loc[df_y['Years'].isin(['2018','2019','2020','2021'])]
years = df_y['Years'].tolist() 

#data for bar graph  
plt.figure(dpi=144)
plt.bar(num,df_y['Germany'], width, label='Germany',color = "red")
plt.bar(num+0.2, df_y['Australia'], width, label='Australia',color = "green")
plt.bar(num-0.2, df_y['United States'], width, label='United States',color = "black")
plt.bar(num-0.4, df_y['India'], width, label='India',color = "maroon")
plt.xticks(num, years)

#lables and legends for graph
plt.xlabel('last 4 Years')
plt.ylabel('Annual growth in Urban pop')
plt.title('Population growth (annual %)', color = "red",fontsize = 15)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

# Bar graph for GDP per capita growth in %

df_c, df_y = get_data_frames("D:/40marks/big.csv"
                             ,countries,'NY.GDP.PCAP.KD.ZG')
num= np.arange(4)
width= 0.1
#selectig years
df_y = df_y.loc[df_y['Years'].isin(['2018','2019','2020','2021'])]
years = df_y['Years'].tolist() 

#data for bar graph  
plt.figure(dpi=144)
plt.bar(num,df_y['Germany'], width, label='Germany',color = "red")
plt.bar(num+0.2, df_y['Australia'], width, label='Australia',color = "green")
plt.bar(num-0.2, df_y['United States'], width, label='United States',color = "black")
plt.bar(num-0.4, df_y['India'], width, label='India',color = "maroon")
plt.xticks(num, years)

#lables and legends for graph
plt.xlabel('last 4 Years')
plt.ylabel('Annual growth in GDP 4 last year')
plt.title('GDP per capita growth (annual %)', color = "red",fontsize = 15)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


#Clustering Analysis with k-means Clustering method from lac.8


indicators = ['SP.POP.GROW','SP.POP.TOTL','SP.URB.GROW','SP.URB.TOTL']
df_y, df_i = get_data_frames1("D:/40marks/small.csv"
                             ,indicators)


df_i = df_i.loc[df_i['Years'].eq('2018')]
df_i = df_i.loc[~df_i['Country Code'].isin(['IND','PAK'])]

df_i.dropna()
    
# Heat Map Plot
map_corr(df_i)
plt.show()

# Scatter Matrix graph
pd.plotting.scatter_matrix(df_i, figsize=(7.0, 7.0))
plt.suptitle("Scatter Matrix Plot",color = "red", fontsize=17)
plt.tight_layout()
plt.show()


# extract columns to fitting
df_fit = df_i[["SP.POP.GROW", "SP.URB.GROW"]].copy()

# normalisation happens on the extracted columns and real measurements
df_fit = norm_df(df_fit)
print(df_fit.describe())
df_fit = df_fit.dropna()



for ic in range(2, 7):
    # set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(df_fit)
    # extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print (ic, skmet.silhouette_score(df_fit, labels))


# Plot for four clusters
kmeans = cluster.KMeans(n_clusters=3)
kmeans.fit(df_fit)

# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_

plt.figure(figsize=(6.0, 6.0))

# Individual colours can be assigned to symbols. The label l is used to the
# select the l-th number from the colour table.
scatter = plt.scatter(df_fit["SP.POP.GROW"], df_fit["SP.URB.GROW"], c=labels
            , cmap="viridis")
# colour map Accent selected to increase contrast between colours

#adding legend 
legend_elements = scatter.legend_elements()
lables = ["Scatter data {}".format(i) for i in range(len(legend_elements[0]))]
plt.legend(legend_elements[0], lables,title = "Data read")

# show cluster centres
for ic in range(3):
    xc, yc = cen[ic,:]
    plt.plot(xc, yc, "dk", markersize=8)
plt.grid(True) 
#lables and legends for graph
plt.xlabel("Total Population Growth")
plt.ylabel("Urban Population Growth")
plt.title("Population growth IN 2018", color = "red",fontsize = 15)
plt.show()


def clean(x):

    # count the number of missing values in each column of the DataFrame
    x.isnull().sum()
    # fill any missing values with 0 and update the DataFrame in place
    x.fillna(0, inplace=True)

    return

clean(df_fit)

df_fit_trans = df_fit.transpose()
print(df_fit_trans.head())








