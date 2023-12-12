import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
#------
df = pd.read_csv(r"D:\a.teerapat\work2\semester1\Prob stats\owid-co2-data.csv")
#Test_filter the rows
#cdf = df[['country','year','co2']]
#test_df and test if a column 'year' is int.
#print(df)
#bb_column_dtype = df['year'].dtype
#if bb_column_dtype == 'int64':
#    print("'year' column is of integer type.")
#else:
#    print("'year' column is not of integer type.")
th_co2 = df[(df['country'] == 'Thailand') & ((df['year'].between(1970, 2020)))]['co2'].tolist()
th_co2_per_pop = df[(df['country'] == 'Thailand') & ((df['year'].between(1970, 2020)))]['co2_per_capita'].tolist()
years = df[(df['country'] == 'Thailand') & ((df['year'].between(1970, 2020)))]['year'].tolist()
#print(th_co2)    #test_variable
#print(th_co2_per_pop)     #test_variable
#print(years)
list_of_co2_years = list(zip(th_co2_per_pop, years))
#print(list_of_co2_years)
mymodel = np.poly1d(np.polyfit(years, th_co2_per_pop, 3))
myline = np.linspace(1970, 2020, 100)
###########################

train_samples = list_of_co2_years[:len(list_of_co2_years)-1]  # Training data up to 2022

def Extract(lst, index):
    return [[item[index]] for item in lst]

train_x = np.asanyarray(Extract(train_samples, 1))  # Years
train_y = np.asanyarray(Extract(train_samples, 0))  # CO2 Emissions per capita

poly = PolynomialFeatures(degree=3)
train_x_poly = poly.fit_transform(train_x)

regr = LinearRegression()
regr.fit(train_x_poly, train_y)

# Predict CO2 emission for 2023
co2_2021 = regr.predict(poly.fit_transform([[2021]]))
predicted_co2_2023 = co2_2021[0][0]

# Print the predicted value for CO2 emissions in 2023
print("Predicted CO2 emission for 2023:", predicted_co2_2023,"tons/person/year")