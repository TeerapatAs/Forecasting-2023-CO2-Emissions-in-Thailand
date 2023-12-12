import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
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
th_co2_per_pop = df[(df['country'] == 'Thailand') & ((df['year'].between(1970, 2022)))]['co2_per_capita'].tolist()
years = df[(df['country'] == 'Thailand') & ((df['year'].between(1970, 2022)))]['year'].tolist()
#print(th_co2)    #test_variable
#print(th_co2_per_pop)     #test_variable
print(years)
list_of_co2_years = list(zip(th_co2_per_pop, years))
print(list_of_co2_years)
mymodel = np.poly1d(np.polyfit(years, th_co2_per_pop, 3))
myline = np.linspace(1970, 2020, 100)
###########################
plt.scatter(years, th_co2_per_pop)
plt.xlabel("years")
plt.ylabel("th_co2_per_pop(1ton/person/year)")
plt.plot(myline, mymodel(myline), 'red')
#check r_2 score below
#print(r2_score(th_co2_per_pop, mymodel(years)))
#plt.show()
###########################
#train and test
train_samples = random.sample(list_of_co2_years,int(0.8*len(years)))
#print(train_samples)
test_samples = []
for h in list_of_co2_years:
    if h not in train_samples:
        test_samples.append(h)
#train&test progression
#extract co2_per_person in list
def Extract_0(lst):
    return [[item[0]] for item in lst]
def Extract_0_only_element(lst):
    return [item[0] for item in lst]
def Extract_1(lst):
    return [[item[1]] for item in lst]
def Extract_1_only_element(lst):
    return [item[1] for item in lst]
train_x = np.asanyarray(Extract_1(train_samples))  #years
train_y = np.asanyarray(Extract_0(train_samples)) #Co_2 Emission
test_x = np.asanyarray(Extract_1(test_samples))
test_y = np.asanyarray(Extract_0(test_samples))

poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)
regr = linear_model.LinearRegression()
train_y_ = regr.fit(train_x_poly, train_y)
# The coefficients of poly degree 2
#print('Coefficients: ', regr.coef_)
#print('Intercept: ',regr.intercept_)
plt.scatter(Extract_1_only_element(train_samples), Extract_0_only_element(train_samples),  color='blue')
XX = np.arange(1970, 2020, 1)
yy = regr.intercept_[0]+ regr.coef_[0][1]*XX+ regr.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, 'red' )
plt.xlabel("years")
plt.ylabel("Emission")
plt.show()

from sklearn.metrics import r2_score

test_x_poly = poly.fit_transform(test_x)
test_y_ = regr.predict(test_x_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )

##########################

poly3 = PolynomialFeatures(degree=3)
train_x_poly3 = poly3.fit_transform(train_x)

regr3 = linear_model.LinearRegression()
regr3.fit(train_x_poly3, train_y)
print ('Coefficients: ', regr3.coef_)
print ('Intercept: ',regr3.intercept_)
plt.scatter(Extract_1_only_element(train_samples), Extract_0_only_element(train_samples) ,  color='blue')
XX = np.arange(1970, 2020, 1)
yy = regr3.intercept_[0]+ regr3.coef_[0][1]*XX+ regr3.coef_[0][2]*np.power(XX, 2)+ regr3.coef_[0][3]*np.power(XX, 3)
#plt.plot(XX, yy, 'red' )
#plt.xlabel("years")
#plt.ylabel("Emission")
#plt.show()

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )

#year_predicted

