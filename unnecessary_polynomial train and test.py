import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

th_co2 = [15.353, 19.221,21.805,24.42,24.211,24.372,28.716,32.065,34.947,36.586,39.956,37.817,37.636,42.117,45.556,48.23,49.223,56.306,66.203,77.784,87.916,96.748,105.964,120.911,133.299,155.517,173.729,178.594,158.145,164.987,168.8,180.708,190.81,201.993,217.171,222.99,226.331,229.41,226.873,242.274,255.405,253.612,270.268,287.598,282.648,285.688,283.435,285.96,276.177,271.625,257.766]
th_pop = [36884524, 37963280,39058592,40159584,41252320,42326308,43377268,44405904,45413084,46401752,47374464,48326268,49255892,50173920,51094872,52026904,52980104,53945872,54891520,55772168,56558196,57232472,57811024,58337772,58875272,59467272,60130192,60846588,61585100,62298572,62952640,63539192,64069092,64549868,64995304,65416188,65812540,66182064,66530976,66866836,67195032,67518384,67835968,68144512,68438752,68714512,68971312,69209824,69428456,69625576,69799976]
years = [] #list_of_year
th_co2_per_pop = [] #list_of_th_co2_per_pop
k = 1970
while k < 2021:
    years.append(k)
    k += 1
for i in range(len(th_co2)):
    th_co2_per_pop.append(th_co2[i]/th_pop[i]*10**6)
list_of_co2_years = []  #list of [co2_per_person,year] #not neccesary
for j in range(len(th_co2)):
    list_of_co2_years.append([th_co2[j]/th_pop[j]*10**6 , years[j]])
#train-test
msk = random.sample(range(len(years)),int(0.8*len(years)))
msk.sort()
train_year = []
train_co2_per_person = []
test_year = []
test_co2_per_person = []
test = []  #list of test [co2,year]
for k in msk:
    train_year.append(years[k])
    train_co2_per_person.append(th_co2_per_pop[k])
for l in list_of_co2_years:
    if l[0] not in train_co2_per_person:
        test_co2_per_person.append(l[0])
    if l[1] not in train_year:
        test_year.append(l[1])
2D_array_test = []
poly = PolynomialFeatures(degree=2)
train_co2_per_person_poly = poly.fit_transform(train_co2_per_person)
regr = linear_model.LinearRegression()
train_y_ = regr.fit(train_co2_per_person_poly, train_year)
# The coefficients
print('Coefficients: ', regr.coef_)
print('Intercept: ',regr.intercept_)

#print(train)
#print(test)