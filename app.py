from flask import Flask, render_template, request
import statistics
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')




@app.route("/about")
def about():
    return render_template('about.html')





@app.route('/', methods = ['POST'])
def getvalue():

# Importing the dataset
    dataset = pd.read_csv(r'C:\Users\Sayan\OneDrive\Desktop\FPM\data.csv')

# Historical Flood Data : http://wbdmd.gov.in/pages/flood2.aspx
    years = [1986,1988,1991,1995,1998,1999,2000,2002,2004,2005,2006,2007,2013]

# Average 7-day rainfall in the moonsoon quarter and addding it to the dataset
    weekly = []
    for i in range(0,len(dataset)):
        weekly.append((dataset['Jun-Sep'][i])*(7/122))
    
    dataset['Weekly'] = weekly

# Storing the monsoon season and annual rainfall data for the years when flood has occured previously
    flood_monsoon = []
    flood_annual = []
    for x in years:
        flood_monsoon.append(dataset.loc[dataset['YEAR'] == x, 'Jun-Sep'].iloc[0])
        flood_annual.append(dataset.loc[dataset['YEAR'] == x, 'ANNUAL'].iloc[0])


    critical_rain = statistics.median(flood_monsoon)
    critical_annual_rain = statistics.median(flood_annual)

# Adding the Flood Column to the dataframe on the basis of historical data
    annual_rain = list(dataset['ANNUAL'])
    monsoon_rain = list(dataset['Jun-Sep'])

    flood = np.zeros((len(dataset),1)).astype(int)

# Updating flood occurence data
    for i in range(0,len(dataset)):
        if annual_rain[i]>critical_annual_rain or monsoon_rain[i]>critical_rain:
            flood[i] = 1

    dataset['Flood'] = flood


# Updating the datasets for the years where there is confirmation of flood occurence
    for i in range(0, len(years)):
        dataset.loc[dataset['YEAR'] == years[i], 'Flood'] = 1
    


# Separating the dependent and independent variables
# Need Reconsideration especially in choosing the features in the matrix
    X = dataset.iloc[:,[16,19]].values
    y = dataset.iloc[:,20].values


# Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting the Regression Model to the dataset
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 500, random_state = 0)
    regressor.fit(X, y)


# Taking input from the form
    d1 = request.form['d1']
    d2 = request.form['d2']
    d3 = request.form['d3']
    d4 = request.form['d4']
    d5 = request.form['d5']
    d6 = request.form['d6']
    d7 = request.form['d7']


    pre_monsoon_average = dataset['Mar-May'].mean()
    
    monsoon_rainfall = d1+d2+d3+d4+d5+d6+d7

# Predicting the result
    y_pred = regressor.predict([[pre_monsoon_average,monsoon_rainfall]])

# Flood Possibility Perecntage
    posb = float(y_pred)*100


#Returning and rendering pass.html file
    return render_template('pass.html', res = posb)

if __name__=="__main__":
    app.run(debug=True)





# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)


# # Predicting the Test set results
# y_pred = regressor.predict(X_test)





# Inputs for the Model
# premonsoon_rainfall = float(input('Enter Pre-Monsoon Rainfall: '))
# monsoon_rainfall = float(input('Enter Rainfall for Last 7 Days: '))
'''
# form = cgi.FieldStorage()
# day1 = float(form.getvalue('d1'))
# day2 = float(form.getvalue('d2'))
# day3 = float(form.getvalue('d3'))
# day4 = float(form.getvalue('d4'))
# day5 = float(form.getvalue('d5'))
# day6 = float(form.getvalue('d6'))
# day7 = float(form.getvalue('d7'))


# print('<h1 style="font-family: Calibri;">Day 1: %f</h1>' %(day1))
# print('<h1>Day 2: %f</h1>' %(day2))
# print('<h1>Day 3: %f</h1>' %(day3))
# print('<h1>Day 4: %f</h1>' %(day4))
# print('<h1>Day 5: %f</h1>' %(day5))
# print('<h1>Day 6: %f</h1>' %(day6))
# print('<h1>Day 7: %f</h1>' %(day7))


'''


'''
print('<pre><h1 style="font-family: Calibri;">Monsoon Rainfall in the last 7 days:     %.2f cm</h1></pre>' %(monsoon_rainfall))
print('<pre><h2 style="font-family: Calibri;">Probability of Flood:     '+'{:.2f}%</h2></pre>'.format(posb))
# input('Enter any key to exit')

if (posb >= 70):
    print('<h2 style="font-family: Calibri; color: red;">There is a high chance of imminent flood</h2>')
elif (posb>=40 and posb<70):
    print('<h2 style="font-family: Calibri; color: orange;">There is a moderate chance of imminent flood</h2>')
else:
    print('<h2 style="font-family: Calibri; color: green;">There is a low chance of imminent flood</h2>')
'''    


