# we define all the routes and functions to perform for each action in this file. 
# This file is the root of our Flask application which we will run in the command line prompt.

from flask import Flask, render_template, redirect, url_for, request
import sqlite3
import pandas as pd
import pickle
import numpy
import logging
import pandas as pd
import matplotlib.pyplot as plt # for plotting graphs
import datetime as dt
import numpy as np
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import seaborn as sns # for plotting graphs


# flask application object
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/main', methods=['POST'])
def login():
    user = request.form['username']
    password = request.form['password']
    if (user == 'bank' and password=='bank123'):
        return render_template('main.html')
    else:
        return render_template('login.html')

@app.route('/analysis', methods=['POST'])
def analysis():
    #Connecting to sqlite
    conn = sqlite3.connect('Saving_pred_data.sqlite')

    #Creating a cursor object using the cursor() method
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM table2")
    data = cursor.fetchall()
    print("all done")

    if(request.form['submitbtn'] == 'predict'):
        return render_template('predict.html', data=data)
    elif(request.form['submitbtn'] == 'Logout'):
        return render_template('login.html')
    else:
        pass


@app.route('/visualize', methods=['POST'])
def visualize():

    data = request.form['myfile']
    dbname = 'Saving_pred_data'
    conn = sqlite3.connect(dbname + '.sqlite')
    df = pd.read_csv(request.form['myfile'])
    df.to_sql(name='Table1', con=conn, if_exists='replace')
    c = 0
    for row in conn.execute(
        'SELECT CreditScore,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary FROM table1;'):
        # print(row)
        x = list(row)
        print(x)
        with open('models/model_file1.pickle', 'rb') as modelFile:
            model = pickle.load(modelFile)
        prediction = model.predict([x])
        print("pred: ", prediction)
        b = int(prediction)
        print("b: ", b)
        print("c: ", c)

        conn.execute("update `table1` set `Exited` = (?) where `index` = (?)", (b, c))
        c = c + 1
     
    print("inserted")
    

    conn.execute("update `table1` set `Exited` = 'Yes' where `Exited` = 1.0")
    conn.execute("update `table1` set `Exited` = 'No' where `Exited` = 0.0")
    conn.commit();

        #Connecting to sqlite
    conn = sqlite3.connect('Saving_pred_data.sqlite')

    #Creating a cursor object using the cursor() method
    cursor = conn.cursor()

    #Droping EMPLOYEE table if already exists.
    cursor.execute("DROP TABLE IF EXISTS table2")

    #Creating table as per requirement
    sql = '''CREATE TABLE table2 as select * from table1 where Exited="Yes" '''
    cursor.execute(sql)
    print("Table created successfully.")
    conn.commit();
    ls = []
    for row in conn.execute('SELECT `index` FROM table2;'):
        row = list(row)
        ls.append(row)
        # m = list(row)
    print("ls ", ls)

    i = 0
    
    for row in conn.execute('SELECT Tenure FROM table2;'):
        # print(row)
        c = ls[i]
        x = list(row)
        print(x)
        with open('models/model_file2.pickle', 'rb') as modelFile:
            model = pickle.load(modelFile)
        prediction = model.predict([x])
        print("pred: ", prediction)
        b = int(prediction)
        print("b: ", b)
        print("c: ", c)

        conn.execute("update `table2` set `Reason for exiting company` = (?) where `index` = (?)", (b, c[0]))
        i = i + 1
     
    print("inserted")

    conn.execute("update `table2` set `Reason for exiting company` = 'High Service Charges/Rate of Interest' where `Reason for exiting company` = 0.0")
    conn.execute("update `table2` set `Reason for exiting company` = 'Long Response Times' where `Reason for exiting company` = 1.0")
    conn.execute("update `table2` set `Reason for exiting company` = 'Inexperienced Staff / Bad customer service' where `Reason for exiting company` = 2.0")
    conn.execute("update `table2` set `Reason for exiting company` = 'Excess Documents Required' where `Reason for exiting company` = 3.0")
    
    conn.execute("update `table2` set `IsActiveMember` = 'Yes' where `IsActiveMember` = 1.0")
    conn.execute("update `table2` set `IsActiveMember` = 'No' where `IsActiveMember` = 0.0")
    
    conn.commit();
    
    cursor.execute("SELECT * FROM table2")
    data = cursor.fetchall()

    df = pd.read_sql('SELECT * from table2', conn)
    # write DataFrame to CSV file
    df.to_csv('templates/churning_customers.csv', index = False)

    df = pd.read_sql('SELECT * from table1', conn)
    # write DataFrame to CSV file
    df.to_csv('templates/all_customers.csv', index = False)


    df = pd.read_csv('templates/churning_customers.csv')
    sns.catplot(y="Geography", 
    hue= "Reason for exiting company", kind="count",data=df) #hue is used to encode the points with different colors
    plt.savefig('static/geo-reason.png')

    index = df.index
    number_of_rows = len(index)

    df2 = pd.read_csv('templates/all_customers.csv')

    index2 = df2.index
    numberofrows2 = len(index2)
    number_of_rows2 = numberofrows2-number_of_rows
    print("numverofrows", number_of_rows)
    print("numverofrows2", number_of_rows2)

    labels = 'Retained', 'Exited'
    sizes = [number_of_rows2, number_of_rows]
    explode = (0, 0.1) # only "explode" the 2nd slice
    colors = ['#ff9999','#66b3ff']
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.pie(sizes,explode=explode, labels=labels, autopct='%1.1f%%', colors=colors) #autopct display the percent value
    plt.title("Percentage of Churned and Retained Customers")
    # plt.legend() 
    plt.savefig('static/pie.png')

    return render_template('visualize.html')
   


if __name__ == "__main__":
    app.run(debug = True) 

    # app.debug = True
 


