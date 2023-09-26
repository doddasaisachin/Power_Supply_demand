import pandas as pd
import numpy as np

df=pd.read_csv('power_dataset.csv')

df.head()

df.shape

df.dtypes

df['Datetime'][0]

str(df['Datetime'][0]).split('-')

new_df=pd.DataFrame(df['Datetime'].apply(lambda x : str(x).split('-')[0]))

new_df.head(5)

new_df['Month']=df['Datetime'].apply(lambda x : str(x).split('-')[1])

new_df.head(5)

new_df['Year']=new_df['Datetime']

new_df.head(2)

new_df.drop('Datetime',axis=1,inplace=True)

new_df.head(5)

x=new_df.pop('Month')

new_df['Month']=x

new_df.head()

str(df['Datetime'][5]).split(' ')[0].split('-')

new_df['Day']=df['Datetime'].apply( lambda x : str(x).split(' ')[0].split('-')[-1])

str(df['Datetime'][0]).split(' ')[-1].split(':')[0]

new_df['Hours']=df['Datetime'].apply(lambda x : str(x).split(' ')[1].split(':')[0])

new_df.head(5)

new_df.shape

new_df.isna().sum()

df.columns

new_df[df.columns[1]]=df['PJMW_MW']

df.head(3)

new_df.head(3)

new_df.shape

new_df.isna().sum()

# new_df['PJMW_MW'].plot()

new_df.describe()

new_df[new_df['PJMW_MW']<2500]

new_df.drop(11828,inplace=True)

new_df.describe()

# new_df['PJMW_MW'].plot()

new_df.tail(2)

new_df.shape

def checkforobj(col):
    for i in range(len(new_df[col])):
        if i==11828:
            pass
        else:
            try:
                int(new_df[col][i])
            except Exception as e:
                print(f"columns {col} ; index : {i} ; error {e}")

checkforobj('Year')

checkforobj('Month')

checkforobj('Hours')

new_df.shape

new_df['conv_year']=new_df['Year'].apply(lambda x : int(x))

new_df['conv_month']=new_df['Month'].apply(lambda x : int(x))

new_df['conv_day']=new_df['Day'].apply(lambda x : int(x))

new_df['conv_hours']=new_df['Hours'].apply(lambda x : int(x))

new_df.columns

new_df.head()

new_df['conv_year']

from sklearn.model_selection import train_test_split
y=new_df['PJMW_MW']
x=new_df.iloc[:,5:].values

x

y

x.shape

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.17)

xtrain.shape,ytrain.shape,xtest.shape,ytest.shape

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

dec_tree_model=DecisionTreeRegressor()

dec_tree_model.fit(xtrain,ytrain)

dec_tree_model.score(xtest,ytest)*100

xtest[:5]

ytest[:5]

dec_tree_model.predict(xtest[:5])



forest_model=RandomForestRegressor()

forest_model.fit(xtrain,ytrain)

forest_model.score(xtest,ytest)*100

xtest[:5]

y_pred=forest_model.predict(xtest[:5])
y_pred

ytest[:5]

from sklearn.metrics import r2_score

r2_score(ytest[:5],y_pred)

# import matplotlib.pyplot as plt

# plt.plot(list(ytest))

# plt.plot(forest_model.predict(xtest))

new_df.tail(2)

x=pd.to_datetime('2018-01-02 01:00:00')

def Generate_Hours(year,month,day):
    res=year+'-'+month+'-'+day+' '
    array=[]
    for i in range(24):
        array.append(res+str(i))
    return array

forecast_30_df=pd.DataFrame(Generate_Hours('2018','8','3'),columns=['Datetime'])

forecast_30_df=pd.concat([forecast_30_df,pd.DataFrame(Generate_Hours('2018','8','4'),columns=['Datetime'])])

forecast_30_df.tail(2)

for i in range(5,32):
    forecast_30_df=pd.concat([forecast_30_df,pd.DataFrame(Generate_Hours('2018','8',str(i)),columns=['Datetime'])])

30*24

forecast_30_df.shape

forecast_30_df=pd.concat([forecast_30_df,pd.DataFrame(Generate_Hours('2018','9','1'),columns=['Datetime'])])

forecast_30_df.shape

forecast_30_df.head()

forecast_30_df['Year']=forecast_30_df['Datetime'].apply(lambda x : x.split('-')[0])

forecast_30_df['Month']=forecast_30_df['Datetime'].apply(lambda x : x.split('-')[1])

forecast_30_df['Day']=forecast_30_df['Datetime'].apply(lambda x : x.split('-')[2][0].split(' ')[0])

forecast_30_df['Hours']=forecast_30_df['Datetime'].apply(lambda x : x.split(' ')[1])

forecast_30_df.reset_index(inplace=True)

forecast_30_df.drop('index',axis=1,inplace=True)

forecast_30_df.head()

forecast_30_df.iloc[:,1:]

forecast_30_pred=forest_model.predict(forecast_30_df.iloc[:,1:])

# plt.plot(list(forecast_30_pred))

forecast_30_predicted_df=pd.DataFrame(forecast_30_df['Datetime'],columns=['Datetime'])

forecast_30_predicted_df['predicted_PJMW']=forecast_30_pred

forecast_30_predicted_df

# forecast_30_predicted_df.to_csv('Predicted_PJMW')

df.tail(5)

def predict_power_supply_demand(datetime):
    year=int(datetime.split('-')[0])
    month=int(datetime.split('-')[1])
    day=int(datetime.split('-')[-1].split(' ')[0])
    hours=int(datetime.split(' ')[1].split(':')[0])
    return forest_model.predict([[year,month,day,hours]])

import pickle

pickle.dump(forest_model,open('random_forest_model.pkl','wb'))
