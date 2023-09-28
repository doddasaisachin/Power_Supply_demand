from flask import Flask,render_template,request
import pickle
applicaton=Flask(__name__)
app=applicaton

def predict_power_supply_demand(model,datetime):
    year=int(datetime.split('-')[0])
    month=int(datetime.split('-')[1])
    day=int(datetime.split('-')[-1].split(' ')[0])
    hours=int(datetime.split(' ')[1].split(':')[0])
    return model.predict([[year,month,day,hours]])

@app.route('/',methods=['POST','GET'])
def home():
    if request.method=='GET':
        return render_template('index.html')
    else:
        datetime=request.form.get('Datetime')

        def predict_power_supply_demand(model,datetime):
            year=int(datetime.split('-')[0])
            month=int(datetime.split('-')[1])
            day=int(datetime.split('-')[-1].split(' ')[0])
            hours=int(datetime.split(' ')[1].split(':')[0])
            return model.predict([[year,month,day,hours]])

        loaded_model = pickle.load(open('random_forest_model.pkl', 'rb'))
        predicted_power=predict_power_supply_demand(loaded_model,datetime)
        predicted_power=predicted_power[0]

        return render_template('index.html',date=datetime,result=predicted_power)
if __name__=='__main__':
    app.run(host='0.0.0.0')


