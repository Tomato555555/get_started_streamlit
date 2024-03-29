import streamlit as st 
import joblib
import PIL
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#step 1 ------------------------------
#Loading Our final trained Knn model 
model_is= open("KnnClassifier_goldFish.pkl", "rb")
knn_clf=joblib.load(model_is)
st.title("Cyprinid Herpes Virus 2 (CyHV-2) in goldfish Classification App")
#Loading images
homeImage= Image.open('home.png')
pcr0= Image.open('gold-fish.png')
pcr1 = Image.open('dead-fish.png')

# Sidebar title and input fields
st.sidebar.title("Features")
number_dna = st.sidebar.number_input("DNA quantity (A260/A280)", key='number_dna',format="%.3f",value=1.842)
number_ph = st.sidebar.number_input("PH value of water",key='number_ph',format="%.3f",value=7.46)
number_temp = st.sidebar.number_input("Water temperature (degrees Celsius)",key='number_temp',format="%.3f",value=23.7)

input_data = {'ndna': number_dna,
              'ph': number_ph,
              'temp': number_temp}

input_df = pd.DataFrame([input_data])

#drop คอลัมน์เลือกเอาเฉพาะที่เกี่ยวข้อง
_column = ['ndna', 'ph','temp']
dataset_X= open("dataset_X.pkl", "rb")
_X=joblib.load(dataset_X)
_X2 = pd.DataFrame(_X, columns = _column)
union_df = pd.concat([_X2, input_df])
union_df = union_df.reset_index(drop=True)

#สร้างฟังก์ชันในการ Scale ข้อมูล
def scaler_func(dataset,_column):
    for i in _column:
        x = np.array(dataset[i]).reshape(-1,1)
        scaler = MinMaxScaler()
        x_scalar = scaler.fit_transform(x)
        dataset[i]=x_scalar
    return dataset
_X3=scaler_func(union_df,_column)

_X4_web = _X3.filter(items = [148], axis=0)
#print(_X2.filter(items = [139], axis=0))

# Classification button
if st.button("Click Here to Classify"):
    prediction = knn_clf.predict(_X4_web)
    print(prediction)
    # Display the corresponding flower image based on the prediction
    if prediction == 0:
        st.image(pcr0, caption='Non-infected', width=250)
    elif prediction == 1:
        st.image(pcr1, caption='Infected', width=250)