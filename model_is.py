import pandas as pd
import numpy as np 
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import scale
# from sklearn.preprocessing import StandardScaler
# 
# from pandas import DataFrame
# import seaborn as sns
# 
# 
# from sklearn.model_selection import cross_val_score

# 
# 
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import f1_score
# from numpy import mean
# from numpy import absolute
# from numpy import sqrt
# from sklearn import metrics
# 

# **Step 1: Inport Data**
import csv
df1= pd.read_csv("cyhv-2-in-goldfish-cm.csv")
#print(data)

#ดึงจาก ggsheet
# gsheetid = "1zKAIVWOP1DXvzewXK-gj-RXmmOKSBXOFxWXMCyirawg"
# sheet_name = "is_table"
# gsheet_url = f"https://docs.google.com/spreadsheets/d/{gsheetid}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
# df1 = pd.read_csv(gsheet_url)
# #print(df1.head())

# **Step 2: Prepare Data**
#2.1 drop คอลัมน์ที่ซ้ำซ้อนออก
drop_column = ['date', 'codf','den','den4','bf','exl','ects','ectst'
               ,'ectsn4','ectg','ectgt','ectgn4','qua1','qua2','pcr2']
df2 = df1.drop(drop_column, axis=1)
#print(df2.head())

#2.2. แก้ไข : แทนค่า null ด้วย ค่าเฉลี่ยของข้อมูล หรือ ค่าความถี่ที่มากที่สุด
# สำหรับตัวแปรต่อเนื่อง 7 ตัวแปร  จะแทนค่า null ด้วย ค่าเฉลี่ยของข้อมูล
df2['tw'].fillna( df2['tw'].mean(), inplace = True)
df2['tl'].fillna( df2['tl'].mean(), inplace = True)
df2['th'].fillna( df2['th'].mean(), inplace = True)
df2['wh'].fillna( df2['wh'].mean(), inplace = True)
df2['vol'].fillna( df2['vol'].mean(), inplace = True)
df2['rr'].fillna( df2['rr'].mean(), inplace = True)
df2['den5'].fillna( 4, inplace = True)

# ตัวแปรไม่ต่อเนื่อง 11 ตัวแปร จะแทนค่า null ด้วย ค่าความถี่ที่มากที่สุด
def fill_null_with_mode(dataframe):
    return dataframe.apply(lambda col: col.fillna(col.mode()[0]))

# ทำการแทนค่า null ด้วยความถี่มากสุด
df2 = fill_null_with_mode(df2)

#ผลลัพธ์ ไม่มีคอลัมน์ที่เป็น null แล้ว
#print(df2.info())

#2.3. ทำ Bootstrapping เพื่อเพิ่มจำนวนข้อมูล

#เปลี่ยน Type ตัวแปร Discrete ให้เป็น object ให้หมดก่อน
object_col = ['den1','den2','den3','den5','sw1','sw2','sw3','sw4','sw5','sw6','sw7','sw8','bf1'
             ,'bf2','exlg1','exlg2','exlg3','exlg4','exlg5','ectst2','sectsn4','ectgt2'
             ,'sectgn4','level','qua3','dura','lot','size','sepa1','sepa2','pcr1']
for col in object_col:
    df2[col] = df2[col].astype('object')

#หาจำนวนแถว ที่ต้องการจะสุ้ม
df_y0 = df2[df2['pcr1'] == 0]
df_y1 = df2[df2['pcr1'] == 1]
num_samples = len(df_y0)-len(df_y1)

#ฟังก์ชัน Bootstrapping สำหรับเพิ่มจำนวนข้อมูล
from sklearn.utils import resample
def bootstrap_dataframe(dataframe, num_samples):
    np.random.seed(42)
    bootstrap_sample = resample(dataframe[dataframe['pcr1'] == 1], 
                                replace=True, n_samples=num_samples)
    bootstrap_sample = pd.concat([dataframe, bootstrap_sample])
    bootstrap_sample = bootstrap_sample.set_index(pd.RangeIndex(start=0,
                                                                stop=len(bootstrap_sample), step=1))
    return bootstrap_sample

df2 = bootstrap_dataframe(df2,num_samples)
#print(df2)

#2.4. แก้ไข : เปลี่ยน type ตัวแปร Discrete ให้อยู่ในรูป int
#เปลี่ยน Type ตัวแปร Discrete ให้เป็น int ให้หมดก่อน
object_col = ['den1','den2','den3','den5','sw1','sw2','sw3','sw4','sw5','sw6','sw7','sw8','bf1'
             ,'bf2','exlg1','exlg2','exlg3','exlg4','exlg5','ectst2','sectsn4','ectgt2'
             ,'sectgn4','level','qua3','dura','lot','size','sepa1','sepa2','pcr1']
for col in object_col:
    df2[col] = df2[col].astype('int')

#2.5. แยกตัวแปร X และ y
X=df2.copy()
y=X.pop('pcr1')

#2.6. Mutual Information Scores
#กำหนดให้ค่าที่แปลงข้างต้นเป็น Discrete Features (ค่าที่ไม่ต่อเนื่อง)
discrete_features = X.dtypes == int

#เรียกใช้งาน mutual_info_classif โดยพารามิเตอร์ที่ต้องการคือ feature, target และกำหนด discrete_features
def make_mi_score(X,y,discrete_features):
    mi_score = mutual_info_classif(X,y,discrete_features=discrete_features)
    mi_score = pd.Series(mi_score,name ='MI Scores Clf',index = X.columns)
    mi_score = mi_score.sort_values(ascending=False)
    return mi_score
mi_score = make_mi_score(X,y,discrete_features)
mi_score.sort_values(ascending=False)
#print(mi_score.sort_values(ascending=False))

#2.7. แปลงตัวแปร Discrete เป็นตัวแปร dummy โดยใช้ one hot encoding
X2 = X.copy()
#แปลงตัวแปร Discrete เป็นตัวแปร dummy โดยใช้ one hot encoding
onehot_den5  = pd.get_dummies(X2.den5, prefix='den5').astype(int)
onehot_ectst2  = pd.get_dummies(X2.ectst2, prefix='ectst2').astype(int)
onehot_sectsn4  = pd.get_dummies(X2.sectsn4, prefix='sectsn4').astype(int)
onehot_ectgt2  = pd.get_dummies(X2.ectgt2, prefix='ectgt2').astype(int)
onehot_sectgn4  = pd.get_dummies(X2.sectgn4, prefix='sectgn4').astype(int)
onehot_level  = pd.get_dummies(X2.level, prefix='level').astype(int)
onehot_qua3  = pd.get_dummies(X2.qua3, prefix='qua3').astype(int)
onehot_dura  = pd.get_dummies(X2.dura, prefix='dura').astype(int)

#ลบคอลัมน์เก่าทิ้ง
drop_col = ['den5','ectst2','sectsn4','ectgt2','sectgn4','level','qua3','dura']
X2 = X2.drop(drop_col, axis=1)

#เพิ่มคอลัมน์ที่แปลงเป็น dummy เรียบร้อยแล้วเข้าไปในชุดข้อมูล 
X2 = X2.join(onehot_den5)
X2 = X2.join(onehot_ectst2)
X2 = X2.join(onehot_sectsn4)
X2 = X2.join(onehot_ectgt2)
X2 = X2.join(onehot_sectgn4)
X2 = X2.join(onehot_level)
X2 = X2.join(onehot_qua3)
X2 = X2.join(onehot_dura)
#print(X2.head())

#2.8. ทำ Feature scaling โดยใช้สูตร MinMaxScaler
#เลือกตัวแปรที่เป็นประเภท Continuous Variables มาเพื่อคำนวนหา Normalize data
num_column = ['th','vol','tl','dsdna','temp','ndna','wh','ph','tlf']

#สร้างฟังก์ชันในการ Scale ข้อมูล
def scaler_func(dataset,num_column):
    for i in num_column:
        x = np.array(dataset[i]).reshape(-1,1)
        scaler = MinMaxScaler()
        x_scalar = scaler.fit_transform(x)
        dataset[i]=x_scalar
    return dataset
X2=scaler_func(X2,num_column)
#print(X2.head())

#2.9. เลือก Fixed Threshold ที่ทำให้โมเดลดีที่สุด
#สร้างฟังก์ชัน threshold เพื่อเลือกตัวแปรจากค่า MI Score โดยใช้วิธี Fixed Threshold
def threshold(mi_score,threshold_value):
    # กำหนดค่า threshold
    threshold_value = threshold_value

    # ใช้ Fixed Threshold เพื่อแบ่งตัวแปรจากค่า mi_score
    selected_features  = mi_score.where(mi_score >= threshold_value).dropna()
    selected_features_column = selected_features .index.to_list()    
    return(selected_features_column)

list_acc_train = []
list_acc_test = []
list_col=[]
list_numCol=[]
for j in [0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28]:
    
    # ใช้ Fixed Threshold เพื่อเลือกตัวแปรจากค่า mi_score =j
    col = threshold(mi_score,j)
    # เลือกตัวแปรตาม col
    filtered_columns = []
    filtered_columns = X2.filter(like=col[0])
    for i in col[1:]: 
         filtered_columns = pd.concat([filtered_columns, X2.filter(like=i)], axis=1)
    #ลบ col ซ้ำออก
    filtered_columns =filtered_columns.loc[:, ~filtered_columns.columns.duplicated()]
    
    # แบ่งข้อมูลออกเป็นชุดข้อมูล training 90% และ ชุดข้อมูล test 10%
    X0=filtered_columns.copy()
    y0=y
    X_train_dtc0, X_test_dtc0, y_train_dtc0, y_test_dtc0 = train_test_split(X0,y0, test_size = 0.10, random_state = 42)
    
    # สร้างโมเดล DecisionTreeClassifier
    #DecisionTree_model = DecisionTreeClassifier(max_depth=5,random_state=42)
    _model = KNeighborsClassifier(n_neighbors=2)
    _model.fit(X_train_dtc0, y_train_dtc0)
    y_traingPred_dtc0 = _model.predict(X_train_dtc0)
    y_testPred_dtc0 = _model.predict(X_test_dtc0)
    
    
    # วัดประสิทธิภาพของแต่ละโมเดล
    acc_train = round(accuracy_score(y_train_dtc0, y_traingPred_dtc0)*100,4)
    acc_test = round(accuracy_score(y_test_dtc0, y_testPred_dtc0)*100,4)
    
    list_acc_train.append(acc_train)
    list_acc_test.append(acc_test)
    list_col.append(col)
    list_numCol.append(len(col))

list_mi =  [0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28]
Acc_table0 = pd.DataFrame({'accuracy_Training':list_acc_train,'accuracy_Test': list_acc_test,'col':list_col,'number of col':list_numCol}
                          ,index =list_mi)
#print(Acc_table0)

#2.10 แบ่งข้อมูลออกเป็นชุดข้อมูล training 90% และ ชุดข้อมูล test 10%

col = threshold(mi_score,0.23)
    
# เลือกตัวแปรตาม col
filtered_columns = []
filtered_columns = X2.filter(like=col[0])
for i in col[1:]: 
    filtered_columns = pd.concat([filtered_columns, X2.filter(like=i)], axis=1)
#ลบ col ซ้ำออก
filtered_columns =filtered_columns.loc[:, ~filtered_columns.columns.duplicated()]

# แบ่งข้อมูลออกเป็นชุดข้อมูล training 90% และ ชุดข้อมูล test 10%
X3=filtered_columns.copy()
y3=y
X_train, X_test, y_train, y_test = train_test_split(X3,y3, test_size = 0.10, random_state = 42)

#**Step 3: สร้างโมเดล**
loocv = LeaveOneOut()
model_knn = KNeighborsClassifier(n_neighbors=2)

model_knn.fit(X_train, y_train)
y_traingPred_knn = model_knn.predict(X_train)
y_testPred_knn  = model_knn.predict(X_test)
#print(accuracy_score(y_train, y_traingPred_knn)*100,4)

import joblib
# Save the model as a pickle in a file
#joblib.dump(model_knn, "KnnClassifier_goldFish.pkl")
#joblib.dump(X, "dataset_X.pkl")