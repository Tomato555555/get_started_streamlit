import pandas as pd
import numpy as np 
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score 

# **Step 1: Inport Data**
import csv
df1= pd.read_csv("cyhv-2-in-goldfish-cm.csv")
#print(df1)

# **Step 2: Prepare Data**
#2.1 drop คอลัมน์ที่ซ้ำซ้อนออก
drop_column = ['date', 'codf','den1','den2','den3','den4','bf','exl','ects','ectst'
               ,'ectsn4','ectg','ectgt','ectgn4','qua1','qua2','ndna','dsdna','pcr2']
df2 = df1.drop(drop_column, axis=1)
#print(df2.head())
#print(df2.info())
#2.2 เปลี่ยน Type ตัวแปร 
# เปลี่ยน Type ตัวแปร   จาก object ให้เป็น numeric ให้หมดก่อน
object_col = ['DO','temp','ammo','nit','ph','tw','tl','th','wh','vol','rr','bw','tlf','slf']
for col in object_col:
    df2[col] = pd.to_numeric(df2[col], errors='coerce') 

#2.3. แก้ไขค่า null ด้วย ค่าเฉลี่ยของข้อมูล หรือ ค่าความถี่ที่มากที่สุด
# แบ่ง DataFrame เป็นสองชุดตามค่าของ Y
df_y0 = df2[df2['pcr1'] == 0].copy()
df_y1 = df2[df2['pcr1'] == 1].copy()
# print(df_y0.info())
# สำหรับ Quantitative data(ข้อมูลเชิงปริมาณ) 6 ตัวแปร  จะแทนค่า null ด้วย ค่าเฉลี่ยของข้อมูล 
# สำหรับชุดข้อมูลไม่ติดเชื้อ
cols_to_fill = ['tw', 'tl', 'th', 'wh', 'vol', 'rr']
# สำหรับชุดข้อมูลไม่ติดเชื้อ
df_y0.fillna(df_y0.mean()[cols_to_fill].to_dict(), inplace=True)

# สำหรับชุดข้อมูลติดเชื้อ
df_y1.fillna(df_y1.mean()[cols_to_fill].to_dict(), inplace=True)

# ตัวแปรไม่ต่อเนื่อง จะแทนค่า null ด้วย ค่าความถี่ที่มากที่สุด
# ตรวจสอบและกำหนด dtype ก่อน
df_y0 = df_y0.infer_objects()
df_y1 = df_y1.infer_objects()

# จากนั้นใช้ fillna ได้อย่างปลอดภัย
df_y0 = df_y0.apply(lambda col: col.fillna(col.mode()[0]))
df_y1 = df_y1.apply(lambda col: col.fillna(col.mode()[0]))



#union ข้อมูล
df2 = pd.concat([df_y0, df_y1]).sort_index(axis=0)

# เปลี่ยน Type ตัวแปร   จาก int ให้เป็น object ให้หมดก่อน
# เปลี่ยน Type ตัวแปร   จาก int ให้เป็น object ให้หมดก่อน
# for i in df2.columns:
#     df2[i] = df2[i].astype('object')

object_col = ['DO','temp','ammo','nit','ph','tw','tl','th','wh','vol','rr','bw','tlf','slf']
for k in df2.columns:
    if k in object_col:
        df2[k] = pd.to_numeric(df2[k], errors='coerce') 
    else: df2[k] = df2[k].astype('int64')
#print(df2.info())
'''
for i in df2.columns:
    df2[i] = df2[i].astype('object')
# เปลี่ยน Type ตัวแปร   จาก object ให้เป็น numeric ให้หมดก่อน
object_col = ['DO','temp','ammo','nit','ph','tw','tl','th','wh','vol','rr','bw','tlf','slf']
for col in object_col:
    df2[col] = pd.to_numeric(df2[col], errors='coerce') 
'''
#print(df2.info())

#2.4. ทำ Bootstrapping เพื่อเพิ่มจำนวนข้อมูล

#หาจำนวนแถว ที่ต้องการจะสุ้ม
df_y0 = df2[df2['pcr1'] ==0] #จำนวนแถวของตัวอย่างไม่ติดเชื้อ
df_y1 = df2[df2['pcr1'] == 1] #จำนวนแถวของตัวอย่างติดเชื้อ
#print(df_y0)
num_samples = len(df_y0)-len(df_y1)

#ฟังก์ชัน Bootstrapping สำหรับเพิ่มจำนวนข้อมูล
from sklearn.utils import resample
def bootstrap_dataframe(dataframe, num_samples):
    np.random.seed(42) #ทำให้สุ่มแต่แบบเดิมทุกครั้ง
    bootstrap_sample = resample(dataframe[dataframe['pcr1'] == 1],
                                replace=True, n_samples=num_samples) #ต้องการสุ่มแค่ตัวอย่างที่ติดเชื้อ num_samples ตัวอย่าง
    bootstrap_sample = pd.concat([dataframe, bootstrap_sample]) #union ข้อมูลเดิม กับ ข้อมูลที่strapping
    bootstrap_sample = bootstrap_sample.set_index(pd.RangeIndex(start=0,
                                                                stop=len(bootstrap_sample), step=1)) #set index ใหม่
    return bootstrap_sample

df2_bootstrap = bootstrap_dataframe(df2,num_samples)
#print(df2_bootstrap)

#2.5. แยกตัวแปร X และ y
X=df2_bootstrap.copy()
y=X.pop('pcr1')
#print(df2.info())
#2.6. Mutual Information Scores
#กำหนดให้ค่าที่แปลงข้างต้นเป็น Discrete Features (ค่าที่ไม่ต่อเนื่อง)
discrete_features = X.dtypes == 'int64'

#เรียกใช้งาน mutual_info_classif โดยพารามิเตอร์ที่ต้องการคือ feature, target และกำหนด discrete_features
def make_mi_score(X,y,discrete_features):
    mi_score = mutual_info_classif(X,y,discrete_features=discrete_features)
    mi_score = pd.Series(mi_score,name ='MI Scores Clf',index = X.columns)
    mi_score = mi_score.sort_values(ascending=False) #sortคะแนน
    return mi_score


mi_score = make_mi_score(X,y,discrete_features).sort_values(ascending=False)
#สร้าง dataframe
features_mi = pd.DataFrame({'Feature':mi_score.index, 'MI':mi_score.values})
x_axis = features_mi['Feature']
y_axis = features_mi['MI']
#print(discrete_features)

#2.6. Feature Selection จากค่า MI Score
#สร้างฟังก์ชัน threshold เพื่อเลือกตัวแปรจากค่า MI Score โดยใช้วิธี Fixed Threshold
def threshold(mi_score,threshold_value):
    # ใช้ Fixed Threshold เพื่อเลือกตัวแปรจากค่า mi_score
    selected_features  = mi_score.where(mi_score >= threshold_value).dropna() #เลือกตัวแปรจากค่า mi_score
    selected_features_column = selected_features .index.to_list() #แปลง index (ชื่อตัวแปร) เป็น list
    return(selected_features_column)
'''
#ทดสอบหา ค่า mi_score ที่ดีที่สุดจากโมเดล KNeighborsClassifier(n_neighbors=2)
list_acc_train = []
list_acc_test = []
list_col = []

#หา mi score ที่มากที่สุด
max_y = y_axis.max()
if(max_y-round(max_y,2))<=0.005:
    max_y = round(max_y,2)+0.01 #เนื่องจากถ้า ค่า mi score ตัวไหน น้อยกว่าเท่ากับ 0.005 จะปัดลง แต่เราอยากให้ปัดขึ้นจึง +0.01
else: max_y=round(max_y,2)

for j in np.arange(0,max_y, 0.01): #กำหนดค่า Fixed Threshold ตั้งแต่ 0 ถึง ค่ามากสุดของ mi score

    #สร้างตาราง X :โดยเลือกตัวแปรจากการใช้เทคนิค  Fixed Threshold จากค่า mi score
    col = threshold(mi_score,j) #เลือกตัวแปร
    df_selected_features = pd.DataFrame(X,columns = col)

    # แบ่งข้อมูลออกเป็นชุดข้อมูล training 80% และ ชุดข้อมูล test 20%
    X0=df_selected_features.copy()
    y0=y
    X_train_dtc0, X_test_dtc0, y_train_dtc0, y_test_dtc0 = train_test_split(X0,y0, test_size = 0.20, random_state = 42)

    #ทดสอบ ***
    # โดยสร้างโมเดล KNeighborsClassifier
    _model = KNeighborsClassifier(n_neighbors=2)
    _model.fit(X_train_dtc0, y_train_dtc0)
    y_traingPred_dtc0 = _model.predict(X_train_dtc0)
    y_testPred_dtc0 = _model.predict(X_test_dtc0)

    #วัดประสิทธิภาพของแต่ละโมเดล โดยใช้ค่า accuracy
    acc_train = round(accuracy_score(y_train_dtc0, y_traingPred_dtc0)*100,4)
    acc_test = round(accuracy_score(y_test_dtc0, y_testPred_dtc0)*100,4)

    #เก็บค่า accuracy ของแต่ละ Fixed Threshold
    list_acc_train.append(acc_train)
    list_acc_test.append(acc_test)
    list_col.append(col)
    
    
    # วัดประสิทธิภาพของแต่ละโมเดล
    acc_train = round(accuracy_score(y_train_dtc0, y_traingPred_dtc0)*100,4)
    acc_test = round(accuracy_score(y_test_dtc0, y_testPred_dtc0)*100,4)
    
    list_acc_train.append(acc_train)
    list_acc_test.append(acc_test)
    list_col.append(col)

list_mi =np.arange(0,max_y, 0.01)
Acc_table0 = pd.DataFrame({'accuracy_Training':list_acc_train,'accuracy_Test': list_acc_test,'feature':list_col}
                          ,index =list_mi) 
#print(Acc_table0)
'''
#ตัวแปรที่เกี่ยวข้องกับการติดเชื้อ
col = threshold(mi_score,0.19)
    
col_final = threshold(mi_score,0.19)
# create a dataframe
mi_df = pd.DataFrame(X,columns = col_final)

#2.7 ปรับขนาดข้อมูล โดยใช้ MinMaxScaler (ปรับให้อยู่ในช่วง 0-1)
X_miScore = mi_df.copy()
#เลือกคอลัมน์ที่ต้องการปรับขนาด
num_column =  ['ph', 'temp', 'tlf', 'tl']
# สร้างตัวแปร MinMaxScaler
scaler = MinMaxScaler()
# ปรับขนาดข้อมูลเฉพาะคอลัมน์ที่เลือก
X_miScore[num_column] = scaler.fit_transform(X_miScore[num_column])

#2.8 แบ่งข้อมูลออกเป็นชุดข้อมูล training 80% และ ชุดข้อมูล test 20%
y = pd.to_numeric(y)
X_train, X_test, y_train, y_test = train_test_split(X_miScore,y, test_size = 0.20, random_state = 42)

#**Step 3: สร้างโมเดล**
RF_model=RandomForestClassifier(n_estimators=18,random_state=60)
model_RF_use = RF_model.fit(X_train,y_train)
y_testPred_rfc2 = model_RF_use.predict(X_test)
#print(accuracy_score(y_test, y_testPred_rfc2)*100)

import joblib
# Save the model as a pickle in a file
# joblib.dump(model_RF_use, "RFClassifier_goldFish.pkl")
# joblib.dump(X, "dataset_X.pkl")