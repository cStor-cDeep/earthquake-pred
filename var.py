import pandas as pd

data = pd.read_csv("data_all1.csv",sep=',',encoding = "utf-8")
print(data)
print(data['label'].isnull().any())
if data["label"].isnull().any():
     print(data[data.isnull().values==True])
     print(data[data.isna().values==True])



'''
df = data.dropna(axis=0)

print(df)
if df["label"].isnull().any():
     print(df[df.isnull().values==True])
     print(df[df.isna().values==True])
df.to_csv("data_all1.csv",sep=',',encoding='utf-8')

#print(data.describe())
data1=data.iloc[:,2:]
print(data1)
print(data1.shape)
var = data1.var(axis=0)
print(var)
'''