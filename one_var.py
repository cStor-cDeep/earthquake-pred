import pandas as pd

data = pd.read_csv("alldata.csv",sep=",",encoding="utf-8")
data = data.dropna(axis=0, how='any')
print(data)

station_id=data["StationID"].unique()
print(station_id)
list1 = []
for i in range(len(station_id)):
    data1 = data[data["StationID"] == station_id[i]]
    data2 = data1.iloc[:,3:-1]
    var = data2.var(axis=0)
    var1 = var.tolist()
    list1.append(var1)
    print(station_id[i])
    print(var1)
    print("****************")
#print(list1)
#pd.DataFrame(list1).to_csv("var_all.csv",sep=",",encoding="utf-8")