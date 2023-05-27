import pandas as pd
from sklearn.neighbors import NearestNeighbors
import re
import pickle
import bz2

data = pd.read_csv("nutritionEdited.csv")
data =data[["name","calories" , "total_fat" , "protein" , "iron" , "calcium" , "sodium" ,"potassm" , "carbohydrate","fiber","vitamin_d","sugars" ]]
RealOne = data[["calories" , "total_fat" , "protein" , "iron" , "calcium" , "sodium" ,"potassm" , "carbohydrate","fiber","vitamin_d","sugars" ]]


name = data.iloc[: , 0]
def prepareNameData(data):
    NewList = []
    pattern1 = re.compile(r",? (unheated|heated|white|red|strained|imported|raw|cooked|uncooked|dried|canned|fried|boiled|unenriched|dry|smoked|prepared|baked|sweetened|whipped|home-prepared|frozen|unprepared|pressurized|round|enriched|sweetened|seeded|fresh|broiled|fillin|filled|drained|roasted|braised)|(,? \(\w+\)|(\d+)|([$&+:;=?@#|'<>.^*%!-//]))|((select|choice|allrades|trimmed).*)|(separable.*)")
    for i in data:
        PreparedItems = re.sub(pattern1 , "" , i)
        NewList.append(PreparedItems)
    return NewList    
namesDataFrame = prepareNameData(name)


def cleanAnd(data):
    NewList = []
    pattern1 = re.compile(r"(and |And |with |or )")
    for i in data:
        PreparedItems = re.sub(pattern1 , "," , i)
        NewList.append(PreparedItems)
    return NewList
namesDataFrame2 = cleanAnd(namesDataFrame)


def addSeprator(data):
    list2 = []
    pattern = re.compile(r"\s")
    for i in data:
        NewItem=re.sub(pattern,",",i)
        list2.append(NewItem)
    return list2
FinalData = addSeprator(namesDataFrame2)

namesDataFrameFinal = pd.Series(FinalData).str.get_dummies(sep=",")

AllData = pd.concat([RealOne , pd.DataFrame(namesDataFrameFinal)] , axis=1)

X_train = AllData[:7000]
X_test = AllData[7000:]


nbrs = NearestNeighbors(n_neighbors=3,radius=1 ,algorithm='auto').fit(X_train)
