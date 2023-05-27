from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel 
import pickle
import pandas as pd
import model
import bz2
app = FastAPI()

class item(BaseModel):
    name:str
    calories:int
    fat:float
    protein:float
    iron:float
    calcium:float
    sodium:float
    potassm :float
    carbohydrate:float
    fiber:float
    vitamin_d:float
    sugars:float

ifile = bz2.BZ2File("CompresedPickle.pickle" , "rb")
RunningModel = pickle.load(ifile)
ifile.close()

def CompleteInputDummies(AllDataDummiesColumns , inputDummiesColumns):
    Dummeies = []
    
    for i in AllDataDummiesColumns:
        state=False
        for x in inputDummiesColumns:
           if i==x:
               state=True
        if state==True:
            Dummeies.append(1)
        else:
            Dummeies.append(0)
    return Dummeies                    
@app.post("/")    
def App(item:item):
    
    nameData = pd.Series(item.name)
    nameDummies = pd.Series(nameData).str.get_dummies(sep=",")

    AllDataDummies = model.namesDataFrameFinal
    
    Ou=CompleteInputDummies(AllDataDummies , nameDummies)
    TextDummies = pd.DataFrame([Ou] ,columns=list(model.namesDataFrameFinal.columns) )
    values = pd.DataFrame([{"calories":item.calories ,"total_fat" :item.fat ,"protein":item.protein ,"iron":item.iron  ,"calcium" :item.calcium , "sodium":item.sodium , "potassm":item.potassm , "carbohydrate":item.carbohydrate , "fiber":item.fiber, "vitamin_d":item.vitamin_d , "sugars":item.sugars}])
    X_input = pd.concat([values , TextDummies] , axis=1)
    distances, indices = RunningModel.kneighbors(X_input)
   

    return {
        "name":[model.data.iloc[indices.item(0),0]],
        "name2":[model.data.iloc[indices.item(1),0]],
        "name3":[model.data.iloc[indices.item(2),0]]
    }

