from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import uvicorn
from additional_functions import join_text_columns, extract_text_column
from io import StringIO
from fastapi import File, UploadFile
from sklearn.model_selection import train_test_split
from fastapi.middleware.cors import CORSMiddleware
from pipeline_funcional import pipe
from typing import List

try:
    pipe = pickle.load(open("modelo_entrenado.pkl", "rb"))
except FileNotFoundError:
    raise HTTPException(status_code=400, detail="Modelo no encontrado. Asegúrate de haber entrenado el modelo previamente.")
app = FastAPI(title="API para detección y reentrenamiento")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextData(BaseModel):
    Titulo: str
    Descripcion: str

class RetrainData(BaseModel):
    Titulo: str
    Descripcion: str
    Label: int

@app.post("/predict")
async def predict(data: List[TextData]):
    if isinstance(data, list):
        df = pd.DataFrame([[item.Titulo, item.Descripcion] for item in data], columns=["Titulo", "Descripcion"])
    elif isinstance(data, TextData):
        df = pd.DataFrame([[data.Titulo, data.Descripcion]], columns=["Titulo", "Descripcion"])
    else:
        raise HTTPException(status_code=400, detail="Tipo de archivo o información pasada de forma errónea")

    try:
        prediction = pipe.predict(df[["Titulo", "Descripcion"]])
        decision_scores = pipe.decision_function(df[["Titulo", "Descripcion"]])
        probabilities_list = [{"Verdadera": float(1/(1 + np.exp(-score))), "Falsa": float(1/(1 + np.exp(score)))} for score in decision_scores]

        result = []
        for i in range(len(data)):
            pred = prediction[i]
            prob = probabilities_list[i]
            prediction_text = f"{pred} ({'Verdadera' if pred == 1 else 'Falsa'})"
            result.append({
                "Titulo": data[i].Titulo,
                "Predicciones": prediction_text,
                "Probabilidades": prob
            })
        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/retrain")
async def retrain(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        data = StringIO(contents.decode("ISO-8859-1"))
        df = pd.read_csv(data, sep=";", encoding='ISO-8859-1')

        if not {"Titulo", "Descripcion", "Label"}.issubset(df.columns):
            raise HTTPException(status_code=400, detail="El archivo CSV debe contener las columnas 'Titulo', 'Descripcion' y 'Label'")

        df.drop_duplicates(subset=["Titulo", "Descripcion"], keep="first", inplace=True)
        X = df[["Titulo", "Descripcion"]]
        y = df["Label"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe.fit(X_train, y_train)  
        y_pred = pipe.predict(X_test)

        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        with open("modelo_entrenado.pkl", "wb") as f:
            pickle.dump(pipe, f)

        return {
            "message": "El modelo ha sido reentrenado exitosamente. Estas son las métricas del nuevo modelo:",
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    except Exception as e:
        print("❌ Error al procesar el archivo:", str(e))
        raise HTTPException(status_code=400, detail=f"Error al procesar el archivo: {str(e)}")

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)