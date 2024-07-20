from fastapi import FastAPI, File, UploadFile, HTTPException
import xray_learning
import os
import shutil
from tempfile import NamedTemporaryFile

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "The Covid-Detection AI assist is online!"}


@app.post("/xray")
async def xray_detection_handler(image: UploadFile = File(...)):
    try:
        with NamedTemporaryFile(delete=False, suffix=image.filename) as temp_file:
            shutil.copyfileobj(image.file, temp_file)
            temp_file_path = temp_file.name

        return {"diagnosis": xray_learning.predict(temp_file_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


