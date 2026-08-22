from fastapi import FastAPI
import uvicorn 


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


if __name__ == "__main__":
    uvicorn.run(app, host='192.168.178.88', port=8000)