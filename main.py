from fastapi import FastAPI
import uvicorn
from contextlib import asynccontextmanager

from lib.model_manager import load_models, clear_all_models

from domain import info_router
from domain import predict_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    clear_all_models()


app = FastAPI(lifespan=lifespan)


# Including API routers
app.include_router(info_router.router, prefix="/api/info")
app.include_router(predict_router.router, prefix="/api/predict")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9090, reload=False)