from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rest_server.database import close_pool, init_pool
from rest_server.routes import datasheets, model_cards


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_pool()
    yield
    await close_pool()


app = FastAPI(
    title="Patra Privacy API",
    description="API for model cards and datasheets with JWT-aware privacy",
    version="1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(model_cards.router)
app.include_router(datasheets.router)


@app.get("/")
async def root():
    return {"message": "Welcome to the Patra Privacy API"}
