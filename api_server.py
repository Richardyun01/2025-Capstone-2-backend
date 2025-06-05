from fastapi import FastAPI, Request, HTTPException, status, Depends  # FastAPI import
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from routers import feature_router
from routers import sambyeon_router
from routers import feature_pattern_router

app = FastAPI(
    title="Backend API",
    description="API 통합",
    version="1.0.0",
)  # (lifespan=lifespan)

app.include_router(sambyeon_router.router)
app.include_router(feature_router.router)
app.include_router(feature_pattern_router.router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인에서 요청 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)


@app.get("/")
def main():
    # db.execute("SELECT count(*) FROM testDB.bookTable")
    # result = db.fetchall()
    return "Hello world!"
