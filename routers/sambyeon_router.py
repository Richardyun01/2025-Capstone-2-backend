from fastapi import APIRouter
from services.sambyeon.models import *
from services.sambyeon.sambyeon import *

router = APIRouter(prefix="/sambyeon")


@router.get("/get_positiion")
def get_position(
    origin: float,
    origin_right: float,
    origin_cross_one: float,
    origin_cross_two: float,
    one_side_length: float,
    knee_to_eyes: float,
):
    # 원시 입력 데이터 구성
    raw_case = [
        (0, 0, -knee_to_eyes, origin),
        (one_side_length, 0, 0, origin_right),
        (one_side_length, one_side_length, 0, origin_cross_one),
        (one_side_length, one_side_length * 2, -knee_to_eyes, origin_cross_two),
    ]

    # 스케일 정규화
    normalized_case = normalize_if_needed(raw_case)

    # 여러 번 실행 후 중앙값
    predicted = run_multiple_times(normalized_case, runs=20)

    # 결과 반환
    return Position(x=predicted[0], y=predicted[1], z=predicted[2])
