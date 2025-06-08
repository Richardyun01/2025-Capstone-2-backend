from fastapi import APIRouter
from services.sambyeon.models import *
from services.sambyeon.sambyeon import *

router = APIRouter(prefix="/sambyeon")


@router.get("/get_positiion")
def get_position(
    up_distance: float,
    down_distance: float,
    front_distance: float,
    left_distance: float,
    arm_length: float,
):
    # 원시 입력 데이터 구성
    raw_case = [
        (0, 0, arm_length, up_distance),
        (0, 0, -arm_length, down_distance),
        (-arm_length, 0, 0, left_distance),
        (0, arm_length, 0, front_distance),
    ]

    # 스케일 정규화
    normalized_case = normalize_if_needed(raw_case)

    # 여러 번 실행 후 중앙값
    predicted = run_multiple_times(normalized_case, runs=5)

    # 결과 반환
    return Position(x=predicted[0], y=predicted[1], z=predicted[2])
