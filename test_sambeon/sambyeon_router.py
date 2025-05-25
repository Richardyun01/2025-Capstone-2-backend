from fastapi import APIRouter
from .models import *
from test_sambeon.sambyeon import AP3D, Trilateration3D

router = APIRouter(prefix="/sambyeon")


@router.get("/get_positiion")
def get_position(
    up_distance: float,
    down_distance: float,
    front_distance: float,
    left_distance: float,
    arm_length: float,
):
    # AP 좌표 설정
    up_ap = AP3D(0, 0, arm_length, up_distance)
    down_ap = AP3D(0, 0, -arm_length, down_distance)
    left_ap = AP3D(-arm_length, 0, 0, left_distance)
    front_ap = AP3D(0, arm_length, 0, front_distance)

    aps = [up_ap, down_ap, left_ap, front_ap]

    # 위치 계산
    trilateration = Trilateration3D(aps)
    predicted = trilateration.calcUserLocation()

    # 결과 반환
    return Position(x=predicted[0], y=predicted[1], z=predicted[2])
