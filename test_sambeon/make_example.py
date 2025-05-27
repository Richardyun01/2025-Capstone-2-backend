import random
import math


def generate_point_within_range(arm_length, min_distance, max_distance):
    fixed_points = {
        "up": [0, 0, arm_length],
        "down": [0, 0, -arm_length],
        "front": [0, arm_length, 0],
        "left": [-arm_length, 0, 0],
    }

    # 반복해서 조건을 만족하는 A를 찾음
    while True:
        # 랜덤한 A 좌표 생성 (범위는 arm_length의 ±2배 정도로 넉넉히 잡음)
        A = [
            random.uniform(-2 * arm_length, 2 * arm_length),
            random.uniform(-2 * arm_length, 2 * arm_length),
            random.uniform(-2 * arm_length, 2 * arm_length),
        ]

        # 모든 고정 좌표로부터 거리 계산
        distances = {}
        valid = True
        for name, point in fixed_points.items():
            dist = math.sqrt(
                (A[0] - point[0]) ** 2 + (A[1] - point[1]) ** 2 + (A[2] - point[2]) ** 2
            )
            if not (min_distance < dist < max_distance):
                valid = False
                break
            distances[name] = dist

        if valid:
            # 조건 만족하는 A 찾으면 리턴
            return {"A": A, "distances": distances}


# 예제 실행
result = generate_point_within_range(arm_length=65, min_distance=50, max_distance=150)
print(f"Generated Point A: {result['A']}")
for name, dist in result["distances"].items():
    print(f"Distance to {name}: {dist:.2f}")
