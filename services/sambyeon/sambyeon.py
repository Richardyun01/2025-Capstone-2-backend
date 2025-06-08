import numpy as np
from scipy.optimize import least_squares
from copy import deepcopy
from .testData import *


class AP3D:
    def __init__(self, x, y, z, distance):
        self.x = x
        self.y = y
        self.z = z
        self.distance = distance


class Trilateration3D:
    def __init__(self, APs):
        if len(APs) != 4:
            raise ValueError(
                "Exactly 4 anchor points are required for 3D trilateration."
            )
        self.APs = APs

    def _residuals(self, guess):
        x, y, z = guess
        return [
            np.linalg.norm([x - ap.x, y - ap.y, z - ap.z]) - ap.distance
            for ap in self.APs
        ]

    def _closed_form_solution(self):
        # 기준점: AP1
        P1, P2, P3, P4 = self.APs

        # 위치 벡터
        p1 = np.array([P1.x, P1.y, P1.z])
        p2 = np.array([P2.x, P2.y, P2.z])
        p3 = np.array([P3.x, P3.y, P3.z])
        p4 = np.array([P4.x, P4.y, P4.z])

        # 거리
        r1, r2, r3, r4 = P1.distance, P2.distance, P3.distance, P4.distance

        # 좌표 차이 계산
        ex = (p2 - p1) / np.linalg.norm(p2 - p1)
        i = np.dot(ex, p3 - p1)
        ey = p3 - p1 - i * ex
        ey /= np.linalg.norm(ey)
        ez = np.cross(ex, ey)

        d = np.linalg.norm(p2 - p1)
        j = np.dot(ey, p3 - p1)
        x = (r1**2 - r2**2 + d**2) / (2 * d)
        y = (r1**2 - r3**2 + i**2 + j**2 - 2 * i * x) / (2 * j)

        z_square = r1**2 - x**2 - y**2

        if z_square < 0:
            z_square = 0  # 거리 오차가 있을 경우 음수 방지
        z = np.sqrt(z_square)

        est = p1 + x * ex + y * ey + z * ez
        return est

    def calcUserLocation(self):
        initial_guess = self._closed_form_solution()
        result = least_squares(
            self._residuals, initial_guess, method="trf", loss="soft_l1", max_nfev=1000
        )
        return result.x


def add_noise_to_distances(test_cases, noise_level=0.05):
    noisy_cases = deepcopy(test_cases)
    for case in noisy_cases:
        for i in range(len(case)):
            x, y, z, d = case[i]
            noise = np.random.normal(
                0, d * noise_level
            )  # 평균 0, 표준편차 = 거리 * noise_level
            noisy_distance = d + noise
            case[i] = (x, y, z, noisy_distance)
    return noisy_cases


def normalize_test_cases_to_m(test_cases, threshold=100, scale=100):
    normalized = []
    scale_flags = []  # 어떤 케이스가 scaling 되었는지 저장
    for case in test_cases:
        count_large_d = sum(1 for (_, _, _, d) in case if d >= threshold)
        if count_large_d >= 3:  # cm -> m 단위 치환
            normalized_case = [
                (x / scale, y / scale, z / scale, d / scale) for x, y, z, d in case
            ]
            scale_flags.append(True)
        else:
            normalized_case = case
            scale_flags.append(False)
        normalized.append(normalized_case)
    return normalized, scale_flags


# api 용
def normalize_if_needed(case, threshold=100, scale=100):
    count_large_d = sum(1 for (_, _, _, d) in case if d >= threshold)
    if count_large_d >= 3:
        normalized_case = [
            (x / scale, y / scale, z / scale, d / scale) for x, y, z, d in case
        ]
    else:
        normalized_case = case

    return normalized_case


def run_multiple_times(noisy_case, runs=5):
    positions = []

    for _ in range(runs):
        aps = [AP3D(x, y, z, d) for x, y, z, d in noisy_case]
        trilateration = Trilateration3D(aps)
        predicted = trilateration.calcUserLocation()
        positions.append(predicted)

    median_position = np.median(np.array(positions), axis=0)
    return median_position


if __name__ == "__main__":
    temp, scaled_flags = normalize_test_cases_to_m(test_cases)
    test_cases_with_noise = add_noise_to_distances(temp, noise_level=0.05)

    for idx, (case, true_pos, scaled) in enumerate(
        zip(test_cases_with_noise, true_positions, scaled_flags)
    ):
        predicted = run_multiple_times(case, runs=5)

        # true_pos를 스케일링 적용 여부에 따라 처리
        scaled_true_pos = (
            tuple(coord / 100 for coord in true_pos) if scaled else true_pos
        )
        error = np.linalg.norm(np.array(predicted) - np.array(scaled_true_pos))

        print(f"\n--- Test Case {idx+1} ---")
        if scaled:
            print("⚠️  [Scaled] (x, y, z, d) and True Position divided by 100")
        print(f"True Position:      {scaled_true_pos}")
        print(
            f"Predicted Position: ({predicted[0]:.4f}, {predicted[1]:.4f}, {predicted[2]:.4f})"
        )
        print(f"Position Error:     {error:.4f} units")
