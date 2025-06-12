import random
import math
import numpy as np


def generate_point_within_range(
    one_side_length, knee_to_eyes, min_distance, max_distance, max_attempts=100000
):
    fixed_points = {
        "origin": [0, 0, 0],
        "origin_right": [one_side_length, 0, -knee_to_eyes],
        "origin_cross_one": [one_side_length, one_side_length, -knee_to_eyes],
        "origin_cross_two": [one_side_length, one_side_length * 2, 0],
    }

    for _ in range(max_attempts):
        A = [
            random.uniform(-2 * one_side_length, 2 * one_side_length),
            random.uniform(-2 * one_side_length, 2 * one_side_length),
            random.uniform(-2 * one_side_length, 2 * one_side_length),
        ]

        distances = {}
        for name, point in fixed_points.items():
            dist = math.sqrt(
                (A[0] - point[0]) ** 2 + (A[1] - point[1]) ** 2 + (A[2] - point[2]) ** 2
            )
            distances[name] = dist

        avg_distance = np.mean(list(distances.values()))
        if min_distance < avg_distance < max_distance:
            return {"A": A, "distances": distances}

    raise ValueError(
        f"Failed to generate point within {min_distance}-{max_distance} after {max_attempts} attempts."
    )


result = generate_point_within_range(
    one_side_length=100, knee_to_eyes=100, min_distance=100, max_distance=300
)
print(f"Generated Point A: {result['A']}")
for name, dist in result["distances"].items():
    print(f"Distance to {name}: {dist:.2f}")
