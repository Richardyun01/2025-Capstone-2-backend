import random
import math


def generate_point_within_range(
    arm_length, min_distance, max_distance, max_attempts=100000
):
    fixed_points = {
        "up": [0, 0, arm_length],
        "down": [0, 0, -arm_length],
        "front": [0, arm_length, 0],
        "left": [-arm_length, 0, 0],
    }

    for _ in range(max_attempts):
        A = [
            random.uniform(-2 * arm_length, 2 * arm_length),
            random.uniform(-2 * arm_length, 2 * arm_length),
            random.uniform(-2 * arm_length, 2 * arm_length),
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


result = generate_point_within_range(arm_length=65, min_distance=50, max_distance=150)
print(f"Generated Point A: {result['A']}")
for name, dist in result["distances"].items():
    print(f"Distance to {name}: {dist:.2f}")
