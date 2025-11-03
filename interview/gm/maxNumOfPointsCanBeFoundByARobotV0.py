import math
from typing import List, Tuple


def find_max_visible_points(points: List[Tuple[int, int]], fov_degrees: int) -> int:
    """
    Finds the maximum number of points a robot at the origin can see
    within a specified field of view.

    Args:
        points: A list of 2D coordinates (x, y).
        fov_degrees: The field of view in degrees.

    Returns:
        The maximum number of points that can be seen simultaneously.
    """
    if not points:
        return 0

    # 1. Convert points to angles in degrees.
    angles = []
    # Count points at the origin separately, as their angle is undefined.
    points_at_origin = 0
    for x, y in points:
        if x == 0 and y == 0:
            points_at_origin += 1
            continue
        angle = math.degrees(math.atan2(y, x))
        angles.append(angle)

    if not angles:
        return points_at_origin

    # 2. Handle the 360-degree wraparound by duplicating angles.
    # This allows the sliding window to work seamlessly.
    angles.sort()
    angles_extended = angles + [angle + 360 for angle in angles]

    max_count = 0
    left = 0

    # 3. Use a sliding window to find the maximum count.
    for right in range(len(angles_extended)):
        # Shrink the window from the left if it exceeds the FOV.
        while angles_extended[right] - angles_extended[left] > fov_degrees:
            left += 1

        # Update the maximum count with the current window size.
        current_count = right - left + 1
        max_count = max(max_count, current_count)

    return max_count + points_at_origin


if __name__ == "__main__":
    # Example 1: User's example
    example_points_1 = [(100, 1), (100, -1), (-1, 0)]
    example_fov_1 = 10

    max_visible_1 = find_max_visible_points(example_points_1, example_fov_1)
    print(f"Points: {example_points_1}")
    print(f"FOV: {example_fov_1}°")
    print(f"Maximum visible points: {max_visible_1}")
    print("-" * 40)

    # Example 2: Demonstrating wraparound and multiple points in the same spot
    example_points_2 = [(1, 1), (1, 1), (1, -1), (1, -1), (0, 0)]
    example_fov_2 = 100

    max_visible_2 = find_max_visible_points(example_points_2, example_fov_2)
    print(f"Points: {example_points_2}")
    print(f"FOV: {example_fov_2}°")
    print(f"Maximum visible points: {max_visible_2}")
