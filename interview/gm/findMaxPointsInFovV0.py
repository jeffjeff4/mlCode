import math


def find_max_points_in_fov(points, fov_angle):
    """
    Finds the maximum number of points visible from the origin (0,0) within a given field of view.

    The method is as follows:
    1.  Handle points at the origin separately, as their angle is undefined.
    2.  For all other points, convert their (x, y) coordinates to an angle in radians
        using `atan2(y, x)`. This correctly handles all quadrants.
    3.  Sort the angles in ascending order.
    4.  To handle the circular nature of angles (e.g., a view crossing from 355 degrees to 5 degrees),
        we create a temporary extended array of angles. This is done by appending a second
        copy of the sorted angles, each shifted by 2*pi (360 degrees).
    5.  A sliding window (using two pointers, `start` and `end`) moves across this extended
        array to find the window that is no wider than the `fov_angle` and contains the
        most points.
    6.  The final result is the maximum count from the sliding window plus the count of any
        points that were at the origin.

    Args:
        points (list of tuples): A list of (x, y) coordinates for each point.
        fov_angle (float): The robot's field of view in degrees.

    Returns:
        int: The maximum number of points that can be seen.
    """
    # --- Complexity Analysis ---
    # Time Complexity: O(N log N) dominated by the sorting of angles.
    #   - Calculating angles: O(N)
    #   - Sorting angles: O(N log N)
    #   - Sliding window part: O(N), as each pointer traverses the extended array once.
    # Space Complexity: O(N) for storing the angles and the extended circular array.

    if not points:
        return 0
    if fov_angle >= 360:
        return len(points)
    if fov_angle <= 0:
        return 0 if not any(p == (0, 0) for p in points) else sum(1 for p in points if p == (0, 0))

    angles = []
    origin_points = 0

    # 步骤1: 将坐标转换为角度，并处理原点上的点
    # Step 1: Convert coordinates to angles and handle points at the origin
    for x, y in points:
        if x == 0 and y == 0:
            origin_points += 1
            continue
        # math.atan2 returns angle in radians from -pi to pi
        angles.append(math.atan2(y, x))

    # 步骤2: 对角度进行排序
    # Step 2: Sort the angles
    angles.sort()

    # 如果除了原点没有其他点，直接返回
    # If there are no points other than the origin, return
    if not angles:
        return origin_points

    # 步骤3: 为了处理环形问题（例如从350度到10度），我们将数组扩展一倍
    # Step 3: To handle the circular nature (e.g., from 350 to 10 degrees), we extend the array
    angles_circular = angles + [angle + 2 * math.pi for angle in angles]

    fov_rad = math.radians(fov_angle)
    max_in_fov = 0
    start = 0

    # 步骤4: 使用滑动窗口寻找最大点数
    # Step 4: Use a sliding window to find the max points
    for end in range(len(angles_circular)):
        # 当窗口角度差大于视场角时，移动start指针
        # While the window's angular difference is greater than the FOV, move the start pointer
        while angles_circular[end] - angles_circular[start] > fov_rad:
            start += 1

        # 更新当前窗口内的最大点数
        # Update the max points found in the current window
        max_in_fov = max(max_in_fov, end - start + 1)

    # 最终结果是窗口中的最大点数加上原点上的点数
    # The final result is the max from the window plus the points at the origin
    return max_in_fov + origin_points


# --- Test Cases ---
def run_tests():
    """Function to run all test cases."""
    print("--- Running Field of View Tests ---")

    # Test Case 1: Basic scenario, no wrap-around
    points1 = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]  # Angles: 45, 0, -45, -90, -135 degrees
    fov1 = 90
    # Expected: (1,1), (1,0), (1,-1) are within a 90-degree FOV.
    assert find_max_points_in_fov(points1, fov1) == 3
    print("Test Case 1 (Basic) PASSED")

    # Test Case 2: Wrap-around scenario (crossing the 0/360 degree line)
    points2 = [(1, 0.1), (1, -0.1), (-1, 0.1), (1, 0)]  # Angles: ~5.7, ~-5.7, ~174.3, 0 degrees
    fov2 = 20
    # Expected: (1, 0.1), (1, -0.1), (1,0) are within a ~11.4 degree span around 0.
    assert find_max_points_in_fov(points2, fov2) == 3
    print("Test Case 2 (Wrap-around) PASSED")

    # Test Case 3: Points at the origin
    points3 = [(1, 1), (-1, -1), (0, 0), (0, 0), (1, 0)]
    fov3 = 45
    # Expected: (1,1) and (1,0) are just outside 45 degrees. Max is 1 in FOV. Plus 2 at origin.
    # Let's check: angle(1,1)=45, angle(1,0)=0. Diff is 45. So they are in.
    # So 2 points in FOV + 2 points at origin = 4
    assert find_max_points_in_fov(points3, fov3) == 4
    print("Test Case 3 (With Origin Points) PASSED")

    # Test Case 4: Full 360-degree field of view
    points4 = [(1, 5), (-3, 4), (5, -12), (0, 0)]
    fov4 = 360
    assert find_max_points_in_fov(points4, fov4) == 4
    print("Test Case 4 (360-degree FOV) PASSED")

    # Test Case 5: Empty and zero FOV
    points5 = [(1, 1), (-1, 1)]
    assert find_max_points_in_fov([], 90) == 0
    assert find_max_points_in_fov(points5, 0) == 0
    assert find_max_points_in_fov([(0, 0), (0, 0)], 0) == 2
    print("Test Case 5 (Empty/Zero FOV) PASSED")

    # Test Case 6: All points in a line
    points6 = [(1, 1), (2, 2), (3, 3), (-1, -1), (-2, -2)]
    fov6 = 10
    # Expected: All points have an angle of 45 or -135 degrees.
    # Max in one direction is 3.
    assert find_max_points_in_fov(points6, fov6) == 3
    print("Test Case 6 (Collinear Points) PASSED")

    # Test Case 7: Larger random case
    points7 = [
        (10, 1),  # ~5.7 deg
        (10, 2),  # ~11.3 deg
        (-1, 10),  # ~95.7 deg
        (-2, 10),  # ~101.3 deg
        (-3, 10),  # ~106.7 deg
        (1, -10),  # ~-84.3 deg
        (10, -1)  # ~-5.7 deg
    ]
    fov7 = 15
    # Expected: The three points near 100 deg are within 15 deg span.
    # Also (10,1), (10,2) and (10, -1) should be checked carefully.
    # angle(10, -1) is -0.099 rad, angle(10, 1) is 0.099 rad, angle(10, 2) is 0.197 rad.
    # Span from -0.099 to 0.197 is 0.296 rad.
    # fov_rad = 15 deg = 0.261 rad.
    # So, (10,-1) and (10,1) are in, but (10,2) is not. Max is 2.
    # BUT, the three points around 100 degrees are: 1.76, 1.78, 1.86 rad.
    # Span from 1.76 to 1.86 is 0.1 rad, which is < 0.261 rad. So, we can see 3.
    assert find_max_points_in_fov(points7, fov7) == 3
    print("Test Case 7 (Larger Random Case) PASSED")

    print("\n--- All Tests Completed Successfully ---")


if __name__ == '__main__':
    run_tests()
