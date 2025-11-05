####Question:
####a room, have many seat, they sit in seats, there are some people got viurs, virus can spread, every minutes, virus could spread to other people, this example it takes 4 minutes. given 1) the seats, 2) people's position in the seats, 3) peeple who are sick, questions: how much time virus could spread to everyone in the room
####
####example:
####
####'''
####| o | x |   |
####| x |   |   |
####| x | x | x |
####
####| o | o |   |
####| o |   |   |
####| x | x | x |
####
####| o | o |   |
####| o |   |   |
####| o | x | x |
####
####| o | o |   |
####| o |   |   |
####| o | o | x |
####
####| o | o |   |
####| o |   |   |
####| o | o | o |
####
####
####'''

####o is the people has virus
####x is the people do not have virus
####it takes 4 minutes virus spread to everyone

import collections
from collections import deque
from typing import List, Tuple


class VirusSpreader:
    """
    Simulates the spread of a virus in a room layout using Multi-Source BFS.

    The room is represented by a 2D grid where seats can be:
    - SICK ('o'): A person who is sick and can spread the virus.
    - HEALTHY ('x'): A person who is not sick but can be infected.
    - EMPTY (' '): An empty seat, which blocks the spread.

    The virus spreads to adjacent (up, down, left, right) HEALTHY people
    in 1-minute intervals. All SICK people spread simultaneously.
    """

    # --- Constants for seat states ---
    SICK = 'o'
    HEALTHY = 'x'
    EMPTY = ' '

    # --- Directions for BFS (Up, Down, Left, Right) ---
    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(self, room_layout: List[List[str]]):
        """
        Initializes the simulator with a deep copy of the room layout.

        Args:
            room_layout: A 2D list of strings representing the room.

        Raises:
            ValueError: If the grid is not rectangular (i.e., rows have
                        different lengths).
        """
        if not room_layout or not room_layout[0]:
            self.grid = []
            self.rows = 0
            self.cols = 0
        else:
            self.rows = len(room_layout)
            self.cols = len(room_layout[0])
            # Production-level check: ensure grid is rectangular
            if not all(len(row) == self.cols for row in room_layout):
                raise ValueError("Grid rows must all be the same length.")

            # Create a deep copy to avoid mutating the original input
            self.grid = [row[:] for row in room_layout]

    def _is_valid(self, r: int, c: int) -> bool:
        """Helper to check if a coordinate is within the grid bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def calculate_spread_time(self) -> int:
        """
        Calculates the total time (in minutes) for the virus to infect all
        reachable healthy people.

        This is a Multi-Source BFS problem. We start the BFS from all
        initially SICK people at the same time (time 0).

        Returns:
            int: The number of minutes required to infect all reachable
                 healthy people.
                 - Returns 0 if there are no healthy people to infect.
                 - Returns -1 if there are healthy people but no sick
                   people to start the spread.
                 - Returns -1 if there are healthy people who are
                   unreachable (e.g., blocked by empty seats).
        """
        if self.rows == 0:
            return 0  # An empty room takes 0 minutes

        # Queue for BFS: stores (row, col, time_infected)
        q: deque[Tuple[int, int, int]] = deque()
        healthy_people_count = 0

        # --- Step 1: Initialize Queue and Count Healthy People ---
        # Scan the grid once to find all initial sources (SICK)
        # and targets (HEALTHY).
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == self.HEALTHY:
                    healthy_people_count += 1
                elif self.grid[r][c] == self.SICK:
                    # Add all initial virus sources to the queue with time 0
                    q.append((r, c, 0))

        # --- Step 2: Handle Edge Cases ---
        if healthy_people_count == 0:
            # No healthy people to infect.
            return 0

        if not q:
            # There are healthy people, but no source of infection.
            return -1  # Impossible to spread

        max_minutes = 0  # This will track the time of the *last* infection

        # --- Step 3: Run the Multi-Source BFS ---
        while q:
            r, c, time = q.popleft()

            # Update the max time seen. Since BFS explores level by level,
            # the last 'time' we process will be the final answer.
            max_minutes = time

            # Explore all 4 adjacent neighbors
            for dr, dc in self.DIRECTIONS:
                nr, nc = r + dr, c + dc

                # Check if the neighbor is valid AND is a healthy person
                if self._is_valid(nr, nc) and self.grid[nr][nc] == self.HEALTHY:
                    # Infect the person
                    self.grid[nr][nc] = self.SICK
                    healthy_people_count -= 1

                    # Add the newly infected person to the queue for the *next* minute
                    q.append((nr, nc, time + 1))

        # --- Step 4: Final Check ---
        # If healthy_people_count is 0, everyone was reached.
        # If it's > 0, some healthy people were unreachable.
        return max_minutes if healthy_people_count == 0 else -1


# --- Test Cases ---
if __name__ == "__main__":

    print("--- Running Test Cases ---")


    def run_test(name, layout, expected):
        print(f"\nTest: {name}")
        for row in layout:
            print(f"  {row}")
        try:
            simulator = VirusSpreader(layout)
            result = simulator.calculate_spread_time()
            assert result == expected
            print(f"Result: {result} (Expected: {expected}) -> ✅ PASS")
        except Exception as e:
            print(f"Result: ERROR ({e}) (Expected: {expected}) -> ❌ FAIL")


    # Test 1: Provided Example
    example_layout = [
        ['o', 'x', ' '],
        ['x', ' ', ' '],
        ['x', 'x', 'x']
    ]
    # Path: (0,0,0) -> (0,1,1) & (1,0,1)
    #       (1,0,1) -> (2,0,2)
    #       (2,0,2) -> (2,1,3)
    #       (2,1,3) -> (2,2,4)
    # Max time is 4.
    run_test("Provided Example", example_layout, 4)

    # Test 2: No Healthy People
    no_healthy_layout = [
        ['o', ' ', 'o'],
        [' ', 'o', ' ']
    ]
    run_test("No Healthy People", no_healthy_layout, 0)

    # Test 3: No Sick People (but healthy exist)
    no_sick_layout = [
        ['x', ' ', 'x'],
        [' ', 'x', ' ']
    ]
    run_test("No Sick People", no_sick_layout, -1)

    # Test 4: Unreachable Healthy People
    unreachable_layout = [
        ['o', 'x', ' '],
        [' ', ' ', 'x'],  # This 'x' is unreachable
        ['x', 'x', ' ']
    ]
    run_test("Unreachable Healthy People", unreachable_layout, -1)

    # Test 5: Empty Room
    empty_layout = []
    run_test("Empty Room", empty_layout, 0)

    # Test 6: Empty Room (variant)
    empty_layout_2 = [[]]
    run_test("Empty Room (Variant)", empty_layout_2, 0)

    # Test 7: Multi-Source Spread
    multi_source_layout = [
        ['o', 'x', 'x', 'x', 'o']
    ]
    # (0,0,0) -> (0,1,1)
    # (0,4,0) -> (0,3,1)
    # (0,1,1) -> (0,2,2)
    # (0,3,1) -> (0,2,2) (but (0,2) is already infected)
    # Max time is 2.
    run_test("Multi-Source Spread", multi_source_layout, 2)

    # Test 8: All Healthy, One Sick
    all_healthy_layout = [
        ['x', 'x', 'x'],
        ['x', 'o', 'x'],
        ['x', 'x', 'x']
    ]
    # Max distance from center is 2 (to a corner)
    run_test("All Healthy, Center Sickness", all_healthy_layout, 2)

    # Test 9: A complex maze
    maze_layout = [
        ['o', 'x', 'x', ' ', 'x', 'x'],
        [' ', ' ', 'x', ' ', 'x', ' '],
        ['x', 'x', 'x', 'x', 'x', 'x'],
        ['x', ' ', ' ', ' ', ' ', ' '],
        ['x', 'x', 'x', 'x', 'x', 'o']
    ]
    # Source 1 (0,0) -> (0,1,1) -> (0,2,2) -> (1,2,3) -> (2,2,4) ...
    # Source 2 (4,5) -> (3,5,X) -> (2,5,X) ...
    # Let's trace from (4,5)
    # (4,5,0) -> (4,4,1) -> (4,3,2) -> (4,2,3) -> (4,1,4) -> (4,0,5)
    # (4,1,4) -> (3,0,5) -> (2,0,6)
    # (4,2,3) -> (2,3,4) -> (2,4,5) -> (1,4,6) -> (0,4,7) -> (0,5,8)
    # (2,4,5) -> (2,5,6)
    # (0,4,7) -> (0,3,X) (empty)
    # The farthest node is (0,5) at time 8.
    run_test("Complex Maze", maze_layout, 8)