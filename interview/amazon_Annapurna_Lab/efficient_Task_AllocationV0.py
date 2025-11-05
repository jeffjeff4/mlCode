from collections import Counter
from typing import List, Dict, Tuple, Optional, Any


class TaskAllocator:
    """
    Allocates tasks to a fleet of modules based on type and size.

    The allocation strategy is designed for maximum efficiency:
    1.  It prioritizes tasks with larger model requirements first, as they have
        fewer placement options.
    2.  It assigns each task to the smallest possible module that can handle it
        (a "best-fit" approach) to conserve larger modules for bigger tasks.
    """

    # Defines the hierarchy of sizes. This can be easily expanded.
    # The integer value represents the capability level.
    SIZE_ORDER: Dict[str, int] = {
        "small": 1,
        "medium": 2,
        "large": 3,
    }

    def __init__(self):
        # Create a reverse mapping from level to size name for easy lookup.
        self._order_to_size: Dict[int, str] = {
            v: k for k, v in self.SIZE_ORDER.items()
        }
        # Get a sorted list of size levels in ascending order of capability.
        self._sorted_levels: List[int] = sorted(self._order_to_size.keys())

    def allocate_tasks(
            self,
            tasks: List[Tuple[str, str]],
            modules: Dict[Tuple[str, str], int]
    ) -> Dict[str, Any]:
        """
        Calculates the most efficient allocation of tasks to available modules.

        Args:
            tasks: A list of task tuples, e.g., [('text', 'small'), ...].
            modules: A dictionary mapping module tuples to their available count,
                     e.g., {('text', 'small'): 3, ...}.

        Returns:
            A dictionary containing the results:
            - 'allocated_tasks': A list of tuples, where each tuple contains
              the original task and the module it was assigned to.
            - 'unallocated_tasks': A list of tasks that could not be assigned.
            - 'modules_remaining': The state of the module fleet after allocation.
        """
        # --- 1. Input Validation and Preparation ---
        if not tasks:
            return {
                "allocated_tasks": [],
                "unallocated_tasks": [],
                "modules_remaining": modules,
            }

        # Create a mutable copy of module counts to track availability.
        available_modules = Counter(modules)

        # Sort tasks by required size, largest first (descending).
        # This is the core of the greedy strategy.
        try:
            tasks.sort(key=lambda t: self.SIZE_ORDER.get(t[1], -1), reverse=True)
        except (TypeError, IndexError):
            raise ValueError("Each task in the list must be a (type, size) tuple.")

        allocated_tasks = []
        unallocated_tasks = []

        # --- 2. Allocation Loop ---
        for task in tasks:
            task_type, required_size = task
            required_level = self.SIZE_ORDER.get(required_size)

            if required_level is None:
                # If task has an unrecognized size, it cannot be allocated.
                unallocated_tasks.append(task)
                continue

            best_fit_module = self._find_best_fit_module(task_type, required_level, available_modules)

            if best_fit_module:
                allocated_tasks.append((task, best_fit_module))
                available_modules[best_fit_module] -= 1
            else:
                unallocated_tasks.append(task)

        # --- 3. Return Results ---
        return {
            "allocated_tasks": allocated_tasks,
            "unallocated_tasks": unallocated_tasks,
            "modules_remaining": dict(available_modules),
        }

    def _find_best_fit_module(
            self,
            task_type: str,
            required_level: int,
            available_modules: Counter
    ) -> Optional[Tuple[str, str]]:
        """
        Finds the smallest available module that meets the task's requirements.

        It iterates through module sizes from smallest to largest.

        Returns:
            The best-fit module tuple (type, size) or None if no suitable
            module is available.
        """
        # Iterate through sizes from smallest to largest.
        for level in self._sorted_levels:
            if level >= required_level:
                # This module is capable enough.
                module = (task_type, self._order_to_size[level])
                if available_modules.get(module, 0) > 0:
                    # And it's available. This is our best fit.
                    return module
        return None


# --- Test Infrastructure ---

def run_tests():
    """
    Executes a comprehensive suite of test cases to validate the TaskAllocator.
    """
    allocator = TaskAllocator()

    test_cases = [
        {
            "name": "Standard Case: Efficient allocation",
            "tasks": [('text', 'small'), ('image', 'large'), ('text', 'large'), ('text', 'small')],
            "modules": {('text', 'small'): 1, ('text', 'large'): 1, ('image', 'large'): 1},
            "expected_allocations": 3,
            "expected_unallocated": 1,  # The second ('text', 'small') has no module
            "expected_remaining": {('text', 'small'): 0, ('text', 'large'): 0, ('image', 'large'): 0},
        },
        {
            "name": "Best-Fit Case: Small task uses small module",
            "tasks": [('text', 'small')],
            "modules": {('text', 'small'): 1, ('text', 'large'): 1},
            "expected_allocations": 1,
            "expected_unallocated": 0,
            "expected_remaining": {('text', 'small'): 0, ('text', 'large'): 1},  # Large module is preserved
        },
        {
            "name": "Upgrade Case: Small task uses medium module",
            "tasks": [('text', 'small')],
            "modules": {('text', 'medium'): 1, ('text', 'large'): 1},
            "expected_allocations": 1,
            "expected_unallocated": 0,
            "expected_remaining": {('text', 'medium'): 0, ('text', 'large'): 1},
        },
        {
            "name": "No Exact Match: Large task cannot be fulfilled",
            "tasks": [('image', 'large')],
            "modules": {('image', 'small'): 2, ('image', 'medium'): 1},
            "expected_allocations": 0,
            "expected_unallocated": 1,
            "expected_remaining": {('image', 'small'): 2, ('image', 'medium'): 1},
        },
        {
            "name": "Resource Exhaustion",
            "tasks": [('audio', 'small'), ('audio', 'small'), ('audio', 'small')],
            "modules": {('audio', 'large'): 2},
            "expected_allocations": 2,
            "expected_unallocated": 1,
            "expected_remaining": {('audio', 'large'): 0},
        },
        {
            "name": "Edge Case: Empty tasks list",
            "tasks": [],
            "modules": {('text', 'small'): 5},
            "expected_allocations": 0,
            "expected_unallocated": 0,
            "expected_remaining": {('text', 'small'): 5},
        },
        {
            "name": "Edge Case: Empty modules dictionary",
            "tasks": [('text', 'small')],
            "modules": {},
            "expected_allocations": 0,
            "expected_unallocated": 1,
            "expected_remaining": {},
        },
        {
            "name": "Complex Mix",
            "tasks": [
                ('text', 'medium'), ('image', 'small'), ('text', 'small'),
                ('text', 'large'), ('image', 'medium')
            ],
            "modules": {
                ('text', 'small'): 1, ('text', 'large'): 1,
                ('image', 'small'): 0, ('image', 'medium'): 2
            },
            # Expected logic:
            # 1. ('text', 'large') -> ('text', 'large')
            # 2. ('text', 'medium') -> fails (no medium/large text module left)
            # 3. ('image', 'medium') -> ('image', 'medium')
            # 4. ('image', 'small') -> ('image', 'medium')
            # 5. ('text', 'small') -> ('text', 'small')
            "expected_allocations": 4,
            "expected_unallocated": 1,
            "expected_remaining": {
                ('text', 'small'): 0, ('text', 'large'): 0,
                ('image', 'small'): 0, ('image', 'medium'): 0
            },
        },
    ]

    all_passed = True
    print("--- Running Task Allocation Tests ---")
    for i, test in enumerate(test_cases):
        print(f"\n[{i + 1}/{len(test_cases)}] Testing: {test['name']}...")
        result = allocator.allocate_tasks(test['tasks'], test['modules'])

        try:
            assert len(result['allocated_tasks']) == test['expected_allocations']
            assert len(result['unallocated_tasks']) == test['expected_unallocated']
            # Normalize remaining modules for comparison
            normalized_remaining = {k: v for k, v in result['modules_remaining'].items() if v > 0}
            normalized_expected = {k: v for k, v in test['expected_remaining'].items() if v > 0}
            assert normalized_remaining == normalized_expected
            print(f"✅ PASSED")
        except AssertionError:
            all_passed = False
            print(f"❌ FAILED")
            print(f"  Expected Allocations: {test['expected_allocations']}, Got: {len(result['allocated_tasks'])}")
            print(f"  Expected Unallocated: {test['expected_unallocated']}, Got: {len(result['unallocated_tasks'])}")
            print(f"  Expected Remaining: {test['expected_remaining']}, Got: {result['modules_remaining']}")

    print("\n--- Test Summary ---")
    if all_passed:
        print("🎉 All test cases passed successfully! 🎉")
    else:
        print("🔥 Some tests failed. Please review the output above. 🔥")


if __name__ == '__main__':
    run_tests()

    # --- Complexity Analysis ---
    #
    # T = number of tasks
    # M = number of distinct module types (e.g., text-small, image-large)
    # S = number of size tiers (a small constant, e.g., 3 for small/medium/large)
    #
    # Time Complexity: O(T * log(T) + T * S)
    # - Sorting the tasks dominates the complexity: O(T * log(T)).
    # - The main allocation loop iterates through T tasks.
    # - Inside the loop, `_find_best_fit_module` iterates at most S times.
    # - Therefore, the allocation part is O(T * S).
    # - Total time is O(T * log(T) + T * S). Since S is a small constant,
    #   this simplifies to O(T * log(T)).
    #
    # Space Complexity: O(T + M)
    # - Storing the sorted list of tasks requires O(T) space.
    # - Storing the `available_modules` counter requires O(M) space.
    # - The result dictionaries for allocated/unallocated tasks require O(T) space.
    # - Total space is O(T + M).
