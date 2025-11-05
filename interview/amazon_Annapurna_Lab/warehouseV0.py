####python code, with test case
####1. MUST be CORRECT
####2. be the efficient code solution in terms of time complexity and space complexity
####3. make it runnable in this chat session
####4. provide time complexity and space complexity analysis
####5. code clearance, class design needs to be expandable, code organization, make it more like production level code
####6. be the efficient code solution in terms of time complexity and space complexity
####
####Question:
####Design a warehouse system, it should have 2 methods:
####put(product_id) -> locker_id
####get(product_id) -> locker_id
####Where each product has its own size, you’re given a function to return product size from its id. Lockers have their own sizes as well, products can only be stored in the locker not smaller than itself.
####Follow up questions, when we upgrade the warehouse, adding more lockers, what changes you need to make? What if we want to add extra-large lockers?

import heapq
import bisect

# --- Product Size Definition ---
# In a real-world scenario, this might be a database call or an API request.
# For this example, we'll use a simple dictionary lookup.
PRODUCT_SIZES = {
    101: 5,  # Product 101 has size 5
    102: 12,  # Product 102 has size 12
    103: 25,  # Product 103 has size 25
    104: 50,  # Product 104 has size 50
    105: 95,  # Product 105 has size 95
    106: 40,
    107: 60,
    108: 7,
}



def get_product_size(product_id: int) -> int:
    """
    Returns the size of a given product.
    Args:
        product_id: The unique identifier for the product.
    Returns:
        The size of the product.
    Raises:
        ValueError: If the product_id is not found.
    """
    if product_id not in PRODUCT_SIZES:
        raise ValueError(f"Product with ID {product_id} not found.")
    return PRODUCT_SIZES[product_id]


# --- Locker Definition ---
class Locker:
    """Represents a single locker in the warehouse."""

    def __init__(self, locker_id: int, size: int):
        if not isinstance(locker_id, int) or not isinstance(size, int):
            raise TypeError("Locker ID and size must be integers.")
        if size <= 0:
            raise ValueError("Locker size must be positive.")

        self.locker_id = locker_id
        self.size = size

    def __repr__(self) -> str:
        return f"Locker(id={self.locker_id}, size={self.size})"


# --- Warehouse System Design ---
class Warehouse:
    """
    Manages storage of products in lockers of various sizes.

    This system uses a sorted list for available lockers to efficiently find the
    best-fit locker for a product. This provides a good balance between
    performance and implementation complexity.
    """

    def __init__(self, lockers: list[Locker]):
        """
        Initializes the warehouse with a list of lockers.
        Args:
            lockers: A list of Locker objects to be managed by the warehouse.
        """
        # Store all locker objects for easy lookup by ID.
        self._all_lockers = {locker.locker_id: locker for locker in lockers}

        # Use a sorted list of (size, locker_id) tuples for available lockers.
        # This allows for efficient searching using binary search (bisect_left).
        self._available_lockers = sorted([(locker.size, locker.locker_id) for locker in lockers])

        # Maps product_id to the locker_id where it is stored. O(1) lookup.
        self._product_location = {}

    def put(self, product_id: int) -> int | None:
        """
        Stores a product in the smallest available locker that can accommodate it.

        Time Complexity: O(log N + N) -> O(N)
           - O(log N) to find the best-fit locker using binary search (bisect_left).
           - O(N) in the worst case for removing the locker from the sorted list.
        Space Complexity: O(1) (excluding storage for internal state)

        Args:
            product_id: The ID of the product to store.

        Returns:
            The locker_id where the product was placed, or None if no suitable
            locker is available.
        """
        if product_id in self._product_location:
            print(f"Product {product_id} is already in the warehouse.")
            return self._product_location[product_id]

        product_size = get_product_size(product_id)

        # Find the insertion point for a locker of the product's size.
        # This gives us the index of the first locker that is large enough.
        index = bisect.bisect_left(self._available_lockers, (product_size, 0))

        if index == len(self._available_lockers):
            # No locker is large enough
            return None

        # The locker at 'index' is the best fit (smallest available that's large enough)
        size, locker_id = self._available_lockers.pop(index)

        # Store the product location
        self._product_location[product_id] = locker_id

        return locker_id

    def get(self, product_id: int) -> int | None:
        """
        Retrieves a product, freeing up its locker.

        Time Complexity: O(log N)
            - O(1) for dictionary lookup.
            - O(log N) to re-insert the locker into the sorted list (using bisect.insort).
        Space Complexity: O(1)

        Args:
            product_id: The ID of the product to retrieve.

        Returns:
            The locker_id where the product was stored, or None if the product
            is not in the warehouse.
        """
        if product_id not in self._product_location:
            return None

        locker_id = self._product_location.pop(product_id)
        locker = self._all_lockers[locker_id]

        # Add the locker back to the available pool, maintaining sorted order.
        bisect.insort(self._available_lockers, (locker.size, locker.locker_id))

        return locker_id

    def add_lockers(self, locker: Locker):
        """
        Adds a new locker to the warehouse system.
        This allows for dynamic expansion of the warehouse.

        Time Complexity: O(log N)
           - bisect.insort efficiently finds the correct position and inserts.
        """
        if locker.locker_id in self._all_lockers:
            raise ValueError(f"Locker with ID {locker.locker_id} already exists.")

        self._all_lockers[locker.locker_id] = locker
        bisect.insort(self._available_lockers, (locker.size, locker.locker_id))
        print(f"\nAdded new {locker} to the warehouse.")

    def get_status(self):
        """Helper method to print the current state of the warehouse."""
        print("\n--- Warehouse Status ---")
        print(f"Available Lockers: {self._available_lockers}")
        print(f"Product Locations: {self._product_location}")
        print("------------------------")


if __name__ == "__main__":
    # Initialize the warehouse with some existing lockers
    initial_lockers = [
        Locker(1, 10), Locker(2, 50), Locker(3, 20), Locker(4, 30), Locker(5, 10)
    ]
    warehouse = Warehouse(initial_lockers)
    print("Initial Warehouse State:", warehouse)

    # --- Demonstrate putting products ---
    print("\n--- Puts ---")
    warehouse.put(101)  # size 5 -> locker 1
    warehouse.put(102)  # size 12 -> locker 3
    warehouse.put(103)  # size 8 -> locker 5
    warehouse.put(104)  # size 25 -> locker 4
    warehouse.put(105)  # size 15 -> no locker available smaller than 20
    print("Warehouse State after Puts:", warehouse)

    # --- Demonstrate getting products ---
    print("\n--- Gets ---")
    warehouse.get(102)
    warehouse.get(105)  # This product was never stored, so it fails gracefully.
    print("Warehouse State after Gets:", warehouse)

    # --- Demonstrate the upgrade scenario ---
    # Upgrade by adding new lockers, including a large one
    new_lockers = [
        Locker(6, 12),
        Locker(7, 20),
        Locker(8, 100)  # An extra-large locker
    ]
    for new_locker in new_lockers:
        warehouse.add_lockers(new_locker)
    print("Updated Warehouse State:", warehouse)

    # Try putting more products, including a very large one
    print("\n--- Puts after Upgrade ---")
    warehouse.put(105)  # This time, it finds a suitable locker (ID 6 or 7)
    warehouse.put(106)  # size 40 -> locker 2
    warehouse.put(107)  # size 60 -> locker 8 (the extra-large one)
    warehouse.put(108)  # size 7 -> locker 1 (it was freed earlier)
    print("Final Warehouse State:", warehouse)
