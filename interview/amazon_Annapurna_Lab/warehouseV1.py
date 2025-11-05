import random
from typing import Dict, List, Optional


class Locker:
    """
    Represents a single locker with a unique ID and a fixed size.
    It tracks whether it is occupied.
    """

    def __init__(self, locker_id: int, size: int):
        self.locker_id = locker_id
        self.size = size
        self.is_occupied = False

    def __repr__(self):
        return f"Locker(id={self.locker_id}, size={self.size}, occupied={self.is_occupied})"


class Warehouse:
    """
    Manages the storage and retrieval of products in lockers.
    """

    def __init__(self, lockers: List[Locker]):
        # Store lockers in a dictionary for quick O(1) lookup by ID.
        self._lockers: Dict[int, Locker] = {locker.locker_id: locker for locker in lockers}

        # A list of Locker objects, sorted by size for the best-fit algorithm.
        self._sorted_lockers = sorted(list(self._lockers.values()), key=lambda l: l.size)

        # A hash map to quickly find a product's locker.
        self._product_to_locker: Dict[int, int] = {}

    def put(self, product_id: int) -> Optional[int]:
        """
        Stores a product in the smallest available locker that can fit it.
        This is a best-fit algorithm.

        Args:
            product_id: The unique ID of the product.

        Returns:
            The ID of the assigned locker, or None if no suitable locker is found.
        """
        product_size = get_product_size(product_id)

        # Find the best-fit locker (first one in the sorted list that fits).
        for locker in self._sorted_lockers:
            if not locker.is_occupied and locker.size >= product_size:
                locker.is_occupied = True
                self._product_to_locker[product_id] = locker.locker_id
                print(
                    f"Product {product_id} (size {product_size}) placed in locker {locker.locker_id} (size {locker.size}).")
                return locker.locker_id

        print(f"Failed to place product {product_id} (size {product_size}). No suitable locker found.")
        return None

    def get(self, product_id: int) -> Optional[int]:
        """
        Retrieves a product and frees its associated locker.

        Args:
            product_id: The unique ID of the product.

        Returns:
            The ID of the retrieved locker, or None if the product is not found.
        """
        locker_id = self._product_to_locker.get(product_id)

        if locker_id is not None:
            locker = self._lockers.get(locker_id)
            if locker:
                locker.is_occupied = False
                del self._product_to_locker[product_id]
                print(f"Product {product_id} retrieved from and locker {locker_id} freed.")
                return locker_id

        print(f"Failed to retrieve product {product_id}. Product not found in the warehouse.")
        return None

    def add_lockers(self, new_lockers: List[Locker]):
        """
        Adds new lockers to the warehouse and re-sorts the list.
        """
        print(f"\nUpgrading warehouse with {len(new_lockers)} new lockers...")
        for locker in new_lockers:
            self._lockers[locker.locker_id] = locker

        # Re-sort the list of locker objects to maintain the best-fit algorithm's efficiency.
        self._sorted_lockers = sorted(list(self._lockers.values()), key=lambda l: l.size)
        print("Warehouse upgrade complete. Lockers have been added and sorted.")

    def __repr__(self):
        occupied_count = sum(1 for locker in self._lockers.values() if locker.is_occupied)
        return f"Warehouse(total_lockers={len(self._lockers)}, occupied={occupied_count})"


def get_product_size(product_id: int) -> int:
    """
    Simulates a function that returns a product's size.
    In a real system, this would query a database.
    """
    # A simple mapping for demonstration purposes
    sizes = {
        101: 5,
        102: 12,
        103: 8,
        104: 25,
        105: 15,
        106: 40,
        107: 60,
        108: 7,
    }
    return sizes.get(product_id, random.randint(1, 100))


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
    warehouse.add_lockers(new_lockers)
    print("Updated Warehouse State:", warehouse)

    # Try putting more products, including a very large one
    print("\n--- Puts after Upgrade ---")
    warehouse.put(105)  # This time, it finds a suitable locker (ID 6 or 7)
    warehouse.put(106)  # size 40 -> locker 2
    warehouse.put(107)  # size 60 -> locker 8 (the extra-large one)
    warehouse.put(108)  # size 7 -> locker 1 (it was freed earlier)
    print("Final Warehouse State:", warehouse)
