import mmh3
import math
import bitarray

class BloomFilter:
    def __init__(self, n_items, fp_rate):
        """
        n_items: expected number of items to store
        fp_rate: desired false positive rate (e.g., 0.01 = 1%)
        """
        self.n = n_items
        self.p = fp_rate

        # Calculate optimal size of bit array (m) and number of hash functions (k)
        # m = -(n * ln(p)) / (ln(2)^2)
        # k = (m/n) * ln(2)
        self.m = int(- (n_items * math.log(fp_rate)) / (math.log(2) ** 2))
        self.k = int((self.m / n_items) * math.log(2)) or 1  # ensure at least 1 hash

        # Initialize bit array
        self.bit_array = bitarray.bitarray(self.m)
        self.bit_array.setall(0)

    def _hashes(self, item):
        """Generate k different hash values for the item"""
        # Use mmh3 with different seeds
        for i in range(self.k):
            yield mmh3.hash(item, i) % self.m

    def add(self, item):
        """Add item to Bloom filter"""
        for h in self._hashes(item):
            self.bit_array[h] = 1

    def check(self, item):
        """Check if item might be in Bloom filter (True = possibly, False = definitely not)"""
        return all(self.bit_array[h] for h in self._hashes(item))


# ✅ Test
bf = BloomFilter(1000, 0.01)
bf.add("doc1")
assert bf.check("doc1") == True
assert bf.check("something_else") in [True, False]  # may be false positive
print("✅ Test passed")
