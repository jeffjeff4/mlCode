####Question:
####/*
####* Question:
####* Implement a cyclic ring buffer of size RING_SIZE. Each element in the ring stores
####* an integer. If the ring is full, newer values should overwrite the oldest
####* values.
####* When reading from the ring, the oldest entry is returned (each entry
####* can be read only once).
####*
####* Implement the data struct and 3 functions below:
####*/
####
####// [null, null, null]
####// Insert 10, 20, 30
####// [10, 20, 30]
####// Insert 40, 50
####// [40, 50, 30]
####// Read
####// [40, 50, null]
####// Read
####// [null, 50, null]
####
#####define RING_SIZE 3
####
####// you don’t have to use a struct, it’s just here as an example.
####struct my_ring {
####};
#### 
####void ring_init(struct my_ring *r)
####{ 
####} 
####
####int ring_read(struct my_ring *r)
####{
####    return 0;
####} 
####
####void ring_write(struct my_ring *r, int val)
####{
####}

class CyclicRingBuffer:
    def __init__(self, k):
        self.len = k
        self.head = 0
        self.tail = 0
        self.size = 0
        self.q = [None for _ in range(k)]

    def ring_read(self):
        if self.size == 0:
            return None

        rst = self.q[self.head]
        self.q[self.head] = None
        self.head += 1
        self.head = (self.head) % self.len
        self.size -= 1
        return rst

    def ring_write(self, val):
        if self.size == self.len:
            self.q[self.head] = val
            self.head += 1
            self.head = self.head % self.len

            self.tail += 1
            self.tail = self.tail % self.len

        else:
            self.q[self.tail] = val
            self.tail += 1
            self.tail = self.tail % self.len

            self.size += 1


buf = CyclicRingBuffer(3)
buf.ring_write(10)
buf.ring_write(20)
buf.ring_write(30)
print("buf.q = ", buf.q)

buf.ring_write(40)
print("buf.q = ", buf.q)

buf.ring_write(50)
print("buf.q = ", buf.q)

buf.ring_read()
print("buf.q = ", buf.q)

buf.ring_read()
print("buf.q = ", buf.q)
