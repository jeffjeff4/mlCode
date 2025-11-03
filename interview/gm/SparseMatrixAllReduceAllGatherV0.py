# you can write to stdout for debugging purposes, e.g.
# Sample sparse tensors
sparse_matrix_1 = [
    [0, 0, 0, 4, 1.3, 0, 0],
    [0.8, 0, 0, 0, 0, 0, 0],
]
sparse_matrix_2 = [
    [0, 0, 7, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
]


from typing import Tuple, List, Dict, Set


class SparseMatrix:

    def __init__(self, shape: Tuple[int, int]):
        # sanity check
        if not (
            isinstance(shape, Tuple)
            and len(shape) == 2
            and shape[0] >= 0
            and shape[1] >= 0
        ):
            raise ValueError("shape input is not correct")

        self.shape = shape
        self.data: Dict[Tuple[int, int], float] = {}

    def clone(self) -> "SparseMatrix":
        new_matrix = SparseMatrix(self.shape)
        new_matrix.data = self.data.copy()
        return new_matrix

    def __setitem__(self, key: Tuple[int, int], value: float):
        if value != 0.0:
            self.data[key] = float(value)
        elif key in self.data:
            del self.data[key]

    def __add__(self, other: "SparseMatrix") -> "SparseMatrix":
        if self.shape != other.shape:
            raise ValueError("matrices shape should be the same")

        result = SparseMatrix(self.shape)
        all_keys: Set[Tuple[int, int]] = set(self.data.keys()) | set(other.data.keys())

        for key in all_keys:
            sum_val = self.data.get(key, 0.0) + other.data.get(key, 0.0)
            if sum_val != 0.0:
                result.data[key] = sum_val

        return result


class TrainingNode:
    _next_id = 0

    def __init__(self, matrix: SparseMatrix):
        self.node_id = TrainingNode._next_id
        TrainingNode._next_id += 1
        self.matrix = matrix

    def AllGather(self, peers: List["TrainingNode"]) -> SparseMatrix:
        all_nodes = [self] + peers

        if not all_nodes:
            return SparseMatrix((0, 0))

        num_cols = self.matrix.shape[1]
        if any(node.matrix.shape[1] != num_cols for node in all_nodes):
            raise ValueError("all matrices should have same # of cols")

        total_rows = sum(node.matrix.shape[0] for node in all_nodes)
        if total_rows == 0:
            return SparseMatrix((0, 0))

        gatherd_matrix = SparseMatrix((total_rows, num_cols))
        curr_row = 0
        for node in all_nodes:
            for (r, c), value in node.matrix.data.items():
                gatherd_matrix.data[(curr_row + r, c)] = value
            curr_row += node.matrix.shape[0]

        return gatherd_matrix

    def AllReduce(self, peers: List["TrainingNode"], avg: bool = False) -> SparseMatrix:
        all_nodes = [self] + peers

        if not all_nodes:
            return SparseMatrix((0, 0))

        if any(node.matrix.shape != self.matrix.shape for node in peers):
            raise ValueError("all matrices shape supposed to be the same")

        reduced_matrix = self.matrix.clone()

        for peer in peers:
            reduced_matrix = reduced_matrix + peer.matrix

        if avg == True:
            num_nodes = len(all_nodes)
            if num_nodes > 0:
                reduced_matrix = reduced_matrix * 1.0 / num_nodes

        return reduced_matrix


sparse_matrix_1 = [
    [0, 0, 0, 4, 1.3, 0, 0],
    [0.8, 0, 0, 0, 1, 0, 0],
]
sparse_matrix_2 = [
    [0, 0, 7, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
]


def makeRealSparseMatrix(sparse_matrix):
    num_row = len(sparse_matrix)
    num_col = len(sparse_matrix[0])
    sparse_matrix_real = SparseMatrix((num_row, num_col))
    for idx0 in range(num_row):
        for idx1 in range(num_col):
            val = sparse_matrix[idx0][idx1]
            if val == 0:
                continue
            sparse_matrix_real[idx0, idx1] = val

    return sparse_matrix_real


sparse_matrix_1_real = makeRealSparseMatrix(sparse_matrix_1)
sparse_matrix_2_real = makeRealSparseMatrix(sparse_matrix_2)


node1 = TrainingNode(sparse_matrix_1_real)
node2 = TrainingNode(sparse_matrix_2_real)

peers_1 = [node2]
reduced_sum_1 = node1.AllReduce(peers_1)
print("reduced_sum1 = ", reduced_sum_1)

for key, val in reduced_sum_1.data.items():
    print("key = ", key, ", val = ", val)


all_gather_1 = node1.AllGather(peers_1)
print("all_gather_1 = ", all_gather_1)

for key, val in all_gather_1.data.items():
    print("key = ", key, ", val = ", val)
