import collections


class SparseMatrix:
    """
    A class to represent a sparse matrix using a dictionary to store non-zero elements.
    The keys of the dictionary are tuples (row, col) representing the coordinates.
    """

    def __init__(self, rows, cols):
        """
        Initializes a sparse matrix of size rows x cols.
        All elements are initially considered to be zero.

        Args:
            rows (int): The number of rows in the matrix.
            cols (int): The number of columns in the matrix.
        """
        if rows <= 0 or cols <= 0:
            raise ValueError("Matrix dimensions must be positive.")
        self.rows = rows
        self.cols = cols
        self._data = {}  # Dictionary to store non-zero elements: {(row, col): value}

    def _check_bounds(self, row, col):
        """Helper function to check if the given coordinates are within the matrix bounds."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise IndexError("Matrix indices out of bounds.")

    def set(self, row, col, value):
        """
        Sets the value of the element at (row, col).
        If the value is 0, the element is removed from the storage to maintain sparsity.

        Time Complexity: O(1) on average (dictionary access)
        Space Complexity: O(1)

        Args:
            row (int): The row index.
            col (int): The column index.
            value (int or float): The value to set.
        """
        self._check_bounds(row, col)
        if value != 0:
            self._data[(row, col)] = value
        elif (row, col) in self._data:
            # If the new value is 0 and the element exists, remove it
            del self._data[(row, col)]

    def get(self, row, col):
        """
        Gets the value of the element at (row, col).
        Returns 0 for elements not explicitly set.

        Time Complexity: O(1) on average (dictionary access)
        Space Complexity: O(1)

        Args:
            row (int): The row index.
            col (int): The column index.

        Returns:
            The value at (row, col), which is 0 if the element is not stored.
        """
        self._check_bounds(row, col)
        return self._data.get((row, col), 0)

    def add(self, other):
        """
        Adds another SparseMatrix to this one.
        The two matrices must have the same dimensions.

        Time Complexity: O(nnz1 + nnz2), where nnz1 and nnz2 are the number
                         of non-zero elements in the respective matrices.
        Space Complexity: O(nnz_result), where nnz_result is the number of
                          non-zero elements in the resulting matrix.

        Args:
            other (SparseMatrix): The matrix to add.

        Returns:
            SparseMatrix: A new matrix representing the sum.
        """
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions for addition.")

        result = SparseMatrix(self.rows, self.cols)

        # Add values from the first matrix
        for (r, c), val in self._data.items():
            result.set(r, c, self.get(r, c) + other.get(r, c))

        # Add values from the second matrix that were not in the first
        for (r, c), val in other._data.items():
            if (r, c) not in self._data:
                result.set(r, c, self.get(r, c) + other.get(r, c))

        return result

    def multiply(self, other):
        """
        Multiplies this matrix by another SparseMatrix (self * other).
        The number of columns in the first matrix must equal the number of
        rows in the second matrix.

        Time Complexity: O(nnz1 + nnz2 + FLOPs), where FLOPs is the number of
                         scalar multiplications needed. A loose upper bound is
                         O(nnz1 * avg_nnz_per_row_in_other).
        Space Complexity: O(nnz_result) for the result matrix.

        Args:
            other (SparseMatrix): The matrix to multiply by.

        Returns:
            SparseMatrix: A new matrix representing the product.
        """
        if self.cols != other.rows:
            raise ValueError(
                "Number of columns of the first matrix must equal the number of rows of the second."
            )

        result = SparseMatrix(self.rows, other.cols)

        # An efficient approach is to iterate through the non-zero elements of `self`
        # and for each element `(r1, c1)`, multiply it with all non-zero elements
        # in row `c1` of `other`.

        # Pre-process `other` to group its elements by row for faster access.
        other_rows = collections.defaultdict(list)
        for (r, c), val in other._data.items():
            other_rows[r].append((c, val))

        # Perform the multiplication
        for (r1, c1), val1 in self._data.items():
            # For each non-zero element in self at (r1, c1)
            # Find corresponding elements in other's row c1
            if c1 in other_rows:
                for c2, val2 in other_rows[c1]:
                    # The product val1 * val2 contributes to result at (r1, c2)
                    current_val = result.get(r1, c2)
                    result.set(r1, c2, current_val + val1 * val2)
        return result

    def __str__(self):
        """
        Returns a string representation of the matrix, printing it in a dense format.
        """
        matrix_str = ""
        for r in range(self.rows):
            row_str = " ".join(str(self.get(r, c)) for c in range(self.cols))
            matrix_str += f"[{row_str}]\n"
        return matrix_str


# --- Test Cases ---
def run_tests():
    """Function to run all test cases."""
    print("--- Running Sparse Matrix Tests ---")

    # 1. Test Initialization and Basic Set/Get
    print("\n1. Test: Initialization and Set/Get")
    m1 = SparseMatrix(3, 4)
    m1.set(0, 1, 5)
    m1.set(2, 2, 9)
    print("Matrix m1 (3x4):")
    print(m1)
    assert m1.get(0, 1) == 5
    assert m1.get(2, 2) == 9
    assert m1.get(1, 1) == 0
    print("Set/Get PASSED")

    # 2. Test Setting to Zero (maintaining sparsity)
    print("\n2. Test: Setting element to zero")
    m1.set(0, 1, 0)
    print("Matrix m1 after setting (0,1) to 0:")
    print(m1)
    assert m1.get(0, 1) == 0
    assert (0, 1) not in m1._data
    print("Set to zero PASSED")

    # 3. Test Edge Cases (Bounds)
    print("\n3. Test: Boundary checks")
    try:
        m1.get(3, 0)
        assert False, "get() should have raised IndexError"
    except IndexError:
        print("- get() out of bounds check PASSED")
    try:
        m1.set(0, 4, 10)
        assert False, "set() should have raised IndexError"
    except IndexError:
        print("- set() out of bounds check PASSED")
    try:
        SparseMatrix(0, 5)
        assert False, "Constructor should have raised ValueError for 0 rows"
    except ValueError:
        print("- Constructor invalid dimension check PASSED")

    # 4. Test Addition
    print("\n4. Test: Addition")
    m2 = SparseMatrix(3, 4)
    m2.set(0, 0, 1)
    m2.set(2, 2, -3)
    m2.set(1, 3, 7)

    m1.set(0, 1, 5)  # reset m1
    m1.set(2, 2, 9)

    print("Matrix m1:")
    print(m1)
    print("Matrix m2:")
    print(m2)

    m3 = m1.add(m2)
    print("Result of m1 + m2:")
    print(m3)
    assert m3.get(0, 0) == 1
    assert m3.get(0, 1) == 5
    assert m3.get(2, 2) == 6  # 9 + (-3)
    assert m3.get(1, 3) == 7
    assert m3.get(1, 1) == 0
    print("Addition PASSED")

    # Addition dimension mismatch
    m_dim_mismatch = SparseMatrix(4, 3)
    try:
        m1.add(m_dim_mismatch)
        assert False, "add() should raise ValueError for dimension mismatch"
    except ValueError:
        print("Addition dimension mismatch check PASSED")

    # 5. Test Multiplication
    print("\n5. Test: Multiplication")
    # A = 2x3 matrix
    A = SparseMatrix(2, 3)
    A.set(0, 0, 1)
    A.set(0, 1, 2)
    A.set(1, 1, 3)
    A.set(1, 2, 4)

    # B = 3x2 matrix
    B = SparseMatrix(3, 2)
    B.set(0, 0, 5)
    B.set(0, 1, 6)
    B.set(1, 1, 7)
    B.set(2, 0, 8)

    print("Matrix A (2x3):")
    print(A)
    print("Matrix B (3x2):")
    print(B)

    # Expected Result C (2x2)
    # C[0,0] = 1*5 + 2*0 + 0*8 = 5
    # C[0,1] = 1*6 + 2*7 + 0*0 = 20
    # C[1,0] = 0*5 + 3*0 + 4*8 = 32
    # C[1,1] = 0*6 + 3*7 + 4*0 = 21
    C = A.multiply(B)
    print("Result of A * B (2x2):")
    print(C)

    assert C.rows == 2 and C.cols == 2
    assert C.get(0, 0) == 5
    assert C.get(0, 1) == 20
    assert C.get(1, 0) == 32
    assert C.get(1, 1) == 21
    print("Multiplication PASSED")

    # Multiplication dimension mismatch
    try:
        A.multiply(m1)  # A is 2x3, m1 is 3x4 -> This should work
        C_ok = A.multiply(m1)
        print("A(2x3) * m1(3x4) should be valid. PASSED")
        assert C_ok.rows == 2 and C_ok.cols == 4
    except ValueError:
        assert False, "Multiplication with valid dimensions failed"

    try:
        B.multiply(A)  # B is 3x2, A is 2x3 -> This should work
        C_ok2 = B.multiply(A)
        print("B(3x2) * A(2x3) should be valid. PASSED")
        assert C_ok2.rows == 3 and C_ok2.cols == 3
    except ValueError:
        assert False, "Multiplication with valid dimensions failed"

    try:
        A.multiply(A)  # A is 2x3, A is 2x3 -> Mismatch
        assert False, "multiply() should raise ValueError for dimension mismatch"
    except ValueError:
        print("Multiplication dimension mismatch check PASSED")

    print("\n--- All Tests Completed Successfully ---")


if __name__ == '__main__':
    run_tests()
