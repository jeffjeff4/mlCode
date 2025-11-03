####https://www.1point3acres.com/bbs/thread-1137431-1-1.html
####
####Write a 2d convolution in numpy. stride = 1. 给的例子是4x4 image 和3x3 filter。
####import numpy as np
##### Define the 4x4 input matrix A
####A = np.array([[1, 2, 3, 4],
####    [5, 6, 7, 8],
####    [9, 10, 11, 12],
####    [13, 14, 15, 16]])
##### Define the 3x3 filter K
####K = np.array([[1, 0, -1],
####    [1, 0, -1],
####    [1, 0, -1]])
####难点是用numpy写【之前没准备，忘记语法是 shape 还是 size，还有array slicing等等】。
####里面的1d conv我用两层for loop，她说能不能用array slicing，vector等等。用for loop手写太繁琐了，index很容易搞错。


import numpy as np
from numpy.lib.stride_tricks import as_strided
import time

def conv2d_numpy(image, kernel):
    """
    Performs 2D convolution (mode='valid') with stride 1
    using NumPy's as_strided.

    This function is highly efficient as it creates a "view" of the
    image with the necessary overlapping patches, then performs a
    single tensor multiplication.

    Args:
        image (np.ndarray): The 2D input image (H, W).
        kernel (np.ndarray): The 2D filter/kernel (h, w).

    Returns:
        np.ndarray: The 2D feature map (output) of size
                    (H - h + 1, W - w + 1).
    """

    # --- 1. Input Validation ---
    if not isinstance(image, np.ndarray) or not isinstance(kernel, np.ndarray):
        raise TypeError("Inputs must be NumPy ndarrays.")

    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError("Inputs must be 2-dimensional arrays.")

    img_h, img_w = image.shape
    ker_h, ker_w = kernel.shape

    if img_h < ker_h or img_w < ker_w:
        raise ValueError("Image dimensions (H, W) must be larger than or "
                         "equal to kernel dimensions (h, w).")

    # --- 2. Calculate Output Dimensions ---
    # This corresponds to 'valid' convolution
    out_h = img_h - ker_h + 1
    out_w = img_w - ker_w + 1

    # --- 3. Create Strided View ---
    # Get the byte strides of the image
    img_s0, img_s1 = image.strides

    # Create a 4D view of the image.
    # This view has shape (out_h, out_w, ker_h, ker_w)
    # Each element [i, j] of this view is the (ker_h, ker_w) patch
    # from the original image needed to calculate output[i, j].
    #
    # strides=(img_s0, img_s1, img_s0, img_s1)
    # - To move to the next row in the output (i -> i+1),
    #   we move one row down in the image (stride = img_s0).
    # - To move to the next col in the output (j -> j+1),
    #   we move one col right in the image (stride = img_s1).
    # - To move to the next row in the kernel patch (k -> k+1),
    #   we move one row down in the image (stride = img_s0).
    # - To move to the next col in the kernel patch (l -> l+1),
    #   we move one col right in the image (stride = img_s1).
    view_shape = (out_h, out_w, ker_h, ker_w)
    view_strides = (img_s0, img_s1, img_s0, img_s1)

    # as_strided creates a new view without copying data
    image_view = as_strided(image, shape=view_shape, strides=view_strides)

    # --- 4. Compute Convolution ---
    # We want to compute:
    # output[i, j] = sum(image_view[i, j, k, l] * kernel[k, l] for k, l)
    #
    # This is a tensor contraction. We can use np.einsum.
    # 'ijkl,kl->ij'
    # i,j are the output dimensions (out_h, out_w)
    # k,l are the kernel dimensions (ker_h, ker_w) to be summed over.
    output = np.einsum('ijkl,kl->ij', image_view, kernel)

    return output

def conv2d_loops(image, kernel):
    """
    A simple, naive implementation of 2D convolution (mode='valid',
    stride=1) using explicit Python loops for comparison.

    Args:
        image (np.ndarray): The 2D input image (H, W).
        kernel (np.ndarray): The 2D filter/kernel (h, w).

    Returns:
        np.ndarray: The 2D feature map (output).
    """
    img_h, img_w = image.shape
    ker_h, ker_w = kernel.shape

    if img_h < ker_h or img_w < ker_w:
        raise ValueError("Image dimensions must be larger than or "
                         "equal to kernel dimensions.")

    out_h = img_h - ker_h + 1
    out_w = img_w - ker_w + 1

    output = np.zeros((out_h, out_w), dtype=image.dtype)

    # Loop over every pixel of the output
    for i in range(out_h):
        for j in range(out_w):
            # Extract the relevant patch from the image
            patch = image[i : i + ker_h, j : j + ker_w]

            # Perform element-wise multiplication and sum
            output[i, j] = np.sum(patch * kernel)

    return output

# --- Complexity Analysis ---
"""
Let H, W = Image dimensions
Let h, w = Kernel dimensions

Function: conv2d_numpy (strided implementation)

Time Complexity: O((H - h + 1) * (W - w + 1) * h * w)
    - Or simplified: O(H * W * h * w)
    - The as_strided operation is O(1) time (it just creates a view).
    - The np.einsum operation must compute (H-h+1) * (W-w+1) output values.
    - Each output value requires (h * w) multiplications and (h * w - 1) 
      additions (a dot product).
    - Total operations are proportional to the product of the output 
      size and the kernel size.
    - If the kernel size (h, w) is considered a small constant (e.g., 3x3), 
      the complexity simplifies to O(H * W), which is linear in 
      the size of the image.

Space Complexity: O((H - h + 1) * (W - w + 1))
    - Or simplified: O(H * W)
    - The as_strided operation is O(1) space (it creates a view, 
      not a data copy).
    - The only significant new memory allocation is for the `output` array,
      which has the dimensions of the output feature map.
"""

# --- Test Cases ---
if __name__ == "__main__":
    print("--- Running 2D Convolution Test Cases ---")

    # === Test Case 1: User's Example ===
    print("\n[Test Case 1: User's 4x4 Image, 3x3 Kernel]")
    A = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])
    K = np.array([[1, 0, -1],
                  [1, 0, -1],
                  [1, 0, -1]])

    print("Image A:\n", A)
    print("Kernel K:\n", K)

    expected_output_1 = np.array([[-6, -6],
                                  [-6, -6]])

    output_1_numpy = conv2d_numpy(A, K)
    output_1_loops = conv2d_loops(A, K)

    print("Expected Output:\n", expected_output_1)
    print("conv2d_numpy Output:\n", output_1_numpy)
    print("conv2d_loops Output:\n", output_1_loops)

    assert np.array_equal(output_1_numpy, expected_output_1)
    assert np.array_equal(output_1_loops, expected_output_1)
    print("Test Case 1 Passed!")

    # === Test Case 2: Identity Kernel ===
    print("\n[Test Case 2: Identity Kernel (1x1)]")
    # Using the same Image A
    K_identity = np.array([[1]])

    print("Kernel K_identity:\n", K_identity)

    # With a 1x1 kernel, output size is (4-1+1, 4-1+1) = (4, 4)
    # The output should be the same as the input image
    expected_output_2 = A

    output_2_numpy = conv2d_numpy(A, K_identity)
    print("conv2d_numpy Output:\n", output_2_numpy)

    assert np.array_equal(output_2_numpy, expected_output_2)
    print("Test Case 2 Passed!")

    # === Test Case 3: Offset Identity Kernel (2x2) ===
    print("\n[Test Case 3: Offset Identity Kernel (2x2)]")
    K_offset = np.array([[0, 0],
                         [0, 1]])

    print("Kernel K_offset:\n", K_offset)

    # Output size: (4-2+1, 4-2+1) = (3, 3)
    # This kernel just picks the bottom-right element of each 2x2 patch
    # O[0,0] = A[0:2, 0:2] * K = 6
    # O[0,1] = A[0:2, 1:3] * K = 7
    # ...
    # The output should be A[1:, 1:]
    expected_output_3 = A[1:, 1:]

    output_3_numpy = conv2d_numpy(A, K_offset)
    print("Expected Output:\n", expected_output_3)
    print("conv2d_numpy Output:\n", output_3_numpy)

    assert np.array_equal(output_3_numpy, expected_output_3)
    print("Test Case 3 Passed!")

    # === Test Case 4: Edge Case (Kernel size == Image size) ===
    print("\n[Test Case 4: Kernel size == Image size]")
    A_small = np.array([[1, 2], [3, 4]])
    K_full = np.array([[1, 2], [3, 4]])

    print("Image A_small:\n", A_small)
    print("Kernel K_full:\n", K_full)

    # Output size: (2-2+1, 2-2+1) = (1, 1)
    # Output is the dot product of the two
    expected_output_4 = np.array([[np.sum(A_small * K_full)]]) # 1*1+2*2+3*3+4*4 = 1+4+9+16 = 30

    output_4_numpy = conv2d_numpy(A_small, K_full)
    print("Expected Output:\n", expected_output_4)
    print("conv2d_numpy Output:\n", output_4_numpy)

    assert np.array_equal(output_4_numpy, expected_output_4)
    print("Test Case 4 Passed!")

    # === Test Case 5: Edge Case (Image < Kernel) ===
    print("\n[Test Case 5: Image < Kernel (Should fail)]")
    try:
        conv2d_numpy(A_small, K)
        print("Test Case 5 FAILED: Did not raise ValueError")
    except ValueError as e:
        print(f"Caught expected error: {e}")
        print("Test Case 5 Passed!")

    # === Test Case 6: Edge Case (Non-2D input) ===
    print("\n[Test Case 6: Non-2D input (Should fail)]")
    try:
        conv2d_numpy(np.array([1, 2, 3]), K)
        print("Test Case 6 FAILED (Image): Did not raise ValueError")
    except ValueError as e:
        print(f"Caught expected error (Image): {e}")
        print("Test Case 6 (Image) Passed!")

    try:
        conv2d_numpy(A, np.array([1, 0, -1]))
        print("Test Case 6 FAILED (Kernel): Did not raise ValueError")
    except ValueError as e:
        print(f"Caught expected error (Kernel): {e}")
        print("Test Case 6 (Kernel) Passed!")

    print("\n--- All Test Cases Run ---")

    # --- Performance Comparison (Optional) ---
    print("\n[Performance Comparison on larger array]")
    large_image = np.random.rand(500, 500)
    medium_kernel = np.random.rand(7, 7)

    start_time_numpy = time.time()
    conv2d_numpy(large_image, medium_kernel)
    end_time_numpy = time.time()
    print(f"conv2d_numpy (strided) time: {end_time_numpy - start_time_numpy:.6f} s")

    start_time_loops = time.time()
    conv2d_loops(large_image, medium_kernel)
    end_time_loops = time.time()
    print(f"conv2d_loops (naive) time:   {end_time_loops - start_time_loops:.6f} s")
    print("Note: The 'strided' numpy version is significantly faster.")
