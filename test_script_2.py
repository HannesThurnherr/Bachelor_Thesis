import numpy as np
import time

# Initialize parameters
num_mats = 500000
dim_mat = 30
A = [np.random.rand(dim_mat, dim_mat) for i in range(num_mats)]
steps = 6
M1 = np.random.rand(dim_mat, dim_mat-10)

# Define relu function
def relu(x):
    return x * (x > 0)

# Define batch_matrix_multiply function
def batch_matrix_multiply(M: np.ndarray, A_list, steps: int):
    A_tensor = np.stack(A_list)
    num_mats = len(A_list)
    M_expanded = np.expand_dims(M, axis=0).repeat(num_mats, axis=0)
    R = M_expanded
    print("batch method:", end ="")
    for step in range(steps):
        R_shape = R.shape[1]
        A_sliced = A_tensor[:, :R_shape, :]
        R_sliced = R[:, :R_shape, :]
        R_next = np.matmul(A_sliced, R_sliced)
        R = relu(R_next[:, :, :M.shape[1]])
        print("|", end="")
    return [R[i] for i in range(num_mats)]


def optimized_batch_matrix_multiply(M: np.ndarray, A_list, steps: int):
    A_tensor = np.stack(A_list)
    num_mats = len(A_list)
    M_expanded = np.expand_dims(M, axis=0).repeat(num_mats, axis=0)

    # Pre-allocate space for R
    R = np.empty_like(M_expanded)

    # Copy the initial matrix M to R
    np.copyto(R, M_expanded)
    print("optimised batch method:", end="")
    for step in range(steps):

        R_shape = R.shape[1]
        A_sliced = A_tensor[:, :R_shape, :]
        R_sliced = R[:, :R_shape, :]

        # Use np.matmul and in-place relu to update R
        R_next = np.matmul(A_sliced, R_sliced)
        np.maximum(R_next, 0, out=R_next)  # In-place ReLU
        np.copyto(R[:, :, :M.shape[1]], R_next[:, :, :M.shape[1]])  # Copy to the pre-allocated R
        print("|", end="")
    return [R[i] for i in range(num_mats)]


# Measure time for batch_matrix_multiply
start_time = time.time()
a = batch_matrix_multiply(M1, A, 6)[0]
batch_time = time.time() - start_time
print()
print("old method", end="")
# Measure time for loop-based multiplication
start_time = time.time()
results = []
n = 0
for mat in A:
    res = np.copy(M1)
    for i in range(steps):
        res = relu(np.dot(mat, res))
    results.append(res)
    n+=1
    if(n % 10000 == 0):
        print("|", end="")
loop_time = time.time() - start_time
print()
b = results[0]
print()
# Output results
print("Batch Time:", batch_time)
print("Loop Time:", loop_time)
print("Shape of 'a':", a.shape)
print("Shape of 'b':", b.shape)
print("Values match:", np.allclose(a, b, atol=1e-7))
