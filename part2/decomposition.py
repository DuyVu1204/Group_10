import math
import random
import numpy as np

# =========================
# Các hàm chức năng
# =========================
#Tạo ma trận 0
def zeros_matrix(rows, cols):
    return [[0.0] * cols for _ in range(rows)]
#Tạo ma trận đơn vị
def identity_matrix(size):
    mat = zeros_matrix(size, size)
    for i in range(size):
        mat[i][i] = 1.0
    return mat
#Chuyển vị ma trận
def transpose(matrix):
    return [list(row) for row in zip(*matrix)]
#Tính tích vô hướng 2 vecto
def dot_product(vec1, vec2):
    return sum(x * y for x, y in zip(vec1, vec2))
#Nhân ma trận
def matrix_multiply(mat1, mat2):
    mat2_t = transpose(mat2)
    return [[dot_product(row1, col2) for col2 in mat2_t] for row1 in mat1]
#Lấy cột thứ i
def get_column(matrix, i):
    return [row[i] for row in matrix]
#Tính norm
def vector_norm(vec):
    return math.sqrt(sum(x * x for x in vec))
#Nhân vector với hàm số
def scale_vector(vec, scalar):
    return [x * scalar for x in vec]
#Trừ vector
def subtract_vectors(vec1, vec2):
    return [x - y for x, y in zip(vec1, vec2)]

# =========================
# Thuật toán phân rã
# =========================

#Gram_schmidt để phân rã QR
def modified_gram_schmidt(A):
    m, n = len(A), len(A[0])
    Q = zeros_matrix(m, n)
    R = zeros_matrix(n, n)
    V = transpose(A)

    for j in range(n):
        v_j = V[j]
        R[j][j] = vector_norm(v_j)

        if R[j][j] < 1e-12:
            q_j = [0.0] * m
        else:
            q_j = scale_vector(v_j, 1.0 / R[j][j])

        for i in range(m):
            Q[i][j] = q_j[i]

        for k in range(j + 1, n):
            v_k = V[k]
            R[j][k] = dot_product(q_j, v_k)
            V[k] = subtract_vectors(v_k, scale_vector(q_j, R[j][k]))

    return Q, R
#Phân rã QR lặp để tìm trị riêng và vector riêng
def qr_eigen_decomposition(A, num_iterations=200, tol=1e-10):
    n = len(A)
    Q_total = identity_matrix(n)
    A_k = [row[:] for row in A]

    for _ in range(num_iterations):
        Q, R = modified_gram_schmidt(A_k)
        A_k = matrix_multiply(R, Q)
        Q_total = matrix_multiply(Q_total, Q)

        off = 0.0
        for i in range(n):
            for j in range(n):
                if i != j:
                    off += A_k[i][j] ** 2
        if math.sqrt(off) < tol:
            break

    eigenvalues = [A_k[i][i] for i in range(n)]
    eigenvectors = Q_total
    return eigenvalues, eigenvectors
#Phân rã SVD theo phương thức giống tính tay
def svd_decomposition(A, eig_use_numpy_if_order_gt=4):
    m, n = len(A), len(A[0])

    At = transpose(A)
    AtA = matrix_multiply(At, A)  

    # Chọn cách tính eigen(AtA)
    if n > eig_use_numpy_if_order_gt:
        # Dùng NumPy cho eigenvalues/eigenvectors (ma trận đối xứng)
        AtA_np = np.array(AtA, dtype=float)

        # eigh dành cho ma trận đối xứng: ổn định và trả eigenvalues tăng dần
        w, V_np = np.linalg.eigh(AtA_np) 

        # Đổi sang giảm dần để khớp quy ước singular values giảm dần
        idx = np.argsort(w)[::-1]
        eigenvalues = w[idx].tolist()
        V_np = V_np[:, idx]

        # Đưa về dạng list theo format dùng ở phần tính tay
        V = V_np.tolist() 
    else:
        # Tính tay hoàn toàn bằng QR iteration
        eigenvalues, V_list = qr_eigen_decomposition(AtA)
        V = transpose(V_list) 

        # Sắp xếp giảm dần theo eigenvalue để đồng bộ V
        eigen_pairs = sorted(zip(eigenvalues, transpose(V)), key=lambda x: x[0], reverse=True)
        eigenvalues = [p[0] for p in eigen_pairs]
        V_cols = [p[1] for p in eigen_pairs]
        V = transpose(V_cols)

    # Từ eigen => singular values + Sigma
    singular_values = [math.sqrt(max(0.0, val)) for val in eigenvalues]

    Sigma = zeros_matrix(m, n)
    for i in range(min(m, n)):
        Sigma[i][i] = singular_values[i]

    # Tính U từ u_i = A v_i / sigma_i
    U_cols = []
    k = min(m, n)
    for i in range(k):
        if singular_values[i] > 1e-12:
            v_i = get_column(V, i)
            Av_i = [dot_product(row, v_i) for row in A]
            u_i = scale_vector(Av_i, 1.0 / singular_values[i])
            U_cols.append(u_i)
        else:
            U_cols.append([0.0] * m)

    # Bổ sung cột cho U để đủ m×m 
    if m > len(U_cols):
        basis = U_cols[:]
        needed = m - len(U_cols)

        while needed > 0:
            random_vec = [random.uniform(-1, 1) for _ in range(m)]
            for b in basis:
                proj = dot_product(random_vec, b)
                random_vec = subtract_vectors(random_vec, scale_vector(b, proj))

            norm = vector_norm(random_vec)
            if norm > 1e-12:
                basis.append(scale_vector(random_vec, 1.0 / norm))
                needed -= 1

        U = transpose(basis)
    else:
        U = transpose(U_cols)

    Vt = transpose(V)
    return U, Sigma, Vt, singular_values



#Kiểm tra kết quả
def check_result(A, U, Sigma, Vt, s_custom, case_name):
    A_np = np.array(A, dtype=float)

    U_np, s_np, Vt_np = np.linalg.svd(A_np, full_matrices=True)
    m, n = A_np.shape
    Sigma_np = np.zeros((m, n))
    for i in range(min(m, n)):
        Sigma_np[i, i] = s_np[i]

    A_rec_custom = np.array(matrix_multiply(matrix_multiply(U, Sigma), Vt), dtype=float)
    A_rec_np = U_np @ Sigma_np @ Vt_np

    rec_err_custom = np.linalg.norm(A_np - A_rec_custom, ord='fro')
    rec_err_numpy = np.linalg.norm(A_np - A_rec_np, ord='fro')
    k = min(m, n)
    s_err = np.linalg.norm(np.array(s_custom[:k]) - s_np[:k], ord=2)

    # In kết quả custom 
    print("Kết quả custom:")
    print("U_custom =\n", np.round(np.array(U, dtype=float), 8))
    print("Sigma_custom =\n", np.round(np.array(Sigma, dtype=float), 8))
    print("Vt_custom =\n", np.round(np.array(Vt, dtype=float), 8))
    print("A_rec_custom =\n", np.round(A_rec_custom, 8))
    # In kết quả chuẩn
    print("\nKết quả chuẩn:")
    print("U_np =\n", np.round(U_np, 8))
    print("Sigma_np =\n", np.round(Sigma_np, 8))
    print("Vt_np =\n", np.round(Vt_np, 8))
    print("A_rec_np =\n", np.round(A_rec_np, 8))

    print("\nSai số:")
    print(f"- ||A - U_custom*Sigma_custom*Vt_custom||_F = {rec_err_custom:.6e}")
    print(f"- ||A - U_np*Sigma_np*Vt_np||_F             = {rec_err_numpy:.6e}")
    print(f"- ||s_custom - s_np||_2                     = {s_err:.6e}")

if __name__ == '__main__':
    random.seed(42)

    examples = [
    # 1) VD1: ma trận 0 (mọi singular value = 0)
    ("VD1", [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ]),

    # 2) VD2: ma trận đơn vị (singular values = 1)
    ("VD2", [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]),

    # 3) VD3: rank-deficient (các hàng phụ thuộc tuyến tính)
    # hàng 2 = 2*hàng 1; hàng 3 = 3*hàng 1  => rank = 1
    ("VD3", [
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 6.0, 9.0]
    ]),

    # 4) VD4: có một cột toàn 0 (làm xuất hiện singular value = 0 rõ rệt)
    ("VD4", [
        [1.0, 0.0, 2.0],
        [3.0, 0.0, 4.0],
        [5.0, 0.0, 6.0]
    ]),

    # 5) VD5: ma trận chữ nhật cao (m > n)
    ("VD5", [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0]
    ]),

    # 6) VD6: ma trận chữ nhật rộng (m < n) (chú ý: U full_matrices sẽ là m×m, Vt là n×n)
    ("VD6", [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0, 9.0, 10.0]
    ]),

    # 7) VD7: giá trị lớn và chênh lệch thang đo
    ("VD7", [
        [1e9,  2.0,  3.0],
        [4.0,  5e-9, 6.0],
        [7.0,  8.0,  9e3]
    ]),
]
    
    for case_name, A in examples:
        print(f"\n================ {case_name} ================")
        print("Ma trận đầu vào A:")
        for row in A:
            print(row)

        U, Sigma, Vt, s_custom = svd_decomposition(A)
        check_result(A, U, Sigma, Vt, s_custom, case_name)