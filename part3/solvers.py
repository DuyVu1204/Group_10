"""
solvers.py — Các thuật toán giải hệ phương trình Ax = b (Phiên bản Thuần Python)
════════════════════════════════════════════════════════════════════════════════
1. Khử Gauss (Import từ Part 1)
2. SVD Solver (Dùng phân rã từ Part 2 + Giải bằng Python thuần)
3. Gauss-Seidel (Cài đặt Python thuần đúng công thức đề bài)
"""

import math
import sys
import os

# Thêm đường dẫn root của Group_10 để có thể import từ các thư mục part1, part2
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from part1 import gaussian as part1_gauss
    from part2 import decomposition as part2_decomp
except ImportError as e:
    # Nếu không import được kiểu package, thử import trực tiếp từng thư mục
    try:
        sys.path.append(os.path.join(project_root, "part1"))
        sys.path.append(os.path.join(project_root, "part2"))
        import gaussian as part1_gauss
        import decomposition as part2_decomp
    except ImportError as e2:
        part1_gauss = None
        part2_decomp = None

def is_strictly_diagonally_dominant(A):
    """Kiểm tra ma trận A có chéo trội chặt hàng hay không (Python thuần)."""
    n = len(A)
    for i in range(n):
        diag = abs(A[i][i])
        row_sum = sum(abs(A[i][j]) for j in range(n) if i != j)
        if diag <= row_sum:
            return False
    return True

# ── 1. GAUSSIAN ELIMINATION (Part 1) ──
def gaussian_eliminate(A_in, b_in):
    """Giải Ax = b bằng khử Gauss nhập từ Part 1."""
    if part1_gauss is None:
        raise ImportError("Không tìm thấy gaussian.py ở Part 1.")
    
    # gaussian_eliminate của part1 trả về (M, x, s)
    _, x, _ = part1_gauss.gaussian_eliminate(A_in, b_in)
    
    if x is None:
        return None
    return x

# ── 2. SVD SOLVER (Part 2 + Logic Python thuần) ──
def svd_solver(A, b):
    """
    Giải Ax = b bằng SVD: x = V Σ⁻¹ Uᵀ b
    Sử dụng kết quả phân rã từ Part 2.
    """
    if part2_decomp is None:
        raise ImportError("Không tìm thấy decomposition.py ở Part 2.")
    
    # 1. Phân rã SVD
    U, Sigma, Vt, sigma = part2_decomp.svd_decomposition(A)
    m = len(A)
    n = len(A[0])
    
    # 2. Tính d = Uᵀ b (Ma trận-Vector product)
    # d_j = Σ_i U[i][j] * b[i]
    d = [0.0] * len(sigma)
    for j in range(len(sigma)):
        col_sum = 0.0
        for i in range(m):
            col_sum += U[i][j] * b[i]
        d[j] = col_sum
            
    # 3. Tính y = Σ⁻¹ d
    y = [0.0] * len(sigma)
    for i in range(len(sigma)):
        if sigma[i] > 1e-15:
            y[i] = d[i] / sigma[i]
            
    # 4. Tính x = V y (V = Vt.T -> x_i = Σ_j V[i][j] * y[j] = Σ_j Vt[j][i] * y[j])
    # x_i là tổ hợp tuyến tính các cột của V (hàng của Vt)
    x = [0.0] * n
    for i in range(n):
        row_sum = 0.0
        for j in range(len(sigma)):
            row_sum += Vt[j][i] * y[j]
        x[i] = row_sum
            
    return x

# ── 3. GAUSS-SEIDEL (Part 3 - Đúng công thức lặp) ──
def gauss_seidel(A, b, tol=1e-10, max_iter=5000):
    """
    Giải Ax = b bằng phương pháp lặp Gauss-Seidel (Python thuần).
    Đúng theo công thức (13) trong yêu cầu đồ án.
    """
    n = len(A)
    x = [0.0] * n  # Khởi tạo vector nghiệm ban đầu bằng 0
    
    for k in range(max_iter):
        x_old = x[:]  # Lưu lại nghiệm cũ để tính sai số dừng
        
        for i in range(n):
            # Công thức (13): x_i = (1/a_ii) * (b_i - Σ_{j<i} a_ij*x_new_j - Σ_{j>i} a_ij*x_old_j)
            
            s1 = 0.0
            for j in range(i):
                s1 += A[i][j] * x[j]  # Đã dùng giá trị mới vừa tính ở bước j < i
                
            s2 = 0.0
            for j in range(i + 1, n):
                s2 += A[i][j] * x_old[j] # Dùng giá trị cũ từ vòng lặp trước
                
            if abs(A[i][i]) < 1e-18:
                # Trường hợp đường chéo có số 0, thuật toán GS không thể tiếp tục
                return x, k  
                
            x[i] = (b[i] - s1 - s2) / A[i][i]
        
        # Kiểm tra điều kiện dừng: chuẩn vô cùng của (x_new - x_old)
        diff = 0.0
        for i in range(n):
            d = abs(x[i] - x_old[i])
            if d > diff:
                diff = d
                
        if diff < tol:
            return x, k + 1
            
    return x, max_iter
