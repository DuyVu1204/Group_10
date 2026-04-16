from typing import List, Optional

# --- Định nghĩa kiểu dữ liệu (Type Hinting) ---
Number = float
Matrix = List[List[Number]]

def _zeros(m: int, n: int) -> Matrix: 
    """Tạo ma trận m x n toàn số 0."""
    return [[0.0 for _ in range(n)] for _ in range(m)]

def _identity(n: int) -> Matrix: 
    """Tạo ma trận đơn vị I kích thước n x n."""
    I = _zeros(n, n)
    for i in range(n):
        I[i][i] = 1.0
    return I

def _swap_rows(M: Matrix, i: int, j: int) -> None:  
    """Hoán đổi hai dòng i và j của ma trận M."""
    M[i], M[j] = M[j], M[i]

def inverse(A: Matrix, eps: float = 1e-12) -> Optional[Matrix]:
    """
    Tìm ma trận nghịch đảo A^-1 bằng phương pháp Gauss-Jordan.
    
    Thuật toán:
    1. Ghép ma trận đơn vị I vào bên phải A tạo thành ma trận mở rộng [A | I].
    2. Dùng Partial Pivoting để đưa nửa trái về dạng ma trận đường chéo.
    3. Chuẩn hóa các phần tử trên đường chéo về 1.
    4. Nửa phải lúc này chính là A^-1.
    """
    n = len(A)
    
    # Bắt lỗi: Ma trận không vuông thì không có nghịch đảo
    if n == 0 or any(len(row) != n for row in A):
        print("Lỗi: Ma trận không vuông, không thể tìm nghịch đảo.")
        return None

    # Bước 1: Tạo ma trận mở rộng M = [A | I]
    M = [A[i][:] + _identity(n)[i] for i in range(n)]
    total_cols = 2 * n
    r = 0
    
    for c in range(n):
        # Bước 2: Partial Pivoting (Tìm chốt lớn nhất trong cột c)
        p = max(range(r, n), key=lambda i: abs(M[i][c]))
        
        if abs(M[p][c]) <= eps:
            print(f"Lỗi: Ma trận suy biến (Singular Matrix), không thể tìm nghịch đảo.")
            return None
            
        if p != r:
            _swap_rows(M, p, r)

        # Bước 3: Chuẩn hóa dòng chứa chốt (chia cả dòng cho phần tử chốt)
        pivot = M[r][c]
        for j in range(c, total_cols):
            M[r][j] /= pivot

        # Bước 4: Khử các phần tử CẢ TRÊN VÀ DƯỚI chốt thành 0 (Đặc trưng của Gauss-Jordan)
        for i in range(n):
            if i == r:
                continue
            if abs(M[i][c]) <= eps:
                continue
            factor = M[i][c]
            for j in range(c, total_cols):
                M[i][j] -= factor * M[r][j]
            M[i][c] = 0.0
            
        r += 1
        if r == n:
            break

    # Bước 5: Trích xuất nửa bên phải của ma trận mở rộng (chính là A^-1)
    invA = [row[n:] for row in M]
    return invA


if __name__ == "__main__":
    import numpy as np
    
    print("\n" + "="*80)
    print("KIỂM THỬ MA TRẬN NGHỊCH ĐẢO VÀ ĐÁNH GIÁ AA^-1 = I (5 TEST CASES)")
    print("="*80)

    # Bộ 5 Test Cases (bao gồm Edge Cases)
    test_cases = [
        (
            "Test 1 (Normal): Ma trận 3x3 bình thường khả nghịch",
            [[2, 1, 1], [1, 3, 2], [1, 0, 0]]
        ),
        (
            "Test 2 (Edge Case - Zero Pivot): A[0][0] = 0, cần Partial Pivoting để tránh Divide by Zero",
            [[0, 2, 3], [1, 1, -1], [2, -1, 1]]
        ),
        (
            "Test 3 (Edge Case - Singular Matrix): Ma trận suy biến (det = 0), không có nghịch đảo",
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        ),
        (
            "Test 4 (Edge Case - Ill-conditioned): Ma trận chứa số cực nhỏ/lớn",
            [[1e-10, 1], [1, 1]]
        ),
        (
            "Test 5 (Edge Case - Non-square): Ma trận chữ nhật (Bắt lỗi)",
            [[1, 2, 3], [4, 5, 6]]
        )
    ]

    for title, A in test_cases:
        print(f"\n{title}")
        invA = inverse(A)
        
        if invA is not None:
            A_np = np.array(A, dtype=float)
            invA_np = np.array(invA, dtype=float)
            
            # 1. Đánh giá tính chất toán học cơ bản: A * A^-1 = I
            I_approx = A_np @ invA_np
            identity_error = float(np.linalg.norm(I_approx - np.eye(len(A)), ord=np.inf))
            
            # 2. Đối chiếu trực tiếp với NumPy
            invA_numpy = np.linalg.inv(A_np)
            diff_norm = float(np.linalg.norm(invA_np - invA_numpy, ord=np.inf))
            
            print(f"  -> Sai số A*A^-1 so với I    : {identity_error:.2e}")
            print(f"  -> Sai số A^-1 so với NumPy  : {diff_norm:.2e}")
            
            if diff_norm < 1e-8 and identity_error < 1e-8:
                print("  -> Kết quả: ✓ PASSED (Đạt chuẩn)")
            else:
                print("  -> Kết quả: ✗ FAILED")
        else:
            print("  -> Kết quả: ✓ PASSED (Đã bắt lỗi thành công)")