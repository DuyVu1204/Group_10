from typing import List, Optional

Number = float
Matrix = List[List[Number]]

def determinant(A: Matrix) -> Optional[Number]:
    """
    Tính định thức của ma trận vuông A bằng phương pháp khử Gauss với Partial Pivoting.
    """

    # Kiểm tra nếu ma trận không vuông hoặc rỗng
    n = len(A)
    if n == 0 or any(len(row) != n for row in A):
        return None

    M = [row[:] for row in A]
    det_sign = 1 # Biến theo dõi dấu của định thức
    det_val = 1.0 # Biến tích lũy giá trị định thức
    eps = 1e-12

    for k in range(n):
        pivot_row = k # Dòng chứa phần tử pivot lớn nhất
        max_val = abs(M[k][k])

        for i in range(k + 1, n):
            if abs(M[i][k]) > max_val :
                max_val = abs(M[i][k])
                pivot_row = i

        if max_val < eps:
            return 0.0
        
        if pivot_row != k:
            M[k], M[pivot_row] = M[pivot_row], M[k] 
            det_sign *= -1

        pivot = M[k][k]
        det_val *= pivot

        for i in range(k + 1, n):
            factor = M[i][k] / pivot
            for j in range(k, n):
                M[i][j] -= factor * M[k][j]

    return det_val * det_sign

def verify_determinant(A: Matrix, atol: float = 1e-8) -> int:
    """
    Xác minh lại kết quả định thức bằng thư viện NumPy.
    Trả về: 0 nếu định thức chính xác
            1 nếu định thức gần đúng
            -1 nếu định thức sai lệch lớn hoặc không thể tính.
    """
    
    import numpy as np
    
    n = len(A)
    if n == 0 or any(len(row) != n for row in A):
        print("  -> [NumPy Verify] Lỗi: Ma trận không vuông, không thể tính định thức.")
        return -1
    
    # Chuyển đổi A sang numpy array
    A_np = np.array(A, dtype=float)
    
    # Tính định thức bằng NumPy (chuẩn)
    numpy_det = np.linalg.det(A_np)
    
    # Tính định thức từ hàm determinant 
    det = determinant(A)
    
    if det is not None:
        print(f"  -> Định thức từ hàm: {det:.6e}")
    else:
        print("  -> Định thức từ hàm của bạn: None")
    print(f"  -> [NumPy Verify] Định thức từ NumPy: {numpy_det:.6e}")
    
    # Tính sai số tuyệt đối giữa hai kết quả
    error = abs(det - numpy_det)
    
    print(f"  -> [NumPy Verify] Sai số tuyệt đối: {error:.2e}")

    if error < atol:
        print("  -> [NumPy Verify] Kết quả: ĐỊNH THỨC CHÍNH XÁC")
        return 0
    elif error < 10 * atol:
        print("  -> [NumPy Verify] Kết quả: ĐỊNH THỨC GẦN ĐÚNG")
        return 1
    else:
        print("  -> [NumPy Verify] Kết quả: ĐỊNH THỨC SAI LỆCH LỚN")
        return -1

if __name__ == "__main__":
    print("\n" + "="*80)
    print("KIỂM THỬ THUẬT TOÁN TÍNH ĐỊNH THỨC BẰNG KHỬ GAUSS ")
    print("="*80)

    test_cases = [
        (
            "Test 1: Ma trận cơ bản",
            [[1, 2, 8, 9], 
             [3, 4, 5, 3],
             [9, 1, 3, 2],
             [4, 5, 7, 6]]
        ),
        (
            "Test 2: Ma trận đường chéo",
            [[5, 0, 0, 0, 0], 
             [0, 3, 0, 0, 0], 
             [0, 0, 2, 0, 0],
             [0, 0, 0, 8, 0],
             [0, 0, 0, 0, 4]]
        ),
        (
            "Test 3: Ma trận tam giác trên",
            [[1, 2, 3, 4],
             [0, 5, 6, 7],
             [0, 0, 8, 9],
             [0, 0, 0, 10]]
        ),
        (
            "Test 3: Ma trận suy biến",
            [[1, 2, 3], 
             [4, 5, 6], 
             [7, 8, 9]],
        ),
        (
            "Test 4: Ma trận suy biến (2 dòng giống nhau)",
            [[1, 2, 3, 4],
             [5, 6, 7, 8],
             [1, 2, 3, 4],
             [9, 10, 11, 12]]
        ),
        (
            "Test 5: Ma trận Hilbert (ill-conditioned)",
            [[1, 1/2, 1/3, 1/4],
             [1/2, 1/3, 1/4, 1/5],
             [1/3, 1/4, 1/5, 1/6],
             [1/4, 1/5, 1/6, 1/7]]
        ),
        (
            "Test 6: Ma trận không vuông",
            [[1, 2, 3], 
             [4, 5, 6]]
        ),
        (
            "Test 7: Ma trận gần suy biến",
            [[1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 2],
             [1, 1, 1, 1, 2, 2],
             [1, 1, 1, 2, 2, 2],
             [1, 1, 2, 2, 2, 2],
             [1, 2, 2, 2, 2, 2]]
        ),
        (
            "Test 8: Ma trận có pivot nhỏ",
            [[1e-10, 1, 0, 0, 0],
             [1, 1e-10, 1, 0, 0],
             [0, 1, 1e-10, 1, 0],
             [0, 0, 1, 1e-10, 1],
             [0, 0, 0, 1, 1e-10]]
        ),
    ]

    for title, A in test_cases:
        print(f"\n{title}")
        result = verify_determinant(A)
        
        if result == 0:
            print("  -> Kết luận: ĐỊNH THỨC ĐÚNG")
        elif result == 1:
            print("  -> Kết luận: ĐỊNH THỨC GẦN ĐÚNG")
        else:
            print("  -> Kết luận: ĐỊNH THỨC SAI")