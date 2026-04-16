from typing import List, Tuple, Dict, Any

# --- Định nghĩa kiểu dữ liệu (Type Hinting) ---
Number = float
Matrix = List[List[Number]]
Vector = List[Number]

def _deepcopy_mat(A: Matrix) -> Matrix: 
    """Tạo một bản sao sâu (deep copy) của ma trận A để tránh thay đổi dữ liệu gốc."""
    return [row[:] for row in A]

def _swap_rows(M: Matrix, i: int, j: int) -> None: 
    """Hoán đổi hai hàng i và j của ma trận M."""
    M[i], M[j] = M[j], M[i]

def rref(A: Matrix, eps: float = 1e-12) -> Tuple[Matrix, List[int]]:
    """
    Đưa ma trận về dạng bậc thang rút gọn (Reduced Row Echelon Form - RREF).
    Sử dụng phương pháp khử Gauss-Jordan có Partial Pivoting.
    
    Trả về:
        Tuple chứa ma trận RREF và danh sách chỉ số các cột chứa phần tử chốt (pivot columns).
    """
    M = _deepcopy_mat(A)
    m = len(M)
    n = len(M[0]) if m > 0 else 0

    r = 0
    pivot_cols: List[int] = []
    
    for c in range(n):
        # 1. Partial Pivoting: Tìm chốt lớn nhất trong cột c từ dòng r trở xuống
        p = None
        maxabs = 0.0
        for i in range(r, m):
            val = abs(M[i][c])
            if val > maxabs:
                maxabs = val
                p = i
                
        # Nếu cột này toàn 0 (hoặc xấp xỉ 0), bỏ qua
        if p is None or maxabs <= eps:
            continue
            
        if p != r:
            _swap_rows(M, p, r)

        # 2. Chuẩn hóa dòng chốt (chia cho phần tử chốt để đưa về 1)
        pivot = M[r][c]
        for j in range(c, n):
            M[r][j] /= pivot

        # 3. Khử các phần tử CẢ TRÊN VÀ DƯỚI chốt về 0
        for i in range(m):
            if i == r:
                continue
            factor = M[i][c]
            if abs(factor) <= eps:
                continue
            for j in range(c, n):
                M[i][j] -= factor * M[r][j]
            M[i][c] = 0.0

        pivot_cols.append(c)
        r += 1
        if r == m:
            break

    # 4. Dọn dẹp sai số float (những số như 1e-16 đưa hẳn về 0.0 cho đẹp)
    for i in range(m):
        for j in range(n):
            if abs(M[i][j]) < eps:
                M[i][j] = 0.0

    return M, pivot_cols


def rank_and_basis(A: Matrix, eps: float = 1e-12) -> Dict[str, Any]:
    """
    Tìm hạng (rank) và cơ sở (basis) của 3 không gian vector quan trọng.
    
    Lý thuyết Toán học:
    - Hạng (Rank): Bằng số lượng cột pivot.
    - Cơ sở không gian cột (Column Space): Trích xuất các cột pivot từ MA TRẬN GỐC A.
    - Cơ sở không gian dòng (Row Space): Các dòng khác 0 trong ma trận RREF.
    - Cơ sở không gian nghiệm (Null Space): Các vector sinh ra từ việc cho từng biến tự do = 1, còn lại = 0.
    """
    m = len(A)
    n = len(A[0]) if m > 0 else 0

    R, pivot_cols = rref(A, eps=eps)
    rank = len(pivot_cols)

    # 1. Cơ sở Không gian cột (Lấy từ ma trận gốc A)
    col_basis: List[Vector] = []
    for j in pivot_cols:
        col_basis.append([A[i][j] for i in range(m)])

    # 2. Cơ sở Không gian dòng (Lấy từ ma trận RREF)
    row_basis: List[Vector] = []
    for i in range(m):
        if any(abs(R[i][j]) > eps for j in range(n)):
            row_basis.append(R[i][:])

    # 3. Cơ sở Không gian nghiệm (Nullspace)
    pivot_set = set(pivot_cols)
    free_cols = [j for j in range(n) if j not in pivot_set]
    null_basis: List[Vector] = []

    # Tạo mapping từ cột chốt sang dòng chứa chốt đó trong RREF
    pivot_row_for_col: Dict[int, int] = {}
    for i in range(m):
        for j in range(n):
            if abs(R[i][j]) > eps:
                if j in pivot_set:
                    pivot_row_for_col[j] = i
                break

    # Tính các vector cơ sở cho Nullspace
    for f in free_cols:
        v = [0.0 for _ in range(n)]
        v[f] = 1.0 # Cho biến tự do hiện tại = 1
        for pc in pivot_cols:
            if pc in pivot_row_for_col:
                i = pivot_row_for_col[pc]
                v[pc] = -R[i][f] # Các biến cơ sở = - hệ số của biến tự do
        null_basis.append(v)

    return {
        "rank": rank,
        "pivot_cols": pivot_cols,
        "column_space_basis": col_basis,
        "row_space_basis": row_basis,
        "null_space_basis": null_basis,
        "rref": R,
    }
    
if __name__ == "__main__":
    import numpy as np
    
    print("\n" + "="*80)
    print("KIỂM THỬ HẠNG & CƠ SỞ (5 TEST CASES BAO GỒM EDGE CASES)")
    print("="*80)
    
    test_cases = [
        (
            "Test 1 (Normal): Ma trận 3x4 có hạng 2",
            [[1, 2, 0, 1], [2, 4, 1, 4], [3, 6, 1, 5]]
        ),
        (
            "Test 2 (Edge Case - Full Rank): Ma trận vuông khả nghịch (Không có Null Space)",
            [[2, 1, 1], [1, 3, 2], [1, 0, 0]]
        ),
        (
            "Test 3 (Edge Case - Zero Matrix): Ma trận toàn số 0 (Hạng 0, Null Space tối đa)",
            [[0, 0, 0], [0, 0, 0]]
        ),
        (
            "Test 4 (Edge Case - Rank 1): Các dòng/cột tỷ lệ với nhau (Rank = 1)",
            [[1, 2, 3], [2, 4, 6], [3, 6, 9]]
        ),
        (
            "Test 5 (Edge Case - Tall Matrix): Ma trận đứng (Nhiều dòng hơn cột)",
            [[1, 2], [3, 4], [5, 6], [7, 8]]
        )
    ]

    for title, A in test_cases:
        print(f"\n{title}")
        res = rank_and_basis(A)
        
        # --- KIỂM CHỨNG VỚI NUMPY ---
        A_np = np.array(A, dtype=float)
        rank_np = np.linalg.matrix_rank(A_np)
        
        n_cols = len(A[0]) if A else 0
        dim_col = len(res['column_space_basis'])
        dim_row = len(res['row_space_basis'])
        dim_null = len(res['null_space_basis'])
        
        # 1. Kiểm tra Số chiều (Rank-Nullity Theorem: rank + nullity = n)
        rank_ok = (res['rank'] == rank_np) and (dim_col == rank_np) and (dim_row == rank_np)
        nullity_ok = (dim_null == n_cols - rank_np)
        
        # 2. Kiểm tra Không gian nghiệm: A * v = 0
        null_math_ok = True
        for v in res['null_space_basis']:
            v_np = np.array(v, dtype=float)
            err = float(np.linalg.norm(A_np @ v_np, ord=np.inf))
            if err > 1e-8:
                null_math_ok = False
                
        # Phán quyết
        print(f"  -> Hạng (Tự code / NumPy): {res['rank']} / {rank_np}")
        print(f"  -> Chiều không gian (Cột/Dòng/Nghiệm): {dim_col} / {dim_row} / {dim_null}")
        
        if rank_ok and nullity_ok and null_math_ok:
            print("  -> Kết quả: ✓ PASSED (Tính toán chính xác và thỏa Định lý Rank-Nullity)")
        else:
            print("  -> Kết quả: ✗ FAILED (Phát hiện sai sót số học)")