from typing import List, Tuple, Optional

# --- Định nghĩa kiểu dữ liệu (Type Hinting) ---
Number = float
Matrix = List[List[Number]]
Vector = List[Number]

def gaussian_eliminate_internal(A: Matrix, b: Vector):
    """
    Hàm nội bộ: Khử Gauss với Partial Pivoting, xử lý cả trường hợp vô nghiệm và vô số nghiệm.
    """

    # 1. Tạo ma trận tăng cường M = [A | b]
    n = len(A)
    m = len(A[0]) if A else 0
    M = [A[i][:] + [b[i]] for i in range(n)]
    s = 0           # Đếm số lần hoán đổi dòng
    eps = 1e-12     # Ngưỡng sai số dấu phẩy động
    pivot_cols = [] # Lưu vị trí các cột chứa chốt
    pivot_rows = [] # Lưu vị trí các dòng chứa chốt
    r = 0 # Con trỏ dòng hiện tại

    # 2. Quá trình khử Gauss biến đổi về dạng bậc thang (Row Echelon Form)
    for k in range(m):
        # Thoát sớm nếu hết dòng để xử lí
        if r >= n: 
            break

        pivot_row = -1 # Dòng chứa pivot lớn nhất
        max_val = 0.0

        for i in range(r, n):
            if abs(M[i][k]) > max_val:
                max_val = abs(M[i][k])
                pivot_row = i
        
        # Bỏ qua nếu cột này phụ thuộc tuyến tính (chốt xấp xỉ 0)
        if max_val < eps:
            continue

        # Hoán đổi dòng để đưa pivot lớn nhất lên trên, giảm sai số làm tròn
        if pivot_row != r:
            M[r], M[pivot_row] = M[pivot_row], M[r]
            s += 1

        pivot_cols.append(k)
        pivot_rows.append(r)

        # Khử các phần tử dưới pivot thành 0
        for i in range(r + 1, n):
            factor = M[i][k] / M[r][k]
            for j in range(k, m + 1):
                M[i][j] -= factor * M[r][j]
        
        r += 1

    free_vars = None # Danh sách biến tự do (nếu có)
    x = None # Nghiệm duy nhất (nếu có)

    # 3. Kiểm tra vô nghiệm: Nếu có dòng nào sau khi khử mà tất cả hệ số đều là 0 nhưng hằng số tự do khác 0, thì hệ vô nghiệm
    for i in range(n):
        all_zero = True # Kiểm tra nếu tất cả hệ số của dòng i đều là 0

        for j in range(m):
            if abs(M[i][j]) > eps:
                all_zero = False
                break

        # Trả về ma trận đã khử, không có nghiệm riêng, số lần hoán đổi dòng
        if all_zero and abs(M[i][-1]) > eps:
            return M, None, s, None, None
        
    # 4. Kiểm tra vô số nghiệm: Nếu số cột chứa pivot nhỏ hơn số ẩn, thì có biến tự do => vô số nghiệm
    if len(pivot_cols) < m:
        # Xác định các biến tự do (các cột không chứa pivot)
        free_vars = [j for j in range(m) if j not in pivot_cols]

        # Đưa về dạng RREF (Bậc thang rút gọn) để dễ trích xuất nghiệm tổng quát
        for i in range(len(pivot_rows) - 1, -1, -1):
            row = pivot_rows[i]
            pivot = pivot_cols[i]

            # Chuẩn hóa dòng pivot để hệ số pivot trở thành 1
            if abs(M[row][pivot]) > eps:
                factor = M[row][pivot]
                for j in range(m + 1):
                    M[row][j] /= factor

            # Khử các phần tử trên pivot thành 0
            for r in range(row):
                if abs(M[r][pivot]) > eps:
                    factor = M[r][pivot]
                    for j in range(m + 1):
                        M[r][j] -= factor * M[row][j]

        # Trả về ma trận đã được đưa về dạng RREF, không có nghiệm riêng, số lần hoán đổi dòng
        return M, None, s, pivot_cols, free_vars

    # 5. Trường hợp nghiệm duy nhất: Gom ma trận tam giác trên U để thế ngược
    U = []
    c = []
    for i in range(n):
        if any(abs(M[i][j]) > eps for j in range(m)):
            U.append(M[i][:m])
            c.append(M[i][m])

    if len(U) == m: 
        x = back_substitution(U, c)

    # Trả về ma trận đã khử, nghiệm duy nhất, số lần hoán đổi dòng
    return M, x, s, pivot_cols, free_vars

def gaussian_eliminate(A: Matrix, b: Vector) -> Tuple[Matrix, Optional[Vector], int]:
    """
    Hàm khử Gauss với Partial Pivoting, xử lý cả trường hợp vô nghiệm và vô số nghiệm. 
    Trả về ma trận tăng cường đã được khử, nghiệm riêng (nếu có), số lần hoán đổi dòng.
    """

    try:
        M, x, s, _, _ = gaussian_eliminate_internal(A, b)
        return M, x, s
    except ValueError as e:
        print("\n" + str(e))
        return None, None, s

def back_substitution(U: Matrix, c: Vector) -> Optional[Vector]:
    """
    Thực hiện thế ngược để giải hệ Ux = c, trong đó U là ma trận tam giác trên.
    Trả về nghiệm x nếu có, hoặc None nếu hệ vô nghiệm.
    """

    n = len(U)
    x = [0.0] * n

    # Bắt đầu từ dòng dưới cùng lên trên
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range (i + 1, n):
            s += U[i][j] * x[j]

        if abs(U[i][i]) < 1e-12:
            return None # Hệ suy biến, không có nghiệm duy nhất
        
        x[i] = (c[i] - s) / U[i][i]

    return x

def general_solution(A: Matrix, b: Vector) -> Tuple[Optional[Vector], Optional[List[Vector]], Optional[List[int]]]:
    """
    Tìm nghiệm tổng quát của hệ Ax = b (Bao gồm nghiệm riêng particular và không gian nghiệm nullspace).
    Trả về: Nghiệm riêng, Cơ sở không gian nghiệm, Danh sách biến tự do.
    """

    eps = 1e-12
    M, x, s, pivot_cols, free_vars = gaussian_eliminate_internal(A, b)
    
    # 1. Trường hợp vô nghiệm hoặc nghiệm duy nhất
    if free_vars is None:
        return None, None, None
    
    # 2. Trường hợp vô số nghiệm
    # Trích xuất không gian nghiệm
    n = len(A)
    m = len(A[0])
    pivot_rows = []

    # Xác định các dòng chứa pivot
    for i in range(n):
        for j in pivot_cols:
            if abs(M[i][j]) > eps:
                pivot_rows.append(i)
                break
    
    # Xây dựng nghiệm riêng bằng cách đặt tất cả biến tự do = 0
    particular = [0.0] * m
    for idx, pc in enumerate(pivot_cols):
        row = pivot_rows[idx]
        particular[pc] = M[row][m]

    # Xây dựng cơ sở không gian nghiệm
    nullspace = []
    for free_idx, free_var in enumerate(free_vars):
        basis_vector = [0.0] * m
        basis_vector[free_var] = 1.0

        # Tính các thành phần của vector cơ sở tương ứng với biến tự do này
        for idx, pc in enumerate(pivot_cols):
            row = pivot_rows[idx]
            coeff = M[row][free_var]

            # Nếu hệ số này không phải là 0, thì nó sẽ đóng góp vào vector cơ sở
            if abs(coeff) > eps:
                basis_vector[pc] = -coeff
            else:
                basis_vector[pc] = 0.0

        nullspace.append(basis_vector)

    # Trả về nghiệm riêng, cơ sở không gian nghiệm, và danh sách biến tự do
    return particular, nullspace, free_vars

def verify_solution(A: Matrix, x: Vector, b: Vector, atol: float = 1e-8) -> int:
    """
    Xác minh lại độ chính xác của vector nghiệm x bằng thư viện NumPy. 
    Trả về: 0 nếu nghiệm chính xác
            1 nếu nghiệm gần đúng
            -1 nếu nghiệm sai lệch lớn.
    """

    import numpy as np
    # Chuyển đổi A, x, b sang numpy array để tính toán
    A_np, x_np, b_np = np.array(A, dtype=float), np.array(x, dtype=float), np.array(b, dtype=float)

    # Tính chuẩn vô cùng của phần dư
    r = A_np @ x_np - b_np
    max_residual = np.max(np.abs(r))
    
    print(f"  -> [NumPy Verify] Residual norm ||Ax - b||_inf: {max_residual:.2e}")
    
    if max_residual < atol:
        print("  -> [NumPy Verify] Kết quả: NGHIỆM CHÍNH XÁC")
        return 0
    elif max_residual < 10 * atol:
        print("  -> [NumPy Verify] Kết quả: NGHIỆM GẦN ĐÚNG")
        return 1
    else:
        print("  -> [NumPy Verify] Kết quả: NGHIỆM SAI LỆCH LỚN")
        return -1

if __name__ == "__main__":
    print("\n" + "="*80)
    print("KIỂM THỬ THUẬT TOÁN KHỬ GAUSS VỚI PARTIAL PIVOTING")
    print("="*80)

    test_cases = [
        (
            "Test 1 (Normal): Hệ phương trình có nghiệm duy nhất với nghiệm kỳ vọng: x = [2, 3, -1]",
            [[2, 1, -1], 
             [-3, -1, 2], 
             [-2, 1, 2]], 
            [8, -11, -3]  
        ),
        (
            "Test 2 (Zero Pivot): A[0][0] = 0, cần Partial Pivoting",
            [[0, 2, 1], 
             [1, -2, -3], 
             [-1, 1, 2]], 
            [-8, 0, 3]
        ),
                (
            "Test 3 (Overdetermined có nghiệm): Hệ phương trình có nhiều phương trình hơn số ẩn với nghiệm kỳ vọng: x = [1, 2]",
            [[1, 1],
             [2, -1],
             [3, 0]],
            [3, 0, 3]
        ),
        (
            "Test 4 (No Solution): Hệ vô nghiệm",
            [[1, 1, 1],
             [2, 2, 2],
             [1, -1, 1]
            ],
            [6, 10, 2]
        ),
        (
            "Test 5 (Overdetermined vô nghiệm): Hệ phương trình có nhiều phương trình hơn số ẩn và vô nghiệm",
            [[2, 1, -1],
             [1, -1, 2],
             [3, 2, -1],
             [1, 3, 1]],
            [4, 5, 7, 10]
        ),
        (
            "Test 6 (Infinite Solutions): Hệ có vô số nghiệm với nghiệm kỳ vọng: x = [4 - t, 2, t] (t là biến tự do)",
            [[1, 1, 1],
             [2, 2, 2],
             [1, -1, 1]
            ],
            [6, 12, 2]
        ),
        (
            "Test 7 (Underdetermined): Hệ phương trình có ít hơn số ẩn với nghiệm kỳ vọng: x = [4 - t, 2, t] (t là biến tự do)",
            [[1, 1, 1],
             [1, -1, 1]],
            [6, 2]
        ),
        (
            "Test 8 (Ill-conditioned): Ma trận Hilbert 4x4 với nghiệm kỳ vọng: x = [1, 1, 1, 1]",
            [[1, 1/2, 1/3, 1/4],
             [1/2, 1/3, 1/4, 1/5],
             [1/3, 1/4, 1/5, 1/6],
             [1/4, 1/5, 1/6, 1/7]],
            [1 + 0.5 + 1/3 + 1/4,
             0.5 + 1/3 + 1/4 + 0.2,
             1/3 + 1/4 + 0.2 + 1/6,
             1/4 + 0.2 + 1/6 + 1/7]
        ),
                (
            "Test 9 (Near Singular): Ma trận gần suy biến với nghiệm kỳ vọng: x = [1, 2, 3, 4, 5]",
            [[1, 1, 1, 1, 1],
             [1, 1, 1, 1, 2],
             [1, 1, 1, 2, 2],
             [1, 1, 2, 2, 2],
             [1, 2, 2, 2, 2]],
            [15, 20, 24, 27, 29]
        ),
        (
            "Test 10 (Small Pivot): Ma trận có nhiều phần tử đường chéo rất nhỏ với nghiệm kỳ vọng: x = [1, 1, 1, 1, 1]",
            [[1e-10, 1, 0, 0, 0],
             [1, 1e-10, 1, 0, 0],
             [0, 1, 1e-10, 1, 0],
             [0, 0, 1, 1e-10, 1],
             [0, 0, 0, 1, 1e-10]],
             [1e-10 + 1, 1 + 1e-10, 1e-10 + 1, 1 + 1e-10, 1e-10 + 1]
        ),
    ]
    
    for title, A, b in test_cases:
        print(f"\n{title}")
        _, x, _ = gaussian_eliminate(A, b)
        
        # Nếu có nghiệm duy nhất -> Verify bằng Numpy
        if x is not None and len(x) > 0:
            print(f"  -> Nghiệm tìm được: {[round(val, 4) for val in x]}")
            result = verify_solution(A, x, b)
            
            if result == 0:
                print("  -> Kết luận: NGHIỆM CHÍNH XÁC")
            elif result == 1:
                print("  -> Kết luận: NGHIỆM GẦN ĐÚNG")
            else:
                print("  -> Kết luận: NGHIỆM SAI LỆCH LỚN")
                
        # Nếu vô nghiệm hoặc vô số nghiệm
        else:
            particular, nullspace, free_vars = general_solution(A, b)
            if particular is None and nullspace is None:
                print("  -> Kết luận: HỆ VÔ NGHIỆM")
            else:
                print("  -> Kết luận: HỆ VÔ SỐ NGHIỆM")
                print(f"  -> Nghiệm riêng: {[round(val, 4) for val in particular]}")
                var_names = [f"x{v+1}" for v in free_vars]
                print(f"  -> Biến tự do: {', '.join(var_names)}")