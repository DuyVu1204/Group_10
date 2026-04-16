import math
import numpy as np

def transpose(M):
    """
    Thực hiện phép chuyển vị ma trận: biến hàng thành cột và ngược lại.
    Sử dụng để hỗ trợ các phép toán ma trận thủ công.
    """
    so_hang = len(M)
    so_cot = len(M[0])
    ket_qua = []
    for i in range(so_cot):
        hang_moi = []
        for j in range(so_hang):
            hang_moi.append(M[j][i])
        ket_qua.append(hang_moi)
    return ket_qua

def mat_mult(A, B):
    """
    Thực hiện phép nhân hai ma trận A và B.
    """
    n = len(A)
    m = len(B[0])
    p = len(B)
    C = []
    for i in range(n):
        hang_C = []
        for j in range(m):
            tong = 0.0
            for k in range(p):
                tong += A[i][k] * B[k][j]
            hang_C.append(tong)
        C.append(hang_C)
    return C

def dot_product(v1, v2):
    """
    Tính tích vô hướng của hai vector v1 và v2.
    Sử dụng trong quá trình trực giao hóa Gram-Schmidt.
    """
    tong = 0.0
    for i in range(len(v1)):
        tong += v1[i] * v2[i]
    return tong

def qr_decomposition(A):
    """
    Phân rã ma trận A thành tích của ma trận trực chuẩn Q và ma trận tam giác trên R.
    Sử dụng thuật toán Gram-Schmidt cổ điển
    """
    n = len(A)
    A_t = transpose(A) 
    Q_t = []
    
    for i in range(n):
        u_i = [] 
        for val in A_t[i]:
            u_i.append(val)
            
        # Trừ đi các hình chiếu lên các vector đã trực giao hóa trước đó
        for j in range(i):
            proj_scalar = dot_product(A_t[i], Q_t[j])
            for k in range(n):
                u_i[k] = u_i[k] - (proj_scalar * Q_t[j][k])

        # Tính độ dài (norm) của vector u_i
        tong_binh_phuong = 0.0
        for x in u_i:
            tong_binh_phuong += x * x
        norm_u = math.sqrt(tong_binh_phuong)
        
        # Chuẩn hóa vector để có độ dài bằng 1 
        q_i = []
        if norm_u > 1e-10:
            for x in u_i:
                q_i.append(x / norm_u)
        else:
            for _ in range(n):
                q_i.append(0.0)
        Q_t.append(q_i)
        
    Q = transpose(Q_t)
    R = mat_mult(Q_t, A) # R = Q^T * A vì Q là ma trận trực giao
    return Q, R

def tim_tri_rieng_nxn(A, max_iter=1000, tol=1e-7):
    """
    Tìm các giá trị riêng của ma trận A bằng thuật toán lặp QR.
    Ma trận sẽ hội tụ về dạng tam giác trên, các giá trị riêng nằm trên đường
    """
    n = len(A)
    A_k = []
    for row in A:
        hang_moi = []
        for val in row:
            hang_moi.append(val)
        A_k.append(hang_moi)
    
    for _ in range(max_iter):
        Q, R = qr_decomposition(A_k)
        A_k_next = mat_mult(R, Q) # A_{k+1} = R_k * Q_k
        
        # Kiểm tra điều kiện hội tụ
        converged = True
        for i in range(1, n):
            for j in range(i):
                if abs(A_k_next[i][j]) > tol:
                    converged = False
                    break
            if not converged:
                break
                
        A_k = A_k_next
        if converged:
            break
            
    # Các giá trị trên đường chéo chính là giá trị riêng 
    eigenvalues = []
    for i in range(n):
        eigenvalues.append(A_k[i][i])
    return eigenvalues

def tim_vector_rieng(A, lamda):
    """
    Tìm vector riêng tương ứng với giá trị riêng lamda bằng cách giải 
    hệ phương trình thuần nhất (A - lamda*I)v = 0 qua phép khử Gauss RREF.
    """
    n = len(A)
    # Tạo ma trận M = A - lamda*I
    M = []
    for i in range(n):
        hang = []
        for j in range(n):
            if i == j:
                hang.append(A[i][j] - lamda)
            else:
                hang.append(A[i][j])
        M.append(hang)

    # Đưa M về dạng bậc thang dòng rút gọn 
    ma_tran_rut_gon, cac_cot_chot = rref(M)

    # Tìm một cột không phải cột chốt để chọn làm biến tự do 
    cot_tu_do = -1
    for j in range(n):
        da_co_chot = False
        for chot in cac_cot_chot:
            if j == chot:
                da_co_chot = True
                break
        if not da_co_chot:
            cot_tu_do = j
            break

    v = [0.0] * n
    if cot_tu_do == -1: # Trường hợp không tìm thấy biến tự do
        return v

    # Gán biến tự do = 1.0 và tính các biến còn lại từ dạng RREF
    v[cot_tu_do] = 1.0
    for idx in range(len(cac_cot_chot)):
        index_hang = idx
        index_cot_chot = cac_cot_chot[idx]
        if index_hang < n:
            he_so = ma_tran_rut_gon[index_hang][cot_tu_do]
            v[index_cot_chot] = -he_so # Nghiệm x_i = -he_so * x_tu_do
            
    return v

def rref(matrix):
    """
    Biến đổi ma trận về dạng bậc thang dòng rút gọn (RREF).
    Đây là kỹ thuật cốt lõi để xác định hạng và tìm cơ sở không gian nghiệm
    """
    M = [row[:] for row in matrix] # Tạo bản sao ma trận
    so_hang = len(M)
    so_cot = len(M[0])
    
    hang_dang_xet = 0
    danh_sach_cot_chot = []
    sai_so = 1e-7  

    for cot in range(so_cot):
        if hang_dang_xet >= so_hang:
            break

        # Tìm dòng có giá trị lớn nhất tại cột đang xét để làm chốt 
        hang_chot = hang_dang_xet
        while hang_chot < so_hang and abs(M[hang_chot][cot]) < sai_so:
            hang_chot += 1

        if hang_chot == so_hang: # Không tìm thấy chốt ở cột này
            continue

        danh_sach_cot_chot.append(cot)
        
        # Hoán đổi dòng hiện tại với dòng chốt
        M[hang_dang_xet], M[hang_chot] = M[hang_chot], M[hang_dang_xet]
        
        # Chuẩn hóa dòng chốt để phần tử chốt bằng 1 
        gia_tri_chot = M[hang_dang_xet][cot]
        for j in range(so_cot):
            M[hang_dang_xet][j] = M[hang_dang_xet][j] / gia_tri_chot

        # Khử các phần tử khác 0 ở cột chốt của các dòng còn lại 
        for i in range(so_hang):
            if i != hang_dang_xet:
                he_so_triet_tieu = M[i][cot]
                for j in range(so_cot):
                    M[i][j] = M[i][j] - (he_so_triet_tieu * M[hang_dang_xet][j])

        hang_dang_xet += 1

    return M, danh_sach_cot_chot

def cheo_hoa_ma_tran(A, danh_sach_tri_rieng):
    """
    Thực hiện phân tích chéo hóa A = PDP^-1.
    Trả về ma trận P (chứa vector riêng ở các cột) và ma trận đường chéo D
    """
    n = len(A)
    cac_vector_rieng = []

    # Tìm vector riêng cho từng giá trị riêng
    for lamda in danh_sach_tri_rieng:
        v = tim_vector_rieng(A, lamda)
        cac_vector_rieng.append(v)

    # Xây dựng ma trận P bằng cách xếp các vector riêng vào từng cột
    P = [[0.0 for _ in range(n)] for _ in range(n)]
    for cot_idx in range(n):
        for hang_idx in range(n):
            P[hang_idx][cot_idx] = cac_vector_rieng[cot_idx][hang_idx]

    # Xây dựng ma trận đường chéo D
    D = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        D[i][i] = danh_sach_tri_rieng[i]
    
    return P, D

def main():
    test_cases = [
        {
            "ten": "Ma trận 2x2",
            "A": [[2, 1], [1, 2]]
        },
        {
            "ten": "Ma trận 2x2",
            "A": [[1, -1], [2, 4]]
        },
        {
            "ten": "Ma trận 3x3",
            "A": [[4, 1, -1], [2, 5, -2], [1, 1, 2]]
        },
        {
            "ten": "Ma trận 4x4",
            "A": [
                [5, 2, 0, 0],
                [2, 5, 0, 0],
                [0, 0, 4, 1],
                [0, 0, 1, 4]
            ]
        },
        {
            "ten": "Ma trận 5x5",
            "A": [
                [6, 1, 0, 0, 0],
                [1, 6, 1, 0, 0],
                [0, 1, 6, 1, 0],
                [0, 0, 1, 6, 1],
                [0, 0, 0, 1, 6]
            ]
        },

        {"ten": "Ma tran suy bien", "A": [[1, 2], [2, 4]]},
        {"ten": "Ma tran don vi", "A": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        {"ten": "Ma tran Hilbert H3", "A": [[1, 0.5, 0.3333], [0.5, 0.3333, 0.25], [0.3333, 0.25, 0.2]]},
        {"ten": "Ma tran co tri rieng lap", "A": [[3, 1], [0, 3]]}

    ]



    for idx, case in enumerate(test_cases):
        A = case["A"]
        ten = case["ten"]
        n = len(A)
        
        print("="*50)
        print(f"TEST CASE {idx + 1}: {ten}")
        print("="*50)
        print("Ma trận A:")
        for row in A: print(f"  {row}")

        try:
            # 1. Tìm giá trị riêng bằng thuật toán QR 
            eigenvalues = tim_tri_rieng_nxn(A)
            ev_print = [round(val, 4) for val in eigenvalues]
            print(f"\n> Giá trị riêng (QR Manual): {ev_print}")

            # 2. Thực hiện chéo hóa 
            P_mat, D_mat = cheo_hoa_ma_tran(A, eigenvalues)
            
            # 3. Kiểm chứng bằng NumPy 
            A_np = np.array(A)
            P_np = np.array(P_mat)
            D_np = np.array(D_mat)
            
            # Tính AP và PD để so sánh
            AP = np.dot(A_np, P_np)
            PD = np.dot(P_np, D_np)

            # In kết quả P và D 
            print("\nMa trận P (Các vector riêng):")
            for row in P_mat: print(f"  {[round(x, 4) for x in row]}")
            
            print("\nMa trận D (Đường chéo):")
            for row in D_mat: print(f"  {[round(x, 4) for x in row]}")

            # 4. Kết luận
            if np.allclose(AP, PD, atol=1e-4):
                print(f"\n=> THÀNH CÔNG (AP ≈ PD)")
            else:
                print(f"\n=> THẤT BẠI (Sai lệch quá lớn)")
                
        except Exception as e:
            print(f"\nKhông thể xử lý test case này. Chi tiết: {e}")
        
        print("\n")

if __name__ == "__main__":
    main()