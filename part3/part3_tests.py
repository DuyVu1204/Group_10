"""
part3_tests.py — Hệ thống kiểm thử cho Phần 3 (Đáp ứng yêu cầu 4.3)
════════════════════════════════════════════════════════════════════════════════
Mỗi hàm được kiểm tra qua ít nhất 5 test cases đặc biệt.
"""

import sys
import os
import math

# Đảm bảo import được solvers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import solvers

def print_result(name, A, b, x, expected_x=None):
    print(f"\n>>> TEST CASE: {name}")
    print(f"  A = {A}")
    print(f"  b = {b}")
    if x is None:
        print("  Ket qua: KHONG CO NGHIEM (hoac loi)")
    else:
        print(f"  Nghiem tim duoc: {[round(val, 6) for val in x]}")
        if expected_x:
            print(f"  Nghiem mong doi: {expected_x}")
            # Tinh sai so tuyet doi
            err = max(abs(x[i] - expected_x[i]) for i in range(len(x)))
            if err < 1e-4: # Tolerance higher for some methods
                print("  => TRANG THAI: PASSED [OK]")
            else:
                print(f"  => TRANG THAI: FAILED [X] (Sai so: {err:.2e})")

def run_tests():
    # 1. Ma tran don vi (Identity) - SDD tuyet doi
    A1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    b1 = [1, 2, 3]
    is_sdd = solvers.is_strictly_diagonally_dominant(A1)
    x1, iters = solvers.gauss_seidel(A1, b1)
    print(f"\n>>> TEST CASE 1: Ma tran don vi (SDD)")
    print(f"  Check SDD: {is_sdd}")
    print_result("Gauss-Seidel", A1, b1, x1, [1, 2, 3])
    print(f"  Hoi tu sau {iters} vong lap.")

    # 2. Ma tran SDD 3x3 - Hoi tu tieu chuan
    A2 = [[4, 1, 1], [1, 5, 2], [1, 2, 4]]
    b2 = [6, 8, 7]
    is_sdd = solvers.is_strictly_diagonally_dominant(A2)
    x2, iters = solvers.gauss_seidel(A2, b2)
    print(f"\n>>> TEST CASE 2: Ma tran SDD 3x3 (Standard)")
    print(f"  Check SDD: {is_sdd}")
    print_result("Gauss-Seidel", A2, b2, x2, [1, 1, 1])
    print(f"  Hoi tu sau {iters} vong lap.")

    # 3. Ma tran SPD nhung KHONG chéo trội (Weakly dominant)
    A3 = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]
    b3 = [1, 0, 1]
    is_sdd = solvers.is_strictly_diagonally_dominant(A3)
    x3, iters = solvers.gauss_seidel(A3, b3)
    print(f"\n>>> TEST CASE 3: Ma tran SPD (Khong SDD)")
    print(f"  Check SDD: {is_sdd}")
    print_result("Gauss-Seidel", A3, b3, x3, [1, 1, 1])
    print(f"  Hoi tu sau {iters} vong lap.")

    # 4. Ma tran Phan ky (Divergent) 
    A4 = [[1, 2], [2, 1]]
    b4 = [3, 3]
    is_sdd = solvers.is_strictly_diagonally_dominant(A4)
    x4, iters = solvers.gauss_seidel(A4, b4, max_iter=100)
    print(f"\n>>> TEST CASE 4: Ma tran Phan ky (Explosion)")
    print(f"  Check SDD: {is_sdd}")
    print_result("Gauss-Seidel", A4, b4, x4) 

    # 5. Ma tran co so 0 tren duong cheo (Edge Case 1)
    A5 = [[0, 1], [1, 2]]
    b5 = [1, 3]
    is_sdd = solvers.is_strictly_diagonally_dominant(A5)
    print(f"\n>>> TEST CASE 5: Zero on Diagonal")
    x5, iters = solvers.gauss_seidel(A5, b5)
    if iters < 100:
        print("  Ket qua: Phat hien so 0 tren duong cheo -> Dung thuat toan (Dung ky vong)")

    # 6. Ma tran 1x1 (Edge Case 2)
    A6 = [[5]]
    b6 = [25]
    is_sdd = solvers.is_strictly_diagonally_dominant(A6)
    x6, iters = solvers.gauss_seidel(A6, b6)
    print(f"\n>>> TEST CASE 6: Ma tran 1x1")
    print(f"  Check SDD: {is_sdd}")
    print_result("Gauss-Seidel", A6, b6, x6, [5.0])

    # 7. Ma tran co he so am nhung SDD 
    A7 = [[-10, 2], [1, -5]]
    b7 = [-8, -4]
    is_sdd = solvers.is_strictly_diagonally_dominant(A7)
    x7, iters = solvers.gauss_seidel(A7, b7)
    print(f"\n>>> TEST CASE 7: Ma tran SDD voi he so am")
    print(f"  Check SDD: {is_sdd}")
    print_result("Gauss-Seidel", A7, b7, x7, [1, 1])

    # 8. Ma tran Suy bien (SDD fails & No solution)
    A8 = [[1, 1], [1, 1]]
    b8 = [2, 3] # Vo nghiem
    is_sdd = solvers.is_strictly_diagonally_dominant(A8)
    x8, iters = solvers.gauss_seidel(A8, b8, max_iter=50)
    print(f"\n>>> TEST CASE 8: Ma tran suy bien (Singular)")
    print(f"  Check SDD: {is_sdd}")
    print_result("Gauss-Seidel", A8, b8, x8)

if __name__ == "__main__":
    print("="*60)
    print("BAT DAU KIEM THU PHAN 3 (PURE PYTHON)")
    print("="*60)
    run_tests()
    print("\n" + "="*60)
    print("KIEM THU HOAN TAT")

