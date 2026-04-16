"""
benchmark.py — Đo kiểm hiệu năng và ổn định (Phiên bản đồng bộ Part 3)
════════════════════════════════════════════════════════════════════════════════
- Tạo ma trận Hilbert (ill-conditioned)
- Tạo ma trận random SPD (well-conditioned)
- Đo thời gian thực thi (5 lần chạy)
- Đo sai số tương đối ||Ax - b|| / ||b||
"""

import time
import numpy as np
import solvers

def generate_hilbert(n):
    return [[1.0 / (i + j + 1) for j in range(n)] for i in range(n)]

def generate_random_spd(n):
    """Tạo ma trận SPD ngẫu nhiên theo yêu cầu đề bài."""
    A = np.random.randn(n, n)
    # SPD = A*A.T + n*I (Đảm bảo xác định dương)
    A_spd = np.dot(A, A.T) + n * np.eye(n)
    return A_spd.tolist()

def get_relative_error(A, x, b):
    """Tính sai số tương đối (Dùng numpy để đo đạc chuẩn xác)."""
    A_np = np.array(A)
    x_np = np.array(x)
    b_np = np.array(b)
    residual = np.dot(A_np, x_np) - b_np
    return np.linalg.norm(residual) / (np.linalg.norm(b_np) + 1e-15)

def benchmark_method(method_func, A, b, repeats=5):
    """Đo thời gian trung bình và sai số của một phương pháp giải."""
    times = []
    x = None
    for i in range(repeats):
        start = time.time()
        x = method_func(A, b)
        times.append(time.time() - start)
    
    avg_time = sum(times) / repeats
    
    # Lấy x để tính sai số (x có thể là tuple (x, iter) cho GS)
    x_final = x[0] if isinstance(x, tuple) else x
    
    error = get_relative_error(A, x_final, b) if x_final is not None else float('nan')
    return avg_time, error

def run_performance_suite(sizes):
    """Thực hiện thí nghiệm trên các kích thước n yêu cầu."""
    results = {
        'Gauss': {'time': [], 'error': []},
        'SVD': {'time': [], 'error': []},
        'Gauss-Seidel': {'time': [], 'error': []}
    }
    
    for n in sizes:
        print(f"Testing n={n} ...")
        A = generate_random_spd(n)
        # Tạo nghiệm x_exact ngẫu nhiên để suy ra vector b
        x_exact = np.random.randn(n).tolist()
        b = np.dot(np.array(A), np.array(x_exact)).tolist()
        
        # 1. Khử Gauss
        t, e = benchmark_method(solvers.gaussian_eliminate, A, b)
        results['Gauss']['time'].append(t)
        results['Gauss']['error'].append(e)
        
        # 2. SVD Solver
        t, e = benchmark_method(solvers.svd_solver, A, b)
        results['SVD']['time'].append(t)
        results['SVD']['error'].append(e)
        
        # 3. Gauss-Seidel
        t, e = benchmark_method(solvers.gauss_seidel, A, b)
        results['Gauss-Seidel']['time'].append(t)
        results['Gauss-Seidel']['error'].append(e)
        
    return results

def benchmark_stability_hilbert(sizes=[3, 5, 7, 8, 10, 12]):
    """Phân tích tính ổn định trên ma trận Hilbert."""
    print("\n" + "="*80)
    print("BENCHMARK TRÊN MA TRẬN HILBERT (Ill-conditioned)")
    print(f"{'n':<6} | {'Gauss err':<12} | {'SVD err':<12} | {'GS err':<12} | {'Cond Number':<12}")
    print("-" * 80)
    
    for n in sizes:
        A = generate_hilbert(n)
        x_true = [1.0] * n
        b = np.dot(np.array(A), np.array(x_true)).tolist()
        cond = np.linalg.cond(A)
        
        # Gauss
        try:
            x_g = solvers.gaussian_eliminate(A, b)
            s_g = f"{get_relative_error(A, x_g, b):.2e}" if x_g else "N/A"
        except: s_g = "N/A"
        
        # SVD
        try:
            x_s = solvers.svd_solver(A, b)
            s_s = f"{get_relative_error(A, x_s, b):.2e}" if x_s else "N/A"
        except: s_s = "N/A"
        
        # Gauss-Seidel 
        try:
            x_gs, _ = solvers.gauss_seidel(A, b, max_iter=1000)
            s_gs = f"{get_relative_error(A, x_gs, b):.2e}" if x_gs else "N/A"
        except: s_gs = "N/A"
            
        print(f"{n:<6} | {s_g:<12} | {s_s:<12} | {s_gs:<12} | {cond:<12.2e}")

def benchmark_stability_spd(sizes=[50, 100, 200, 500]):
    """Phân tích tính ổn định trên ma trận SPD ngẫu nhiên."""
    print("\n" + "="*80)
    print("BENCHMARK TRÊN MA TRẬN SPD (Well-conditioned)")
    print(f"{'n':<6} | {'Gauss err':<12} | {'SVD err':<12} | {'GS err':<12}")
    print("-" * 80)
    
    for n in sizes:
        A = generate_random_spd(n)
        x_true = [1.0] * n
        b = np.dot(np.array(A), np.array(x_true)).tolist()
        
        # Lấy sai số các phương pháp
        x_g = solvers.gaussian_eliminate(A, b)
        x_s = solvers.svd_solver(A, b)
        x_gs, _ = solvers.gauss_seidel(A, b)
        
        s_g = f"{get_relative_error(A, x_g, b):.2e}" if x_g else "N/A"
        s_s = f"{get_relative_error(A, x_s, b):.2e}" if x_s else "N/A"
        s_gs = f"{get_relative_error(A, x_gs, b):.2e}" if x_gs else "N/A"
        
        print(f"{n:<6} | {s_g:<12} | {s_s:<12} | {s_gs:<12}")

if __name__ == "__main__":
    benchmark_stability_hilbert()
    benchmark_stability_spd()
