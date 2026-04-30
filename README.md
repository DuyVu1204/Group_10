# Group_10

# ĐỒ ÁN 1 Ma Trận và Cơ Sở của Tính Toán Khoa Học

## Ngôn ngữ & Thành phần
- Jupyter Notebook: 65%
- Python: 35%

---

```text
Group_10/
├── README.md                  
├── requirements.txt                 
├── report.pdf             
├── part1/                
│   ├── gaussian.py              
│   ├── determinant.py     
│   ├── inverse.py         
│   ├── rank_basis.py                                                    
│   └── users/                
├── part2/                   
│   ├── decomposition.py                  
│   ├── diagonalization.py              
│   ├── manim_scene.py        
│   └── demo_video.mp4 
├── part3/                   
    ├── analysis.ipynb                  
    ├── benchmark.py     
    ├── part3_tests.py
    └── solvers.py        
```

## Cài đặt
1. Clone repo:
   git clone https://github.com/DuyVu1204/Group_10.git
   
2. Python 3.11 trở lên
## 3. Tạo môi trường ảo và cài đặt

### 🔹 Sử dụng venv

* Tạo môi trường ảo:

```bash
python -m venv venv
```

* Kích hoạt môi trường ảo (Windows):

```bash
venv\Scripts\activate
```

* Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

---

### 🎬 Chạy file `manim_scene.py`

* Cần cài LaTeX (khuyến nghị dùng MiKTeX trên Windows):
  https://miktex.org/download

* Sau khi cài, kiểm tra:

```bash
latex --version
```

---

### ▶️ Chạy chương trình

```bash
manim -pql part2/manim_scene.py SVDFullVisualization
```

---

### 📁 Kết quả

Video sẽ được tạo tại:

```
media/videos/manim_scene/480p15/
```

---

### ⚠️ Lưu ý

* Nếu lỗi LaTeX → kiểm tra lại MiKTeX đã cài đúng
* Nếu không thấy video → kiểm tra đúng tên file và Scene
* `-p`: tự mở video sau khi render
* `-ql`: render nhanh (low quality)

---

