from manim import *
import numpy as np

# ════════════════════════════════════════════════════════════════
COLOR_A = "#F0F0F0"
COLOR_U = "#03A9F4"
COLOR_SIGMA = "#FFCA28"
COLOR_VT = "#E91E63"
COLOR_SUBTITLE = "#DDDDDD"
COLOR_P = "#66BB6A"
COLOR_D = "#FFCA28"
VIETNAMESE_FONT = "Arial"


class Subtitle(Text):
    def __init__(self, text, **kwargs):
        super().__init__(text, font=VIETNAMESE_FONT, font_size=20,
                         color=COLOR_SUBTITLE, line_spacing=0.8, **kwargs)
        self.to_edge(DOWN, buff=0.5)


# ════════════════════════════════════════════════════════════════
# SCENE 1: GIỚI THIỆU
# ════════════════════════════════════════════════════════════════
class SVDIntroScene(Scene):
    def construct(self):
        title = Text("Phân rã ma trận và Trực quan hóa", font=VIETNAMESE_FONT, font_size=48)
        sub_title = Text("Singular Value Decomposition (SVD)", font_size=32, color=BLUE_B).next_to(title, DOWN)
        intro_grp = VGroup(title, sub_title).center()
        sub1 = Subtitle("Trong phần này, nhóm chúng em lựa chọn kỹ thuật SVD...")
        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(sub_title, shift=UP), FadeIn(sub1))
        self.wait(2)
        sub2 = Subtitle("...một trong những công cụ mạnh mẽ nhất của đại số tuyến tính hiện đại.")
        self.play(Transform(sub1, sub2))
        self.wait(2.5)
        self.play(FadeOut(intro_grp, shift=UP), FadeOut(sub1))

        sub3 = Subtitle("Xét ma trận thực A có kích thước m x n.")
        formula_a = MathTex("A", "_{m \\times n}", color=COLOR_A).scale(1.5).move_to(ORIGIN)
        self.play(Write(formula_a), FadeIn(sub3))
        self.wait(1.5)
        full_formula = MathTex(
            "A", "_{m \\times n}", "=",
            "U", "_{m \\times m}", "\\Sigma", "_{m \\times n}", "V^T", "_{n \\times n}"
        ).scale(1.2)
        full_formula[0].set_color(COLOR_A); full_formula[1].set_color(COLOR_A).scale(0.6)
        full_formula[3].set_color(COLOR_U); full_formula[4].set_color(COLOR_U).scale(0.6)
        full_formula[5].set_color(COLOR_SIGMA); full_formula[6].set_color(COLOR_SIGMA).scale(0.6)
        full_formula[7].set_color(COLOR_VT); full_formula[8].set_color(COLOR_VT).scale(0.6)
        self.play(TransformMatchingTex(formula_a, full_formula))
        self.wait(2)
        self.play(full_formula.animate.to_edge(UP, buff=1.2))
        summary_items = VGroup(
            Text("• U : Ma trận trực giao trái (m x m)", font=VIETNAMESE_FONT, font_size=20, color=COLOR_U),
            Text("• Σ : Ma trận co giãn (m x n)", font=VIETNAMESE_FONT, font_size=20, color=COLOR_SIGMA),
            Text("• Vᵀ: Ma trận trực giao phải (n x n)", font=VIETNAMESE_FONT, font_size=20, color=COLOR_VT)
        ).arrange(DOWN, aligned_edge=LEFT).next_to(full_formula, DOWN, buff=1.0)
        sub_final = Subtitle("Hãy cùng thực hiện tính toán chi tiết qua một ví dụ cụ thể.")
        self.play(FadeIn(summary_items, shift=UP), Transform(sub3, sub_final))
        self.wait(3)
        self.play(FadeOut(VGroup(full_formula, summary_items, sub3)))


# ════════════════════════════════════════════════════════════════
# SCENE 2: TÍNH TOÁN SVD — A = [[5,0],[4,3]]
# ════════════════════════════════════════════════════════════════
class SVDStepByStepScene(Scene):
    def construct(self):
        # ─── SLIDE 1: Giới thiệu A + Tại sao SVD ───
        sub = Subtitle("Ví dụ: Phân rã SVD cho ma trận A vuông 2x2 (không đối xứng).")
        self.play(FadeIn(sub))
        a_mat = MathTex(
            "A", "=", "\\begin{bmatrix} 5 & 0 \\\\ 4 & 3 \\end{bmatrix}", color=COLOR_A
        ).scale(1.3)
        self.play(Write(a_mat))
        self.wait(2)

        sub1b = Subtitle("Ma trận này sẽ được dùng lại cho phần chéo hóa ở cuối video.")
        self.play(Transform(sub, sub1b))
        self.wait(2)
        self.play(FadeOut(a_mat))

        # ─── Tại sao cần SVD ───
        sub_why = Subtitle("Tại sao cần SVD? Vì SVD phân tích mọi phép biến đổi thành 3 bước đơn giản.")
        self.play(Transform(sub, sub_why))
        why1 = MathTex("A\\vec{x}", "=", "U", "\\cdot", "\\Sigma", "\\cdot", "V^T", "\\vec{x}").scale(1.2)
        why1[2].set_color(COLOR_U); why1[4].set_color(COLOR_SIGMA); why1[6].set_color(COLOR_VT)
        self.play(Write(why1))
        self.wait(2)

        step_labels = VGroup(
            VGroup(MathTex("V^T:", color=COLOR_VT).scale(0.8), Text("Xoay", font=VIETNAMESE_FONT, font_size=22, color=COLOR_VT)).arrange(RIGHT, buff=0.1),
            MathTex("\\rightarrow"),
            VGroup(MathTex("\\Sigma:", color=COLOR_SIGMA).scale(0.8), Text("Co giãn", font=VIETNAMESE_FONT, font_size=22, color=COLOR_SIGMA)).arrange(RIGHT, buff=0.1),
            MathTex("\\rightarrow"),
            VGroup(MathTex("U:", color=COLOR_U).scale(0.8), Text("Xoay", font=VIETNAMESE_FONT, font_size=22, color=COLOR_U)).arrange(RIGHT, buff=0.1),
        ).arrange(RIGHT, buff=0.5).scale(0.85).next_to(why1, DOWN, buff=1.0)

        sub_why2 = Subtitle("Bất kỳ ma trận nào cũng có SVD — kể cả không vuông, không khả nghịch.")
        self.play(Transform(sub, sub_why2), FadeIn(step_labels, shift=UP))
        self.wait(3)
        self.play(FadeOut(why1), FadeOut(step_labels))

        # ─── SLIDE 2: A^T A ───
        sub2 = Subtitle("Bước 1: Tính tích A chuyển vị nhân A.")
        self.play(Transform(sub, sub2))
        s1 = MathTex(
            "A^T A", "=",
            "\\begin{bmatrix} 5 & 4 \\\\ 0 & 3 \\end{bmatrix}",
            "\\begin{bmatrix} 5 & 0 \\\\ 4 & 3 \\end{bmatrix}",
            "=", "\\begin{bmatrix} 41 & 12 \\\\ 12 & 9 \\end{bmatrix}"
        ).scale(0.85).move_to(ORIGIN)
        self.play(Write(s1))
        self.wait(3)
        self.play(FadeOut(s1))

        # ─── SLIDE 3: Eigenvalues ───
        sub3 = Subtitle("Bước 2: Giải phương trình đặc trưng để tìm trị riêng.")
        self.play(Transform(sub, sub3))
        eq1 = MathTex("\\det(A^T A - \\lambda I) = 0").scale(0.85)
        eq2 = MathTex("\\det \\begin{bmatrix} 41-\\lambda & 12 \\\\ 12 & 9-\\lambda \\end{bmatrix} = 0").scale(0.8)
        eq3 = MathTex("(41-\\lambda)(9-\\lambda) - 144 = 0").scale(0.8)
        eq4 = MathTex("\\lambda^2 - 50\\lambda + 225 = 0").scale(0.8)
        eq5 = MathTex("(\\lambda - 45)(\\lambda - 5) = 0").scale(0.8)
        result = MathTex(
            "\\lambda_1 = 45,\\; \\lambda_2 = 5",
            "\\;\\Rightarrow\\;",
            "\\sigma_1 = 3\\sqrt{5},\\; \\sigma_2 = \\sqrt{5}", color=COLOR_SIGMA
        ).scale(0.8)
        eqs = VGroup(eq1, eq2, eq3, eq4, eq5, result).arrange(DOWN, buff=0.3).move_to(ORIGIN)
        for e in [eq1, eq2, eq3, eq4, eq5, result]:
            self.play(FadeIn(e, shift=UP * 0.3), run_time=0.7)
            self.wait(0.6)
        self.wait(2)
        self.play(FadeOut(eqs))

        # ─── SLIDE 4: Find V — chi tiết ───
        sub4 = Subtitle("Bước 3: Tìm vector riêng của A chuyển vị A để xây dựng ma trận V.")
        self.play(Transform(sub, sub4))

        # λ₁ = 45
        ev1 = VGroup(
            MathTex("\\lambda_1 = 45:", color=COLOR_SIGMA).scale(0.75),
            MathTex("\\begin{bmatrix} 41-45 & 12 \\\\ 12 & 9-45 \\end{bmatrix} \\vec{v} = \\vec{0}").scale(0.65),
            MathTex("\\begin{bmatrix} -4 & 12 \\\\ 12 & -36 \\end{bmatrix} \\vec{v} = \\vec{0}").scale(0.65),
            MathTex("-4v_1 + 12v_2 = 0 \\;\\Rightarrow\\; v_1 = 3v_2").scale(0.65),
            MathTex("\\vec{v}_1 = \\frac{1}{\\sqrt{10}}\\begin{bmatrix} 3 \\\\ 1 \\end{bmatrix}", color=COLOR_VT).scale(0.7),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)

        # λ₂ = 5
        ev2 = VGroup(
            MathTex("\\lambda_2 = 5:", color=COLOR_SIGMA).scale(0.75),
            MathTex("\\begin{bmatrix} 36 & 12 \\\\ 12 & 4 \\end{bmatrix} \\vec{v} = \\vec{0}").scale(0.65),
            MathTex("36v_1 + 12v_2 = 0 \\;\\Rightarrow\\; v_1 = -\\frac{v_2}{3}").scale(0.65),
            MathTex("\\vec{v}_2 = \\frac{1}{\\sqrt{10}}\\begin{bmatrix} -1 \\\\ 3 \\end{bmatrix}", color=COLOR_VT).scale(0.7),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)

        ev_all = VGroup(ev1, ev2).arrange(RIGHT, buff=0.8, aligned_edge=UP).move_to(UP * 0.8)

        sub4b = Subtitle("Gộp v1, v2 thành các cột của V, lấy chuyển vị để được V chuyển vị.")
        vt_res = MathTex(
            "V", "=", "\\frac{1}{\\sqrt{10}} \\begin{bmatrix} 3 & -1 \\\\ 1 & 3 \\end{bmatrix}",
            "\\;\\Rightarrow\\;",
            "V^T", "=", "\\frac{1}{\\sqrt{10}} \\begin{bmatrix} 3 & 1 \\\\ -1 & 3 \\end{bmatrix}",
        ).scale(0.7)
        vt_res[0].set_color(COLOR_VT); vt_res[4].set_color(COLOR_VT)
        vt_res.next_to(ev_all, DOWN, buff=0.5)

        self.play(FadeIn(ev1, shift=DOWN))
        self.wait(2.5)
        self.play(FadeIn(ev2, shift=DOWN))
        self.wait(2.5)
        self.play(Transform(sub, sub4b), Write(vt_res))
        self.wait(3)
        self.play(FadeOut(ev_all), FadeOut(vt_res))

        # ─── SLIDE 5: Compute U — chi tiết ───
        sub5 = Subtitle("Bước 4: Tính từng cột của U bằng công thức u = A nhân v chia sigma.")
        self.play(Transform(sub, sub5))

        u_formula = MathTex("\\vec{u}_i = \\frac{A \\vec{v}_i}{\\sigma_i}", color=COLOR_U).scale(0.85).move_to(UP * 2.6)

        u1_step1 = MathTex(
            "\\vec{u}_1 = \\frac{1}{3\\sqrt{5}} \\begin{bmatrix} 5&0\\\\4&3 \\end{bmatrix} \\frac{1}{\\sqrt{10}} \\begin{bmatrix} 3\\\\1 \\end{bmatrix}"
        ).scale(0.55)
        u1_step2 = MathTex(
            "= \\frac{1}{3\\sqrt{50}} \\begin{bmatrix} 15\\\\15 \\end{bmatrix} = \\frac{1}{\\sqrt{2}} \\begin{bmatrix} 1\\\\1 \\end{bmatrix}", color=COLOR_U
        ).scale(0.55)
        u1_grp = VGroup(u1_step1, u1_step2).arrange(DOWN, buff=0.15, aligned_edge=LEFT)

        u2_step1 = MathTex(
            "\\vec{u}_2 = \\frac{1}{\\sqrt{5}} \\begin{bmatrix} 5&0\\\\4&3 \\end{bmatrix} \\frac{1}{\\sqrt{10}} \\begin{bmatrix} -1\\\\3 \\end{bmatrix}"
        ).scale(0.55)
        u2_step2 = MathTex(
            "= \\frac{1}{\\sqrt{50}} \\begin{bmatrix} -5\\\\5 \\end{bmatrix} = \\frac{1}{\\sqrt{2}} \\begin{bmatrix} -1\\\\1 \\end{bmatrix}", color=COLOR_U
        ).scale(0.55)
        u2_grp = VGroup(u2_step1, u2_step2).arrange(DOWN, buff=0.15, aligned_edge=LEFT)

        # Đặt 2 khối tính u1, u2 nằm ngang để không đè sub
        u_calcs = VGroup(u1_grp, u2_grp).arrange(RIGHT, buff=0.8, aligned_edge=UP).next_to(u_formula, DOWN, buff=0.4)

        u_mat = MathTex(
            "U", "=", "\\frac{1}{\\sqrt{2}} \\begin{bmatrix} 1&-1\\\\1&1 \\end{bmatrix}", color=COLOR_U
        ).scale(0.75).next_to(u_calcs, DOWN, buff=0.4)

        self.play(Write(u_formula))
        self.wait(1)
        self.play(FadeIn(u1_grp, shift=UP))
        self.wait(2)
        self.play(FadeIn(u2_grp, shift=UP))
        self.wait(2)
        self.play(Write(u_mat))
        self.wait(2)
        self.play(FadeOut(u_formula), FadeOut(u_calcs), FadeOut(u_mat))

        # ─── SLIDE 6: Final SVD ───
        sub6 = Subtitle("Kết quả SVD hoàn chỉnh: A = U Sigma V chuyển vị.")
        self.play(Transform(sub, sub6))
        final = MathTex(
            "\\begin{bmatrix} 5&0\\\\4&3 \\end{bmatrix}", "=",
            "\\frac{1}{\\sqrt{2}}\\begin{bmatrix} 1&-1\\\\1&1 \\end{bmatrix}",
            "\\begin{bmatrix} 3\\sqrt{5}&0\\\\0&\\sqrt{5} \\end{bmatrix}",
            "\\frac{1}{\\sqrt{10}}\\begin{bmatrix} 3&1\\\\-1&3 \\end{bmatrix}"
        ).scale(0.8)
        final[0].set_color(COLOR_A); final[2].set_color(COLOR_U)
        final[3].set_color(COLOR_SIGMA); final[4].set_color(COLOR_VT)
        self.play(Write(final))
        self.wait(4)
        self.play(FadeOut(final), FadeOut(sub))


# ════════════════════════════════════════════════════════════════
# SCENE 3: TRỰC QUAN HÓA HÌNH HỌC
# ════════════════════════════════════════════════════════════════
class SVDGeometricScene(Scene):
    def construct(self):
        # ─── Phần 0: Show A's full effect first ───
        sub = Subtitle("Khi nhân A với hình tròn đơn vị, kết quả là một hình elip.")
        self.play(FadeIn(sub))

        plane = NumberPlane(x_range=[-8, 8, 1], y_range=[-6, 6, 1],
                            background_line_style={"stroke_opacity": 0.2})
        
        # Define v1, v2 based on A = [[5,0],[4,3]]
        ANGLE_V = np.arctan2(1, 3)
        v1_dir = np.array([np.cos(ANGLE_V), np.sin(ANGLE_V), 0])
        v2_dir = np.array([np.cos(ANGLE_V + 90 * DEGREES), np.sin(ANGLE_V + 90 * DEGREES), 0])

        circle = Circle(radius=1, color=WHITE, stroke_width=2).set_fill(BLUE, opacity=0.15)
        v1_arrow = Arrow(ORIGIN, v1_dir, color=RED, buff=0)
        v2_arrow = Arrow(ORIGIN, v2_dir, color=GREEN, buff=0)
        v1_lbl = MathTex("v_1", color=RED).scale(0.6).next_to(v1_arrow.get_end(), UR, buff=0.1)
        v2_lbl = MathTex("v_2", color=GREEN).scale(0.6).next_to(v2_arrow.get_end(), UL, buff=0.1)
        geom = VGroup(circle, v1_arrow, v2_arrow)

        self.play(Create(plane))
        self.play(Create(circle), GrowArrow(v1_arrow), GrowArrow(v2_arrow), Write(v1_lbl), Write(v2_lbl))
        self.wait(2)

        # Apply full A = [[5,0],[4,3]] with SCALE
        mat_A = np.array([[5, 0], [4, 3]])
        SCALE = 0.4
        A_scaled = mat_A * SCALE
        
        # Calculate new positions for labels (A * v)
        v1_end_2d = np.dot(A_scaled, v1_dir[:2])
        v1_pos_new = np.array([v1_end_2d[0], v1_end_2d[1], 0])
        
        v2_end_2d = np.dot(A_scaled, v2_dir[:2])
        v2_pos_new = np.array([v2_end_2d[0], v2_end_2d[1], 0])

        sub2 = Subtitle("Nhân A: Các vector v1, v2 biến thành các trục chính của elip.")
        self.play(Transform(sub, sub2))
        self.play(
            geom.animate.apply_matrix(A_scaled),
            v1_lbl.animate.move_to(v1_pos_new + UR * 0.2), # Di chuyển nhãn tới vị trí mới
            v2_lbl.animate.move_to(v2_pos_new + UL * 0.2),
            run_time=2
        )
        self.wait(2)

        # Reset
        sub3 = Subtitle("SVD giúp ta hiểu phép biến đổi này qua 3 bước: Xoay - Co giãn - Xoay.")
        self.play(Transform(sub, sub3))
        self.wait(1)
        self.play(FadeOut(geom), FadeOut(v1_lbl), FadeOut(v2_lbl))

        # ─── Phần 1: Decompose step by step ───
        # Sidebar
        sigma_disp = MathTex("\\Sigma = \\begin{bmatrix} 3\\sqrt{5}&0\\\\0&\\sqrt{5} \\end{bmatrix}", color=COLOR_SIGMA).scale(0.45)
        u_disp = MathTex("U = \\frac{1}{\\sqrt{2}}\\begin{bmatrix} 1&-1\\\\1&1 \\end{bmatrix}", color=COLOR_U).scale(0.45)
        vt_disp = MathTex("V^T = \\frac{1}{\\sqrt{10}}\\begin{bmatrix} 3&1\\\\-1&3 \\end{bmatrix}", color=COLOR_VT).scale(0.45)
        sidebar = VGroup(vt_disp, sigma_disp, u_disp).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        sidebar.to_corner(UR, buff=0.3)
        bg_box = SurroundingRectangle(sidebar, color=GRAY, buff=0.12, fill_opacity=0.15, fill_color=BLACK)
        self.play(Create(bg_box), FadeIn(sidebar))

        # New circle với các vector v1, v2 thực tế
        circle2 = Circle(radius=1, color=WHITE, stroke_width=2).set_fill(BLUE, opacity=0.15)
        v1_arrow2 = Arrow(ORIGIN, v1_dir, color=RED, buff=0)
        v2_arrow2 = Arrow(ORIGIN, v2_dir, color=GREEN, buff=0)
        v1_lbl_init = MathTex("v_1", color=RED).scale(0.6).next_to(v1_arrow2.get_end(), UR, buff=0.1)
        v2_lbl_init = MathTex("v_2", color=GREEN).scale(0.6).next_to(v2_arrow2.get_end(), UL, buff=0.1)
        
        geom2 = VGroup(circle2, v1_arrow2, v2_arrow2)

        sub4 = Subtitle("Bắt đầu lại với các vector v1, v2 ban đầu.")
        self.play(Transform(sub, sub4))
        self.play(Create(circle2), GrowArrow(v1_arrow2), GrowArrow(v2_arrow2), Write(v1_lbl_init), Write(v2_lbl_init))
        self.wait(2)

        # Step 1: V^T — rotation by -ANGLE_V
        VT_mat = np.array([[3, 1], [-1, 3]]) / np.sqrt(10)
        sub_vt = Subtitle("Bước 1: Nhân VT xoay các vector v1, v2 về các trục chuẩn e1, e2.")
        hl1 = SurroundingRectangle(vt_disp, color=YELLOW, buff=0.08)
        self.play(Transform(sub, sub_vt), Create(hl1))
        self.play(
            geom2.animate.apply_matrix(VT_mat),
            FadeOut(v1_lbl_init), FadeOut(v2_lbl_init),
            run_time=2
        )
        # Gán nhãn trục chuẩn
        e1_res = MathTex("e_1", color=RED).scale(0.6).next_to(v1_arrow2.get_end(), DOWN, buff=0.1)
        e2_res = MathTex("e_2", color=GREEN).scale(0.6).next_to(v2_arrow2.get_end(), LEFT, buff=0.1)
        self.play(Write(e1_res), Write(e2_res))
        self.wait(2)

        # Step 2: Sigma stretch
        s1_val = 3 * np.sqrt(5)
        s2_val = np.sqrt(5)
        sub_sig = Subtitle("Bước 2: Sigma co giãn elip theo các trục chuẩn.")
        hl2 = SurroundingRectangle(sigma_disp, color=YELLOW, buff=0.08)
        self.play(Transform(sub, sub_sig), ReplacementTransform(hl1, hl2))
        
        Sigma_mat = np.array([[s1_val * SCALE, 0], [0, s2_val * SCALE]])
        self.play(
            geom2.animate.apply_matrix(Sigma_mat),
            e1_res.animate.next_to(v1_arrow2.get_end(), DOWN, buff=0.15),
            e2_res.animate.next_to(v2_arrow2.get_end(), LEFT, buff=0.15),
            run_time=2
        )
        sig_lbls = VGroup(
            MathTex("\\sigma_1", color=COLOR_SIGMA).scale(0.6).next_to(e1_res, RIGHT, buff=0.1),
            MathTex("\\sigma_2", color=COLOR_SIGMA).scale(0.6).next_to(e2_res, UP, buff=0.1)
        )
        self.play(Write(sig_lbls))
        self.wait(2)

        # Step 3: U rotation
        U_mat = np.array([[1, -1], [1, 1]]) / np.sqrt(2)
        sub_u = Subtitle("Bước 3: U xoay Elip về vị trí cuối cùng khớp với kết quả nhân A trực tiếp.")
        hl3 = SurroundingRectangle(u_disp, color=YELLOW, buff=0.08)
        self.play(Transform(sub, sub_u), ReplacementTransform(hl2, hl3))
        self.play(
            geom2.animate.apply_matrix(U_mat),
            FadeOut(e1_res), FadeOut(e2_res), FadeOut(sig_lbls),
            run_time=2
        )
        u1_final = MathTex("\\sigma_1 u_1", color=RED).scale(0.65).next_to(v1_arrow2.get_end(), UR, buff=0.1)
        u2_final = MathTex("\\sigma_2 u_2", color=GREEN).scale(0.65).next_to(v2_arrow2.get_end(), UL, buff=0.1)
        self.play(Write(u1_final), Write(u2_final))
        self.wait(3)

        sub_final = Subtitle("Kết quả cuối cùng hoàn toàn thống nhất.")
        self.play(Transform(sub, sub_final), FadeOut(hl3))
        self.wait(4)

        # CLEANUP: fade out EVERYTHING including plane
        self.play(FadeOut(VGroup(geom2, u1_final, u2_final, sidebar, bg_box, plane, sub)))
        self.wait(0.5)


# ════════════════════════════════════════════════════════════════
# SCENE 4: CHÉO HÓA — A = [[5,0],[4,3]]
# ════════════════════════════════════════════════════════════════
class DiagonalizationScene(Scene):
    def construct(self):
        # ─── SLIDE 1: Intro ───
        sub = Subtitle("Chéo hóa ma trận: biến đổi A thành dạng đường chéo.")
        self.play(FadeIn(sub))
        a_mat = MathTex("A", "=", "\\begin{bmatrix} 5 & 0 \\\\ 4 & 3 \\end{bmatrix}").scale(1.2)
        a_mat[0].set_color(COLOR_A)
        diag_eq = MathTex("A", "=", "P", "\\cdot", "D", "\\cdot", "P^{-1}").scale(1.2)
        diag_eq[0].set_color(COLOR_A); diag_eq[2].set_color(COLOR_P)
        diag_eq[4].set_color(COLOR_D); diag_eq[6].set_color(COLOR_P)
        intro_grp = VGroup(a_mat, diag_eq).arrange(DOWN, buff=0.6)
        self.play(Write(a_mat))
        self.wait(1)
        sub2 = Subtitle("Mục tiêu: tìm P và D sao cho A = P nhân D nhân P nghịch đảo.")
        self.play(Transform(sub, sub2), Write(diag_eq))
        self.wait(3)
        self.play(FadeOut(intro_grp))

        # ─── SLIDE 2: Eigenvalue equation ───
        sub3 = Subtitle("Bước 1: Từ phương trình trị riêng, suy ra điều kiện det = 0.")
        self.play(Transform(sub, sub3))
        eq1 = MathTex("A", "\\vec{v}", "=", "\\lambda", "\\vec{v}").scale(1.1)
        eq1[0].set_color(COLOR_A); eq1[3].set_color(COLOR_D)
        eq2 = MathTex("(A - \\lambda I)", "\\vec{v}", "=", "\\vec{0}").scale(1.0)
        eq3_note = Text("Để có nghiệm v khác 0, cần:", font=VIETNAMESE_FONT, font_size=22, color=GRAY).scale(0.9)
        eq3 = MathTex("\\det(A - \\lambda I) = 0", color=YELLOW).scale(1.0)
        eqs = VGroup(eq1, eq2, eq3_note, eq3).arrange(DOWN, buff=0.5)
        self.play(Write(eq1)); self.wait(1.5)
        self.play(Write(eq2)); self.wait(1.5)
        self.play(FadeIn(eq3_note)); self.wait(0.5)
        self.play(Write(eq3)); self.wait(2)
        self.play(FadeOut(eqs))

        # ─── SLIDE 3: Solve ───
        sub4 = Subtitle("Bước 2: Khai triển và giải phương trình đặc trưng.")
        self.play(Transform(sub, sub4))
        d1 = MathTex("\\det \\begin{bmatrix} 5-\\lambda & 0 \\\\ 4 & 3-\\lambda \\end{bmatrix} = 0").scale(0.85)
        d2 = MathTex("(5-\\lambda)(3-\\lambda) - 0 \\cdot 4 = 0").scale(0.8)
        d3 = MathTex("\\lambda^2 - 8\\lambda + 15 = 0").scale(0.8)
        d4 = MathTex("(\\lambda - 5)(\\lambda - 3) = 0").scale(0.8)
        d5 = MathTex("\\lambda_1 = 5, \\quad \\lambda_2 = 3", color=COLOR_D).scale(0.9)
        det_grp = VGroup(d1, d2, d3, d4, d5).arrange(DOWN, buff=0.3).move_to(ORIGIN)
        for d in [d1, d2, d3, d4, d5]:
            self.play(FadeIn(d, shift=UP * 0.3), run_time=0.7)
            self.wait(0.5)
        self.wait(2)
        self.play(FadeOut(det_grp))

        # ─── SLIDE 4: Eigenvectors ───
        sub5 = Subtitle("Bước 3: Tìm vector riêng ứng với mỗi trị riêng.")
        self.play(Transform(sub, sub5))

        ev1 = VGroup(
            MathTex("\\lambda_1 = 5:", color=COLOR_D).scale(0.75),
            MathTex("(A-5I)\\vec{v}=\\vec{0}").scale(0.65),
            MathTex("\\begin{bmatrix} 0&0\\\\4&-2 \\end{bmatrix}\\vec{v}=\\vec{0}").scale(0.65),
            MathTex("4v_1 - 2v_2 = 0").scale(0.65),
            MathTex("v_2 = 2v_1 \\Rightarrow \\vec{v}_1 = \\begin{bmatrix} 1 \\\\ 2 \\end{bmatrix}", color=COLOR_P).scale(0.7),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)

        ev2 = VGroup(
            MathTex("\\lambda_2 = 3:", color=COLOR_D).scale(0.75),
            MathTex("(A-3I)\\vec{v}=\\vec{0}").scale(0.65),
            MathTex("\\begin{bmatrix} 2&0\\\\4&0 \\end{bmatrix}\\vec{v}=\\vec{0}").scale(0.65),
            MathTex("2v_1 = 0").scale(0.65),
            MathTex("v_1 = 0 \\Rightarrow \\vec{v}_2 = \\begin{bmatrix} 0 \\\\ 1 \\end{bmatrix}", color=COLOR_P).scale(0.7),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)

        ev_all = VGroup(ev1, ev2).arrange(RIGHT, buff=1.0, aligned_edge=UP).move_to(ORIGIN)
        self.play(FadeIn(ev1, shift=RIGHT)); self.wait(2.5)
        self.play(FadeIn(ev2, shift=RIGHT)); self.wait(3)
        self.play(FadeOut(ev_all))

        # ─── SLIDE 5: Build P from eigenvectors ───
        sub6 = Subtitle("Bước 4: Ghép các vector v1, v2 thành các cột của ma trận P.")
        self.play(Transform(sub, sub6))

        # Show P = [v1 | v2] explicitly
        p_build = MathTex(
            "P", "=", "\\begin{bmatrix} \\vec{v}_1 & \\vec{v}_2 \\end{bmatrix}",
            "=", "\\begin{bmatrix} 1 & 0 \\\\ 2 & 1 \\end{bmatrix}"
        ).scale(0.9)
        p_build[0].set_color(COLOR_P)
        p_build[2].set_color(COLOR_P)

        d_build = MathTex(
            "D", "=", "\\begin{bmatrix} \\lambda_1 & 0 \\\\ 0 & \\lambda_2 \\end{bmatrix}",
            "=", "\\begin{bmatrix} 5 & 0 \\\\ 0 & 3 \\end{bmatrix}"
        ).scale(0.9)
        d_build[0].set_color(COLOR_D)
        d_build[2].set_color(COLOR_D)

        pd_grp = VGroup(p_build, d_build).arrange(DOWN, buff=0.6).move_to(UP * 0.5)

        self.play(Write(p_build))
        self.wait(2)
        self.play(Write(d_build))
        self.wait(2)

        sub6b = Subtitle("Vì A không đối xứng, ta tính ma trận nghịch đảo P-1.")
        pinv = MathTex(
            "P^{-1}", "=", "\\begin{bmatrix} 1 & 0 \\\\ -2 & 1 \\end{bmatrix}"
        ).scale(0.9).next_to(pd_grp, DOWN, buff=0.5)
        pinv[0].set_color(COLOR_P)

        self.play(Transform(sub, sub6b), FadeIn(pinv, shift=UP))
        self.wait(3)
        self.play(FadeOut(pd_grp), FadeOut(pinv))

        # ─── SLIDE 6: Final result ───
        sub7 = Subtitle("Kết quả chéo hóa: A = P nhân D nhân P nghịch đảo.")
        self.play(Transform(sub, sub7))
        final = MathTex(
            "\\begin{bmatrix} 5&0\\\\4&3 \\end{bmatrix}", "=",
            "\\begin{bmatrix} 1&0\\\\2&1 \\end{bmatrix}",
            "\\begin{bmatrix} 5&0\\\\0&3 \\end{bmatrix}",
            "\\begin{bmatrix} 1&0\\\\-2&1 \\end{bmatrix}"
        ).scale(0.85)
        final[0].set_color(COLOR_A); final[2].set_color(COLOR_P)
        final[3].set_color(COLOR_D); final[4].set_color(COLOR_P)
        self.play(Write(final))
        self.wait(4)
        self.play(FadeOut(final), FadeOut(sub))


# ════════════════════════════════════════════════════════════════
class SVDFullVisualization(Scene):
    def construct(self):
        SVDIntroScene.construct(self)
        SVDStepByStepScene.construct(self)
        SVDGeometricScene.construct(self)
        DiagonalizationScene.construct(self)
