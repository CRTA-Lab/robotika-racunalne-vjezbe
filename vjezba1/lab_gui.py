"""
Vjezba 1 - GUI. Studenti ovo ne mijenjaju.

Robot se crta studentskim funkcijama; dok forward_kinematics ne radi, crta se referentni.
Sivi "duh" je uvijek referentni robot.
"""

import time

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

import lab_core as core

COL_X, COL_Y, COL_Z = "#d62728", "#2ca02c", "#1f3fd6"
LINK_COL = ("#ff7f0e", "#9467bd", "#17becf", "#8c564b")
SLIDER_COL = "#1f77b4"
OK_COL, BAD_COL, TODO_COL = "#1a9850", "#d73027", "#8c8c8c"
VEL_COL = "#c2185b"

VEL_SCALE = 0.5  # strelica brzine: duljina = |v| * 0.5 s
SING_EPS = 0.01  # rub dosega
TIMER_MS = 40
VEL_R, VEL_T = 30, 0.1  # °/s, m/s
ACC_R, ACC_T = 30, 0.1  # °/s², m/s²


class LabGUI:
    def __init__(self, student, robot_name="RR"):
        if robot_name not in core.ROBOTS:
            raise ValueError(
                f"Nepoznat robot '{robot_name}'. Dostupni: {list(core.ROBOTS)}"
            )
        self.student = student
        self.checks = core.run_checks(student)
        self.fig = plt.figure(figsize=(17.5, 9.6))
        self.timer = self.fig.canvas.new_timer(interval=TIMER_MS)
        self.timer.add_callback(self._tick)
        self.fig.canvas.mpl_connect("draw_event", self._on_draw)
        self.animated = []
        self._sym = {}
        self._J_fn = {}
        self._build(robot_name)

    def _build(self, robot_name):
        """Slozi cijeli prozor za zadanog robota (na pocetku i kod promjene robota)."""
        self.robot = core.ROBOTS[robot_name]
        self.m = len(self.robot.rows)  # s alatom
        self.is3d = self.robot.dim == 3
        self.trace = []
        self._widgets, self._cids = [], []
        for attr in ("ghost_arrow", "frame_z"):  # ostaci prethodnog robota
            self.__dict__.pop(attr, None)
        try:
            self.fig.canvas.manager.set_window_title(f"Vježba 1 - {self.robot.title}")
        except Exception:
            pass
        if self.is3d:
            self._build_scene3d()
        else:
            self._build_scene()
        self._build_panel()
        self._build_controls()
        self._setup_blit()
        self.update()

    def _on_robot(self, name):
        # gradi se tek nakon sto matplotlib zavrsi s klikom
        self._switch_timer = self.fig.canvas.new_timer(interval=10)
        self._switch_timer.single_shot = True
        self._switch_timer.add_callback(self.switch_robot, name)
        self._switch_timer.start()

    def switch_robot(self, name):
        if name == self.robot.name:
            return
        self.timer.stop()
        for w in self._widgets:
            w.disconnect_events()
        for cid in self._cids:
            self.fig.canvas.mpl_disconnect(cid)
        self.animated = []
        self.fig.clf()
        self.fig.text(0.5, 0.5, f"{name} ...", ha="center", fontsize=16, color="0.5")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.fig.clf()
        self._build(name)
        self.fig.canvas.draw_idle()

    # ---------------------------------------------------------------- scena
    def _build_scene(self):
        r = self.robot
        ax = self.ax = self.fig.add_axes([0.05, 0.31, 0.39, 0.62])
        (x0, x1), (y0, y1) = r.bounds
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.35)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.axvline(0, color="0.6", lw=0.8)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        self.fl = 0.10 * max(x1 - x0, y1 - y0)  # duljina osi KS

        # duh
        (self.ghost_line,) = ax.plot(
            [0], [0], color="0.55", lw=12, alpha=0.35, solid_capstyle="round", zorder=1
        )
        # brzina vrha v = J q'
        self.ghost_arrow = FancyArrowPatch(
            (0, 0), (0, 0), arrowstyle="-|>", mutation_scale=18, lw=3, color="0.6", zorder=8
        )
        self.v_arrow = FancyArrowPatch(
            (0, 0), (0, 0), arrowstyle="-|>", mutation_scale=16, lw=2, color=VEL_COL, zorder=9
        )
        ax.add_patch(self.ghost_arrow)
        ax.add_patch(self.v_arrow)

        # tvoj robot
        (self.trace_line,) = ax.plot(
            [0], [0], color="#e377c2", lw=1.3, alpha=0.8, zorder=2
        )
        self.rails, self.boxes, self.links, self.joints = {}, {}, [], {}
        for i, row in enumerate(r.rows):
            self.links.append(
                ax.plot(
                    [0],
                    [0],
                    color=LINK_COL[i % 4],
                    lw=7,
                    solid_capstyle="round",
                    zorder=3,
                )[0]
            )
            if row.kind == "T":
                self.rails[i] = ax.plot([0], [0], color="0.55", lw=3, zorder=2.5)[0]
                self.boxes[i] = ax.plot(
                    [0], [0], color=SLIDER_COL, lw=18, solid_capstyle="butt", zorder=4
                )[0]
            if row.kind == "R":
                self.joints[i] = ax.plot(
                    [0],
                    [0],
                    marker="o",
                    ms=13,
                    mfc="#222",
                    mec="white",
                    mew=1.6,
                    ls="",
                    zorder=5,
                )[0]

        self.frame_x, self.frame_y, self.frame_txt = [], [], []
        for i in range(self.m + 1):
            self.frame_x.append(
                ax.plot([0, 0], [0, 0], color=COL_X, lw=2.4, zorder=7)[0]
            )
            self.frame_y.append(
                ax.plot([0, 0], [0, 0], color=COL_Y, lw=2.4, zorder=7)[0]
            )
            self.frame_txt.append(
                ax.text(0, 0, "{%d}" % i, fontsize=11, fontweight="bold", zorder=8)
            )

    def _build_scene3d(self):
        r = self.robot
        ax = self.ax = self.fig.add_axes([0.02, 0.27, 0.44, 0.71], projection="3d")
        (x0, x1), (y0, y1), (z0, z1) = r.bounds
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_zlim(z0, z1)
        ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))
        ax.view_init(elev=25, azim=-60)
        # misem se okrece samo oko osi z (granice x/y su simetricne pa os prolazi kroz bazu)
        ax.disable_mouse_rotation()
        self._elev, self._drag = 25, None
        cid = self.fig.canvas.mpl_connect
        self._cids += [
            cid("button_press_event", self._rot_press),
            cid("motion_notify_event", self._rot_move),
            cid("button_release_event", self._rot_release),
        ]
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        # va="top" da drugi redak naslova ne izade iz prozora
        ax.set_title(" ", fontsize=11, y=1.07, va="top")
        self.fl = 0.10 * max(x1 - x0, y1 - y0, z1 - z0)

        b = 0.08
        ax.plot([-b, b, b, -b, -b], [-b, -b, b, b, -b], [0] * 5, color="0.35", lw=1.5)

        (self.ghost_line,) = ax.plot(
            [0], [0], [0], color="0.55", lw=12, alpha=0.35, solid_capstyle="round"
        )
        (self.trace_line,) = ax.plot([0], [0], [0], color="#e377c2", lw=1.3, alpha=0.8)
        (self.v_arrow,) = ax.plot(
            [0], [0], [0], color=VEL_COL, lw=2.5, marker="o", ms=5, markevery=[-1]
        )
        self.rails, self.boxes, self.links, self.joints = {}, {}, [], {}
        for i, row in enumerate(r.rows):
            self.links.append(
                ax.plot(
                    [0], [0], [0], color=LINK_COL[i % 4], lw=7, solid_capstyle="round"
                )[0]
            )
            if row.kind == "T":  # vreteno
                self.boxes[i] = ax.plot(
                    [0], [0], [0], color=SLIDER_COL, lw=9, solid_capstyle="butt"
                )[0]
            if row.kind == "R":
                self.joints[i] = ax.plot(
                    [0], [0], [0], marker="o", ms=11, mfc="#222", mec="white", ls=""
                )[0]

        self.frame_x, self.frame_y, self.frame_z, self.frame_txt = [], [], [], []
        for i in range(self.m + 1):
            for lst, col in (
                (self.frame_x, COL_X),
                (self.frame_y, COL_Y),
                (self.frame_z, COL_Z),
            ):
                lst.append(ax.plot([0, 0], [0, 0], [0, 0], color=col, lw=2.2)[0])
            self.frame_txt.append(
                ax.text(0, 0, 0, "{%d}" % i, fontsize=10, fontweight="bold")
            )

    # --------------------------------------------------------------- panel
    def _build_panel(self):
        fig, r, m = self.fig, self.robot, self.m
        axp = self.axp = fig.add_axes([0.465, 0.02, 0.53, 0.96])
        axp.axis("off")
        axp.set_xlim(0, 1)
        axp.set_ylim(0, 1)

        # izbor robota (umjesto naslova)
        x = 0.465
        for name in core.ROBOTS:
            w = 0.03 + 0.005 * len(name)
            active = name == r.name
            b = Button(
                fig.add_axes([x, 0.945, w, 0.03]),
                name,
                color=SLIDER_COL if active else "0.9",
                hovercolor=SLIDER_COL if active else "0.78",
            )
            b.label.set_fontsize(11)
            if active:
                b.label.set_color("white")
                b.label.set_fontweight("bold")
            b.on_clicked(lambda _e, name=name: self._on_robot(name))
            self._widgets.append(b)
            x += w + 0.005

        clanak = (
            r"članak:  $A_i = \mathrm{Transl}(x_i, 0, z_i)\,\cdot\,\mathrm{Rot}_z(\theta_i)$"
            if self.is3d
            else r"članak:  $A_i = \mathrm{Transl}(x_i, 0)\,\cdot\,\mathrm{Rot}(\theta_i)$"
        )
        renderer = fig.canvas.get_renderer()
        prod = r"$T_{0e} = " + " ".join("A_%d" % (i + 1) for i in range(m)) + r"$  (baza $\to$ alat)"
        t = axp.text(0, 0.955, clanak + "      " + prod, fontsize=10.5, va="top")
        y = 0.955 - 0.027
        if t.get_window_extent(renderer).width / axp.bbox.width > 0.59:
            t.set_text(clanak)
            axp.text(0, y, prod, fontsize=10.5, va="top")
            y -= 0.027
        axp.text(
            0,
            y,
            r"oznake:  $c_i=\cos q_i$,  $s_i=\sin q_i$,  $c_{ij}=\cos(q_i+q_j)$,  "
            r"$s_{ij}=\sin(q_i+q_j)$",
            fontsize=9.5,
            va="top",
            color="0.35",
        )

        # checklista
        sym = {"ok": ("✔", OK_COL), "fail": ("✘", BAD_COL), "todo": ("…", TODO_COL)}
        x, y = 0.0, y - 0.027
        for c in self.checks:
            mark, col = sym[c.status]
            t = axp.text(
                x, y, f"{mark} {c.name}", fontsize=9.5, color=col, family="monospace", va="top"
            )
            w = t.get_window_extent(renderer).width / axp.bbox.width
            if x > 0 and x + w > 0.59:
                x, y = 0.0, y - 0.022
                t.set_position((x, y))
            x += w + 0.02
        left_bottom = y - 0.022

        # tablica elemenata
        y0 = 0.985
        if self.is3d:
            cols = [(0.65, "x", r"$x_i$"), (0.765, "z", r"$z_i$")]
            cols.append((0.915, "theta", r"$\theta_i$"))
            fs_tab, dy = 9.0, 0.021
        else:
            cols = [(0.66, "x", r"$x_i$"), (0.82, "theta", r"$\theta_i$")]
            fs_tab, dy = 9.5, 0.026
        axp.text(0.62, y0, "Tablica elemenata", fontsize=10, fontweight="bold", va="top")
        axp.text(0.62, y0 - 0.024, r"$i$", fontsize=10, va="top")
        for xc, _f, head in cols:
            axp.text(xc, y0 - 0.024, head, fontsize=10, va="top")
        self.row_lat = [
            tuple(core.compact_latex(getattr(row, f)) for _x, f, _h in cols)
            for row in r.rows
        ]
        self.row_zero = [
            tuple(getattr(row, f) == 0 for _x, f, _h in cols) for row in r.rows
        ]
        self.tab_cells = []
        for i in range(m):
            yy = y0 - 0.052 - dy * i
            axp.text(0.62, yy, str(i + 1), fontsize=fs_tab, va="top")
            self.tab_cells.append(
                [axp.text(xc, yy, "", fontsize=fs_tab, va="top") for xc, _f, _h in cols]
            )
        table_bottom = y0 - 0.052 - dy * m

        # matrice
        n = r.dim + 1
        # SCARA nema status ispod matrice pa je razmak manji
        h, gap = (0.075, 0.040) if n == 3 else (0.072, 0.020)
        row_h = h + gap
        top = min(left_bottom, table_bottom) - 0.005
        self.panel_free_top = top - (m + 1) * row_h - 0.012
        if r.name not in self._sym:  # sporo, pa se pamti po robotu
            self._sym[r.name] = core.student_symbolic(self.student, r)
        A_sym, T_sym, J_sym, note_A, note_T, note_J, _st_J = self._sym[r.name]
        latex = lambda M: [
            [core.compact_latex(M[i, j]) for j in range(M.cols)] for i in range(M.rows)
        ]

        self.num_txt, self.row_status = [], []
        for k, M in enumerate(A_sym + [T_sym]):
            ytop = top - k * row_h
            if k == m:  # crta iznad T_0e
                axp.plot([0, 1], [ytop + 0.002, ytop + 0.002], color="0.5", lw=1.0)
                ytop -= 0.012
            yc = ytop - h / 2 - 0.008
            label = r"$A_%d$" % (k + 1) if k < m else r"$T_{0e}$"
            axp.text(0.0, yc, label + r"$=$", fontsize=14, va="center", ha="left")
            note = note_A if k < m else note_T
            col = TODO_COL if note else "k"
            self._draw_symbolic(axp, latex(M), 0.085, 0.50, yc, h, col)
            axp.text(0.085, yc - h / 2 - 0.016, note, fontsize=8.5, color=TODO_COL, va="center")
            axp.text(0.625, yc, "=", fontsize=14, va="center", ha="center")
            self.num_txt.append(self._make_numeric(axp, 0.655, 0.32, yc, h, n))
            self.row_status.append(
                axp.text(
                    0.655, yc - h / 2 - 0.016, "", fontsize=8.5, va="center", ha="left"
                )
            )

        # jakobijan
        ytop = self.panel_free_top
        axp.plot([0, 1], [ytop + 0.002, ytop + 0.002], color="0.5", lw=1.0)
        ytop -= 0.012
        yc = ytop - h / 2 - 0.008
        nc = J_sym.cols
        axp.text(0.0, yc, r"$J=$", fontsize=14, va="center", ha="left")
        self._draw_symbolic(
            axp, latex(J_sym), 0.085, 0.50, yc, h, TODO_COL if note_J else "k"
        )
        axp.text(0.085, yc - h / 2 - 0.016, note_J, fontsize=8.5, color=TODO_COL, va="center")
        axp.text(0.625, yc, "=", fontsize=14, va="center", ha="center")
        self.J_txt = self._make_numeric(axp, 0.655, 0.32 / J_sym.rows * nc, yc, h, J_sym.rows, nc)
        self.J_status = axp.text(
            0.655, yc - h / 2 - 0.016, "", fontsize=8.5, va="center", ha="left"
        )
        # rang J (J nije nuzno kvadratna); u singularitetu rang pada
        y_det = yc - h / 2 - 0.045
        self.det_txt = axp.text(0.655, y_det, "", fontsize=10, va="center")
        self.v_txt = axp.text(0.0, y_det - 0.04, "", fontsize=10, va="center")

    def _draw_symbolic(self, ax, M, x0, w, yc, h, col="k"):
        n, nc = len(M), len(M[0])
        fs = 12.0 if n == 3 else 10.5
        txt = [
            [
                ax.text(
                    0, yc + h / 2 - (i + 0.5) * h / n, "$" + M[i][j] + "$",
                    fontsize=fs, ha="center", va="center", color=col,
                )
                for j in range(nc)
            ]
            for i in range(n)
        ]
        # ako je presiroko, smanji font
        renderer = self.fig.canvas.get_renderer()
        colw = [
            max(txt[i][j].get_window_extent(renderer).width for i in range(n))
            / ax.bbox.width
            for j in range(nc)
        ]
        gap = 0.025
        scale = min(1.0, max((w - gap * nc) / sum(colw), 0.55))
        colw = [cw * scale + gap for cw in colw]
        tot = sum(colw)
        x = x0 + (w - tot) / 2
        for j, cw in enumerate(colw):
            for i in range(n):
                txt[i][j].set_x(x + cw / 2)
                txt[i][j].set_fontsize(fs * scale)
            x += cw
        self._brackets(
            ax, x0 + (w - tot) / 2 - 0.004, x0 + (w + tot) / 2 + 0.004, yc, h
        )

    @staticmethod
    def _brackets(ax, xl, xr, yc, h):
        yt, yb, tk = yc + h / 2 + 0.004, yc - h / 2 - 0.004, 0.006
        ax.plot([xl + tk, xl, xl, xl + tk], [yt, yt, yb, yb], color="k", lw=1.3)
        ax.plot([xr - tk, xr, xr, xr - tk], [yt, yt, yb, yb], color="k", lw=1.3)

    def _make_numeric(self, ax, x0, w, yc, h, n=3, nc=None):
        nc = nc or n
        cw = w / nc
        txt = []
        for i in range(n):
            y = yc + h / 2 - (i + 0.5) * h / n
            txt.append(
                [
                    ax.text(
                        x0 + (j + 0.5) * cw,
                        y,
                        "",
                        fontsize=11 if n == 3 else 9.5,
                        family="monospace",
                        ha="center",
                        va="center",
                    )
                    for j in range(nc)
                ]
            )
        self._brackets(ax, x0 - 0.004, x0 + w + 0.004, yc, h)
        return txt

    # ------------------------------------------------------------ kontrole
    def _build_controls(self):
        fig, r = self.fig, self.robot
        nj = len(r.var_rows)
        self.is_rot = np.array([row.kind == "R" for row in r.var_rows])
        self.unit = np.where(self.is_rot, np.pi / 180, 1.0)  # klizaci su u ° i m
        self.q_lo = np.array([row.q_min for row in r.var_rows])
        self.q_hi = np.array([row.q_max for row in r.var_rows])
        self.qd_max = np.where(self.is_rot, np.radians(VEL_R), VEL_T)
        self.q_state = np.array(r.q_home, dtype=float)
        self.qd, self.qdd = np.zeros(nj), np.zeros(nj)
        self.mode_vel, self.paused = False, False

        # redak po zglobu: polozaj | brzina | akceleracija
        y0, dy, hs = (0.205, 0.042, 0.024) if nj <= 3 else (0.215, 0.034, 0.022)
        cols = ((0.04, 0.055), (0.186, 0.055), (0.332, 0.055))
        for (xc, w), head in zip(cols, ("položaj", "brzina", "akceleracija")):
            fig.text(xc + w / 2, y0 + 0.034, head, fontsize=9.5, ha="center", color="0.3")
        self.sliders, self.sliders_v, self.sliders_a = [], [], []
        for i, row in enumerate(r.var_rows):
            rot, y = row.kind == "R", y0 - dy * i

            def mk(k, sym, lo, hi, init, fmt, step):
                xc, w = cols[k]
                s = Slider(
                    fig.add_axes([xc, y, w, hs]),
                    sym % (i + 1),
                    lo,
                    hi,
                    valinit=init,
                    valstep=step,
                    valfmt=fmt,
                )
                s.label.set_fontsize(10)
                s.valtext.set_fontsize(9)
                s.drawon = False  # crta ga update()
                # tipka "0": klizac je tesko pogoditi mišem, pa ga ova postavlja tocno na 0
                # (ili na najblizu granicu raspona ako 0 nije unutar njega)
                b = Button(fig.add_axes([xc + w + 0.058, y + 0.002, 0.012, hs - 0.004]), "0")
                b.label.set_fontsize(8)
                b.on_clicked(lambda _e, s=s: s.set_val(min(max(0.0, s.valmin), s.valmax)))
                self._widgets.append(b)
                return s

            if rot:
                sq = mk(0, r"$q_%d$", -180, 180, np.degrees(r.q_home[i]), "%.0f°", 1)
                sv = mk(1, r"$\dot{q}_%d$", -VEL_R, VEL_R, 0, "%.0f°/s", 1)
                sa = mk(2, r"$\ddot{q}_%d$", -ACC_R, ACC_R, 0, "%.0f°/s²", 1)
            else:
                sq = mk(0, r"$q_%d$", row.q_min, row.q_max, r.q_home[i], "%.2f m", 0.01)
                sv = mk(1, r"$\dot{q}_%d$", -VEL_T, VEL_T, 0, "%.2f m/s", 0.01)
                sa = mk(2, r"$\ddot{q}_%d$", -ACC_T, ACC_T, 0, "%.2f m/s²", 0.01)
            sq.on_changed(lambda _v, i=i: self._on_q(i))
            sv.on_changed(lambda _v, i=i: self._on_qd(i))
            sa.on_changed(lambda _v, i=i: self._on_qdd(i))
            self.sliders.append(sq)
            self.sliders_v.append(sv)
            self.sliders_a.append(sa)
            self._widgets += [sq, sv, sa]

        self.btn = Button(fig.add_axes([0.02, 0.03, 0.065, 0.045]), "Reset")
        self.btn.on_clicked(self._reset)

        axm = fig.add_axes([0.095, 0.012, 0.085, 0.08])
        axm.set_frame_on(False)
        # useblit=False, inace _refresh nacrta staro stanje gumba
        self.radio = RadioButtons(axm, ("use pos", "use vel"), useblit=False)
        self.radio.on_clicked(self._on_mode)

        axc = fig.add_axes([0.19, 0.012, 0.25, 0.08])
        axc.set_frame_on(False)
        self.chk = CheckButtons(
            axc,
            ["referentni robot (duh)", "trag krajnje točke", "osvježavaj matrice"],
            [True, True, True],
            useblit=False,
        )
        self.chk.on_clicked(lambda _l: self.update())
        self._widgets += [self.btn, self.radio, self.chk]
        self._t_last = time.perf_counter()

    # ------------------------------------------------------------ blitting
    # staticni dio se crta jednom, poslije se crta samo ono sto je u self.animated
    def _setup_blit(self):
        sliders = self.sliders + self.sliders_v + self.sliders_a
        self.animated = (
            [self.ghost_line, self.trace_line, self.v_arrow]
            + ([self.ghost_arrow] if hasattr(self, "ghost_arrow") else [])
            + self.links
            + list(self.rails.values())
            + list(self.boxes.values())
            + list(self.joints.values())
            + self.frame_x
            + self.frame_y
            + getattr(self, "frame_z", [])
            + self.frame_txt
            + [self.ax.title]
            + [t for M in self.num_txt + [self.J_txt] for row in M for t in row]
            + self.row_status
            + [self.J_status, self.det_txt, self.v_txt]
            + [t for cell in self.tab_cells for t in cell]
            + [a for s in sliders for a in (s.poly, s.vline, s._handle, s.valtext)]
        )
        for a in self.animated:
            a.set_animated(True)
        self._bg = None

    def _on_draw(self, _evt):
        if self.fig.canvas.is_saving():
            return
        self._bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self._draw_animated()

    def _draw_animated(self):
        for a in self.animated:
            self.fig.draw_artist(a)

    def _refresh(self):
        canvas = self.fig.canvas
        if self._bg is None or not getattr(canvas, "supports_blit", False):
            canvas.draw_idle()
            return
        canvas.restore_region(self._bg)
        self._draw_animated()
        canvas.blit(self.fig.bbox)

    # ------------------------------------------------------------ klizaci
    @staticmethod
    def _set_silent(s, v):
        s.eventson = False
        s.set_val(v)
        s.eventson = True

    def _sync_sliders(self):
        for s, v, u in zip(self.sliders, self.q_state, self.unit):
            self._set_silent(s, v / u)
        for s, v, u in zip(self.sliders_v, self.qd, self.unit):
            self._set_silent(s, v / u)

    def _on_q(self, i):
        if self.mode_vel:  # klizac polozaja tu ne radi nista
            self._set_silent(self.sliders[i], self.q_state[i] / self.unit[i])
            return
        self.q_state[i] = self.sliders[i].val * self.unit[i]
        self.update()

    def _on_qd(self, i):
        self.qd[i] = self.sliders_v[i].val * self.unit[i]
        self.paused = False
        if not self.mode_vel:
            self.update()

    def _on_qdd(self, i):
        self.qdd[i] = self.sliders_a[i].val * self.unit[i]
        self.paused = False

    def _on_mode(self, label):
        self.mode_vel = label == "use vel"
        self.paused = False
        if self.mode_vel:
            self._t_last = time.perf_counter()
            self.timer.start()
        else:
            self.timer.stop()
            self._sync_sliders()
        self.update()

    def _tick(self):
        now = time.perf_counter()
        dt, self._t_last = min(now - self._t_last, 0.1), now
        if not self.mode_vel or self.paused:
            return
        q0, qd0 = self.q_state.copy(), self.qd.copy()
        self.step(dt)
        if self.paused or not (
            np.array_equal(q0, self.q_state) and np.array_equal(qd0, self.qd)
        ):
            self._sync_sliders()
            self.update()

    def step(self, dt):
        # mali podkoraci (1° / 5 mm) da se ne preskoci singularitet
        r = self.robot
        q, qd = self.q_state.copy(), self.qd.copy()
        max_d = np.where(self.is_rot, np.radians(1.0), 0.005)
        n_sub = np.max((np.abs(qd) + np.abs(self.qdd) * dt) * dt / max_d)
        n_sub = int(min(max(np.ceil(n_sub), 1), 500))
        h = dt / n_sub
        s_old = r.sigma_min_pos(q)
        for _ in range(n_sub):
            qd = np.clip(qd + self.qdd * h, -self.qd_max, self.qd_max)
            qn = q + qd * h
            # zglob na limitu stane
            hit = ((qn <= self.q_lo) & (qd < 0)) | ((qn >= self.q_hi) & (qd > 0))
            qn = np.clip(qn, self.q_lo, self.q_hi)
            qd[hit] = 0.0
            # rub dosega: stane cijeli robot dok se ne promijeni brzina ili Reset
            s_new = r.sigma_min_pos(qn)
            if s_new < SING_EPS and s_new < s_old:
                self.paused = True
                break
            q, s_old = qn, s_new
        self.q_state, self.qd = q, qd

    def _reset(self, _evt):
        if self.mode_vel:
            self.radio.set_active(0)
        self.q_state = np.array(self.robot.q_home, dtype=float)
        self.qd[:] = 0.0
        self.qdd[:] = 0.0
        self.paused = False
        self._sync_sliders()
        for s in self.sliders_a:
            self._set_silent(s, 0)
        self.trace.clear()
        self.update()

    def get_q(self):
        return self.q_state.copy()

    def set_q(self, q):
        # q u rad / m
        self.q_state = np.array(q, dtype=float)
        self._sync_sliders()
        self.update()

    @staticmethod
    def _safe_mat(fn, *args, shape=(3, 3)):
        try:
            out = fn(*args)
        except NotImplementedError:
            return None, "todo", "funkcija nije gotova"
        except Exception as e:
            return None, "fail", f"greška: {type(e).__name__}: {e}"[:60]
        try:
            arr = np.asarray(out, dtype=float)
        except Exception:
            return None, "fail", "krivi tip povratne vrijednosti"
        if arr.shape != shape:
            return None, "fail", f"matrica mora biti {shape[0]}x{shape[1]}, a ima oblik {arr.shape}"
        return arr, "ok", ""

    @staticmethod
    def _entry(stu, ref, err=None):
        M, st, msg = stu
        if M is None:
            return ref, TODO_COL, st, msg
        err = np.max(np.abs(M - ref)) if err is None else err
        if err < 1e-9:
            return M, "k", "ok", ""
        return M, "k", "fail", "razlikuje se od reference"

    @staticmethod
    def _show(txt, s, M, col="k", st=None, msg=""):
        for i, row in enumerate(txt):
            for j, t in enumerate(row):
                v = 0.0 if abs(M[i, j]) < 5e-4 else M[i, j]
                t.set_text(f"{v:+.3f}")
                t.set_color(col)
        if st is None:
            return
        if st == "ok":
            s.set_text("✔ podudara se s referentnim rješenjem")
            s.set_color(OK_COL)
        elif st == "todo":
            s.set_text(msg + " - prikazano referentno")
            s.set_color(TODO_COL)
        else:
            s.set_text("✘ " + msg[:58])
            s.set_color(BAD_COL)

    def _safe_fk(self, rows):
        try:
            out = self.student.forward_kinematics(rows)
        except NotImplementedError:
            return None, "todo", "funkcija nije gotova"
        except Exception as e:
            return None, "fail", f"greška: {type(e).__name__}: {e}"[:60]
        try:
            arr = [np.asarray(T, dtype=float) for T in out]
        except Exception:
            return None, "fail", "funkcija mora vratiti listu 3x3 matrica"
        if len(arr) != len(rows) or any(T.shape != (3, 3) for T in arr):
            return None, "fail", f"očekivana lista od {len(rows)} matrica 3x3"
        return arr, "ok", ""

    # ---------------------------------------------------------- osvjezavanje
    def update(self, _val=None):
        if self.is3d:
            self._update3d()
            return
        r, m = self.robot, self.m
        q = self.get_q()
        rows = r.link_rows(q)
        T_ref = r.reference_T(q)
        I3 = np.eye(3)

        live = self.chk.get_status()[2]  # isključeno: tekst matrica se ne osvježava
        A_stu = [self._safe_mat(self.student.link_matrix, x, th) for x, th in rows] if live else []
        T_stu, T_st, T_msg = self._safe_fk(rows)
        T_draw = ([I3] + T_stu) if T_stu is not None else ([I3] + T_ref)
        using_ref = T_stu is None

        if live:
            refs_A = [core.ref_link(x, th) for x, th in rows]
            entries = [self._entry(A_stu[i], refs_A[i]) for i in range(m)]
            if T_stu is None:
                entries.append((T_ref[-1], TODO_COL, T_st, T_msg))
            else:
                err = max(np.max(np.abs(a - b)) for a, b in zip(T_stu, T_ref))
                entries.append(self._entry((T_stu[-1], "ok", ""), T_ref[-1], err))
            for k, e in enumerate(entries):
                self._show(self.num_txt[k], self.row_status[k], *e)

        # jakobijan i brzina vrha
        J_ref = r.reference_J(q)
        _, _, J_sym, _, _, _, (st_J, msg_J) = self._sym[r.name]
        if st_J != "ok":
            J_stu = (None, st_J, msg_J)
        else:
            if r.name not in self._J_fn:  # lambdify je spor, pa se pamti po robotu
                self._J_fn[r.name] = sp.lambdify(r._syms, J_sym, "numpy")
            J_stu = self._safe_mat(self._J_fn[r.name], *r._args(q), shape=J_ref.shape)
        J_show, col, st, msg = self._entry(J_stu, J_ref)
        v_ref = J_ref @ self.qd
        v = J_show @ self.qd
        if live:
            self._show(self.J_txt, self.J_status, J_show, col, st, msg)
            self._show_det(J_show, col)
            self._show_v(v, col)

        for i, (x, th) in enumerate(rows):
            lx, lt = self.row_lat[i]
            zx, zt = self.row_zero[i]
            x = 0.0 if abs(x) < 5e-4 else x
            th = 0.0 if abs(np.degrees(th)) < 5e-2 else th
            self.tab_cells[i][0].set_text(
                "$0$" if zx else f"${lx} = {x:.3f}\\ \\mathrm{{m}}$"
            )
            self.tab_cells[i][1].set_text(
                "$0$" if zt else f"${lt} = {np.degrees(th):.1f}^\\circ$"
            )

        self._draw(T_draw, rows)
        O = np.array([T[:2, 2] for Tm in [[I3] + T_ref] for T in Tm])
        self.ghost_line.set_data(O[:, 0], O[:, 1])
        show_ghost = self.chk.get_status()[0]
        self.ghost_line.set_visible(show_ghost)
        self._set_arrow(self.ghost_arrow, O[-1], v_ref[:2], show_ghost)

        T_n = T_draw[-1]
        p = self._tcp_position(T_n)
        phi = self._tcp_angle(T_n)
        title = "krajnja točka (TCP):  x = %+.3f m   y = %+.3f m   φ = %+.1f°" % (
            p[0],
            p[1],
            np.degrees(phi),
        )
        if using_ref:
            title += "\n(prikazan je referentni robot - forward_kinematics još ne radi)"
        title += self._pause_note()
        self._set_arrow(self.v_arrow, p, v[:2], J_stu[0] is not None)
        self.ax.set_title(title, fontsize=11, color=("0.4" if using_ref else "k"))

        self.trace.append(p)
        self.trace = self.trace[-500:]
        if self.chk.get_status()[1] and len(self.trace) > 1:
            tr = np.array(self.trace)
            self.trace_line.set_data(tr[:, 0], tr[:, 1])
            self.trace_line.set_visible(True)
        else:
            self.trace_line.set_visible(False)
        self._refresh()

    def _pause_note(self):
        if self.mode_vel and self.paused:
            return "\nzaustavljeno: rub dosega (singularitet) - promijeni brzinu ili Reset"
        return ""

    def _show_det(self, J, col):
        # manje zglobova nego prostornih stupnjeva slobode: gleda se pozicijski dio (x, y),
        # inace redak phi uvijek odrzava rang pun
        dim = 2 if J.shape[0] == 3 else 3
        if J.shape[1] <= dim:
            J = J[:dim]
        k = min(J.shape)
        rank = np.linalg.matrix_rank(J, tol=1e-3)
        self.det_txt.set_text(f"rang J = {rank} (maks. {k})"
                              + ("   singularitet" if rank < k else ""))
        self.det_txt.set_color(col)

    def _show_v(self, v, col):
        names = [r"\dot{x}", r"\dot{y}", r"\dot{z}"][: len(v) - 1]
        parts = [f"${n}$ = {x:+.3f} m/s" for n, x in zip(names, v)]
        parts.append(rf"$\dot{{\varphi}}$ = {np.degrees(v[-1]):+.1f} °/s")
        self.v_txt.set_text(r"$v = J\,\dot{q}$:    " + "    ".join(parts))
        self.v_txt.set_color(col)

    def _set_arrow(self, arrow, p, v, visible=True):
        visible = visible and np.linalg.norm(v) > 1e-6
        arrow.set_visible(visible)
        if visible:
            arrow.set_positions(tuple(p), tuple(p + VEL_SCALE * np.asarray(v)))

    def _tcp_position(self, T):
        try:
            p = np.asarray(self.student.tcp_position(T), dtype=float)
            if p.shape == (2,):
                return p
        except Exception:
            pass
        return np.asarray(T[:2, 2], float)

    def _tcp_angle(self, T):
        try:
            return float(self.student.tcp_angle(T))
        except Exception:
            return float(np.arctan2(T[1, 0], T[0, 0]))

    def _draw(self, T_all, rows):
        r = self.robot
        O = [T[:2, 2] for T in T_all]
        for i, row in enumerate(r.rows):
            self.links[i].set_data([O[i][0], O[i + 1][0]], [O[i][1], O[i + 1][1]])
            ex = T_all[i][:2, 0]  # smjer stapa
            if row.kind == "T":
                end = O[i] + row.q_max * ex
                self.rails[i].set_data([O[i][0], end[0]], [O[i][1], end[1]])
                a, b = O[i + 1] - 0.045 * ex, O[i + 1] + 0.045 * ex
                self.boxes[i].set_data([a[0], b[0]], [a[1], b[1]])
            if row.kind == "R":
                self.joints[i].set_data([O[i + 1][0]], [O[i + 1][1]])
        for i, T in enumerate(T_all):
            o, ex, ey = T[:2, 2], T[:2, 0], T[:2, 1]
            self.frame_x[i].set_data(
                [o[0], o[0] + self.fl * ex[0]], [o[1], o[1] + self.fl * ex[1]]
            )
            self.frame_y[i].set_data(
                [o[0], o[0] + self.fl * ey[0]], [o[1], o[1] + self.fl * ey[1]]
            )
            off = o - 0.45 * self.fl * (ex + ey)
            self.frame_txt[i].set_position((off[0], off[1]))

    # ---------------------------------------------------------------- SCARA
    def _rot_press(self, evt):
        if evt.inaxes is self.ax and evt.button == 1:
            self._drag = (evt.x, self.ax.azim)

    def _rot_move(self, evt):
        if self._drag is None or evt.x is None:
            return
        x0, azim0 = self._drag
        azim = azim0 - 180.0 * (evt.x - x0) / self.ax.bbox.width
        self.ax.view_init(elev=self._elev, azim=azim, roll=0)
        self.fig.canvas.draw_idle()

    def _rot_release(self, _evt):
        self._drag = None

    def _update3d(self):
        r = self.robot
        q = self.get_q()
        rows = r.link_rows(q)
        T_ref = r.reference_T(q)
        A_ref = r.reference_A(q)

        live = self.chk.get_status()[2]
        J = r.reference_J(q)
        v = J @ self.qd
        if live:
            for k, M in enumerate(A_ref + [T_ref[-1]]):
                self._show(self.num_txt[k], None, M)
            self._show(self.J_txt, None, J)
            self._show_det(J, "k")
            self._show_v(v, "k")

        for i, (x, z, th) in enumerate(rows):
            lx, lz, lt = self.row_lat[i]
            zx, zz, zt = self.row_zero[i]
            x = 0.0 if abs(x) < 5e-4 else x
            z = 0.0 if abs(z) < 5e-4 else z
            th = 0.0 if abs(np.degrees(th)) < 5e-2 else th
            cx, cz, ct = self.tab_cells[i]
            cx.set_text("$0$" if zx else f"${lx} = {x:.3f}\\ \\mathrm{{m}}$")
            cz.set_text("$0$" if zz else f"${lz} = {z:.3f}\\ \\mathrm{{m}}$")
            ct.set_text("$0$" if zt else f"${lt} = {np.degrees(th):.1f}^\\circ$")

        T_all = [np.eye(4)] + T_ref
        P = self._draw3d(T_all, rows)
        self.ghost_line.set_data_3d(P[:, 0], P[:, 1], P[:, 2])
        show_ghost = self.chk.get_status()[0]
        self.ghost_line.set_visible(show_ghost)

        T_n = T_all[-1]
        p = T_n[:3, 3]
        phi = np.arctan2(T_n[1, 0], T_n[0, 0])
        # set_title bi resetirao polozaj naslova
        self.ax.title.set_text(
            "krajnja točka (TCP):  x = %+.3f m   y = %+.3f m   z = %+.3f m   φ = %+.1f°"
            % (p[0], p[1], p[2], np.degrees(phi))
            + self._pause_note()
        )
        tip = p + VEL_SCALE * v[:3]
        self.v_arrow.set_data_3d([p[0], tip[0]], [p[1], tip[1]], [p[2], tip[2]])
        self.v_arrow.set_visible(np.linalg.norm(v[:3]) > 1e-6)

        self.trace.append(p.copy())
        self.trace = self.trace[-500:]
        if self.chk.get_status()[1] and len(self.trace) > 1:
            tr = np.array(self.trace)
            self.trace_line.set_data_3d(tr[:, 0], tr[:, 1], tr[:, 2])
            self.trace_line.set_visible(True)
        else:
            self.trace_line.set_visible(False)
        self._refresh()

    def _draw3d(self, T_all, rows):
        r = self.robot
        O = [T[:3, 3] for T in T_all]
        pts = [O[0]]
        for i, row in enumerate(r.rows):
            # prvo x (vodoravno), pa z
            Pm = (T_all[i] @ np.array([rows[i][0], 0.0, 0.0, 1.0]))[:3]
            seg = np.array([O[i], Pm, O[i + 1]])
            self.links[i].set_data_3d(seg[:, 0], seg[:, 1], seg[:, 2])
            pts += [Pm, O[i + 1]]
            if row.kind == "T":
                top = O[i + 1] + (row.q_max + 0.05) * T_all[i + 1][:3, 2]
                self.boxes[i].set_data_3d(
                    [O[i + 1][0], top[0]], [O[i + 1][1], top[1]], [O[i + 1][2], top[2]]
                )
            if row.kind == "R":
                self.joints[i].set_data_3d([O[i + 1][0]], [O[i + 1][1]], [O[i + 1][2]])

        for i, T in enumerate(T_all):
            o = T[:3, 3]
            for lst, k in ((self.frame_x, 0), (self.frame_y, 1), (self.frame_z, 2)):
                e = o + self.fl * T[:3, k]
                lst[i].set_data_3d([o[0], e[0]], [o[1], e[1]], [o[2], e[2]])
            # oznake KS-ova u istoj tocki jedna iznad druge
            same = sum(np.linalg.norm(O[j] - o) < 1e-6 for j in range(i))
            off = o - 0.45 * self.fl * (T[:3, 0] + T[:3, 1])
            self.frame_txt[i].set_position_3d((off[0], off[1], off[2] + 0.06 * same))
        return np.array(pts)


def run(student, robot_name="RR"):
    gui = LabGUI(student, robot_name)
    plt.show()
    return gui
