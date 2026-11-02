"""
Vjezba 1 - roboti, referentna kinematika i provjere studentskih funkcija.
Studenti ovo ne mijenjaju.

Clanak: A_i = Transl(x_i, 0) * Rot(theta_i). Kod R zgloba je theta_i = q_i, kod T zgloba
x_i = q_i, a zadnji redak je alat. SCARA je 3D demo: A_i = Transl(x_i, 0, z_i) * Rot_z(theta_i).
"""
from dataclasses import dataclass

import numpy as np
import sympy as sp
from sympy.printing.latex import LatexPrinter

q1, q2, q3, q4 = sp.symbols("q_1 q_2 q_3 q_4", real=True)
L0, L1, L2, L3 = sp.symbols("L_0 L_1 L_2 L_3", positive=True)


def link_symbolic(x, theta):
    c, s = sp.cos(theta), sp.sin(theta)
    return sp.Matrix([[c, -s, x],
                      [s,  c, 0],
                      [0,  0, 1]])


def link_symbolic3(x, z, theta):
    c, s = sp.cos(theta), sp.sin(theta)
    return sp.Matrix([[c, -s, 0, x],
                      [s,  c, 0, 0],
                      [0,  0, 1, z],
                      [0,  0, 0, 1]])


# LaTeX ispis s kraticama c1, s12, ...
class _CompactLatex(LatexPrinter):
    @staticmethod
    def _index(arg):
        # q_1 + q_2 -> "12", q_1 - q_2 -> "1-2"
        out = ""
        for term in sp.Add.make_args(sp.expand(arg)):
            coeff, sym = term.as_coeff_Mul()
            if not (isinstance(sym, sp.Symbol) and sym.name.startswith("q_") and coeff in (1, -1)):
                return None
            out += ("-" if coeff == -1 else "") + sym.name.split("_")[1]
        return out

    def _trig(self, letter, expr, exp=None):
        idx = self._index(expr.args[0])
        s = (r"%s_{%s}" % (letter, idx)) if idx is not None else (
            r"\%s(%s)" % ("cos" if letter == "c" else "sin", self._print(expr.args[0])))
        if exp is not None:
            s += r"^{%s}" % exp
        return s

    def _print_cos(self, expr, exp=None):
        return self._trig("c", expr, exp)

    def _print_sin(self, expr, exp=None):
        return self._trig("s", expr, exp)


def compact_latex(expr):
    s = _CompactLatex({"mul_symbol": None}).doprint(expr)
    return s.replace(r"\left(", "(").replace(r"\right)", ")")


@dataclass
class Row:
    kind: str  # R, T ili F (alat)
    q: sp.Symbol
    x: sp.Expr
    theta: sp.Expr
    q_min: float = 0.0
    q_max: float = 0.0
    z: sp.Expr = 0  # samo SCARA


class Robot:
    def __init__(self, name, title, rows, lengths, q_home, bounds, dim=2):
        self.name, self.title = name, title
        self.dim = dim
        self.rows = rows
        self.var_rows = [r for r in rows if r.kind != "F"]
        self.lengths = lengths
        self.q_home = list(q_home)
        self.bounds = bounds

        self._syms = [r.q for r in self.var_rows] + list(lengths)
        if dim == 2:
            self.A_sym = [link_symbolic(r.x, r.theta) for r in rows]
            cols = [[r.x, r.theta] for r in rows]
        else:
            self.A_sym = [link_symbolic3(r.x, r.z, r.theta) for r in rows]
            cols = [[r.x, r.z, r.theta] for r in rows]

        Ts, T = [], sp.eye(dim + 1)
        for A in self.A_sym:
            T = T * A
            Ts.append(T)
        self._f_T = [sp.lambdify(self._syms, Tk, "numpy") for Tk in Ts]
        self._f_A = [sp.lambdify(self._syms, A, "numpy") for A in self.A_sym]
        self._f_rows = sp.lambdify(self._syms, cols, "numpy")
        self._T_total = None

        self.joints = "".join(r.kind for r in rows)  # npr. "RRF"
        self._f_J = sp.lambdify(self._syms, self._jacobian(T), "numpy")
        self._J_sym = None

    def _jacobian(self, T):
        # redci x, y (, z) i phi; phi je zbroj svih theta
        pos = [T[k, self.dim] for k in range(self.dim)]
        phi = sum(r.theta for r in self.rows)
        return sp.Matrix(pos + [phi]).jacobian([r.q for r in self.var_rows])

    def _args(self, q):
        return list(q) + [float(v) for v in self.lengths.values()]

    def link_rows(self, q):
        rows = self._f_rows(*self._args(q))
        return [tuple(float(v) for v in row) for row in rows]

    def reference_T(self, q):
        args = self._args(q)
        return [np.array(f(*args), dtype=float) for f in self._f_T]

    def reference_A(self, q):
        args = self._args(q)
        return [np.array(f(*args), dtype=float) for f in self._f_A]

    def reference_J(self, q):
        return np.array(self._f_J(*self._args(q)), dtype=float)

    def sigma_min_pos(self, q):
        # ~0 na rubu dosega (singularitet)
        return np.linalg.svd(self.reference_J(q)[:self.dim], compute_uv=False)[-1]

    def symbolic_A(self):
        return [A.applyfunc(sp.simplify) for A in self.A_sym]

    def symbolic_T_total(self):
        if self._T_total is None:
            T = sp.eye(self.dim + 1)
            for A in self.A_sym:
                T = (T * A).applyfunc(sp.simplify)
            self._T_total = T
        return self._T_total

    def symbolic_J(self):
        if self._J_sym is None:
            self._J_sym = self._jacobian(self.symbolic_T_total()).applyfunc(sp.simplify)
        return self._J_sym


ROBOTS = {}


def _add(robot):
    ROBOTS[robot.name] = robot


deg = np.radians

_add(Robot(
    "RR", "RR manipulator",
    rows=[
        Row("R", q1, 0, q1, -np.pi, np.pi),
        Row("R", q2, L1, q2, -np.pi, np.pi),
        Row("F", None, L2, 0),
    ],
    lengths={L1: 0.50, L2: 0.40},
    q_home=[deg(30), deg(45)],
    bounds=((-1.0, 1.0), (-1.0, 1.0)),
))

_add(Robot(
    "RT", "RT manipulator",
    rows=[
        Row("R", q1, 0, q1, -np.pi, np.pi),
        Row("T", q2, q2, 0, 0.0, 0.7),
    ],
    lengths={},
    q_home=[deg(30), 0.40],
    bounds=((-0.9, 0.9), (-0.9, 0.9)),
))

_add(Robot(
    "TR", "TR manipulator",
    rows=[
        Row("T", q1, q1, 0, 0.0, 0.6),
        Row("R", q2, 0, q2, -np.pi, np.pi),
        Row("F", None, L1, 0),
    ],
    lengths={L1: 0.40},
    q_home=[0.30, deg(40)],
    bounds=((-0.4, 1.2), (-0.8, 0.8)),
))

_add(Robot(
    "RRR", "RRR manipulator",
    rows=[
        Row("R", q1, 0, q1, -np.pi, np.pi),
        Row("R", q2, L1, q2, -np.pi, np.pi),
        Row("R", q3, L2, q3, -np.pi, np.pi),
        Row("F", None, L3, 0),
    ],
    lengths={L1: 0.40, L2: 0.40, L3: 0.30},
    q_home=[deg(30), deg(45), deg(-60)],
    bounds=((-1.2, 1.2), (-1.2, 1.2)),
))

# SCARA: stup L0, stapovi L1 i L2, vreteno se spusta za q3 (z = -q3), alat L3 prema dolje
_add(Robot(
    "SCARA", "SCARA (RRTR) manipulator",
    rows=[
        Row("R", q1, 0, q1, -np.pi, np.pi, z=L0),
        Row("R", q2, L1, q2, -np.pi, np.pi),
        Row("T", q3, L2, 0, 0.0, 0.25, z=-q3),
        Row("R", q4, 0, q4, -np.pi, np.pi),
        Row("F", None, 0, 0, z=-L3),
    ],
    lengths={L0: 0.40, L1: 0.35, L2: 0.30, L3: 0.10},
    q_home=[deg(30), deg(45), 0.10, deg(0)],
    bounds=((-0.7, 0.7), (-0.7, 0.7), (0.0, 0.7)),
    dim=3,
))


def ref_rot(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def ref_transl(x, y):
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], dtype=float)


def ref_link(x, t):
    return ref_transl(x, 0) @ ref_rot(t)


def student_jacobian(student, robot, T):
    """Studentski jacobian(T, q) za dani simbolicki T_0e. Vraca (J, status, poruka)."""
    q = [r.q for r in robot.var_rows]
    fn = getattr(student, "jacobian", None)
    if fn is None:
        return None, "todo", "funkcija nije gotova"
    try:
        J = sp.Matrix(fn(T, q))
    except NotImplementedError:
        return None, "todo", "funkcija nije gotova"
    except Exception as e:
        return None, "fail", f"greška: {type(e).__name__}: {e}"[:60]
    if J.shape != (3, len(q)):
        return None, "fail", f"matrica mora biti 3x{len(q)}, a ima oblik {J.shape}"
    try:
        return J.applyfunc(sp.simplify), "ok", ""
    except Exception as e:
        return None, "fail", f"greška: {type(e).__name__}: {e}"[:60]


def _try(fn, *args):
    try:
        return fn(*args), "ok"
    except NotImplementedError:
        return None, "todo"
    except Exception:
        return None, "fail"


def _as_array(out, shape):
    try:
        arr = np.asarray(out, dtype=float)
    except Exception:
        return None
    return arr if arr.shape == shape else None


# Opci oblik matrica iz studentskog koda.
# rot/transl/link_matrix ne zovemo sa simbolima (pisane su za brojeve), nego ih uzorkujemo
# i svaki element fitamo na bazu ispod. forward_kinematics i J se onda izvrse sa simbolima.
_t, _x, _y = sp.symbols("t x y")
_c, _s = sp.cos(_t), sp.sin(_t)
_FIT = {
    "rot": ((_t,), [1, _c, _s], lambda rng: (rng.uniform(-np.pi, np.pi),)),
    "transl": ((_x, _y), [1, _x, _y], lambda rng: tuple(rng.uniform(-1, 1, 2))),
    "link_matrix": ((_x, _t), [1, _x, _c, _s, _x * _c, _x * _s],
                    lambda rng: (rng.uniform(-1, 1), rng.uniform(-np.pi, np.pi))),
}


def _fit_matrix_fn(fn, syms, basis, sample):
    rng = np.random.default_rng(7)
    f_basis = sp.lambdify(syms, basis, "numpy")
    X, Y = [], []
    for _ in range(24):
        args = sample(rng)
        out, st = _try(fn, *args)
        if st != "ok":
            return None, st
        got = _as_array(out, (3, 3))
        if got is None:
            return None, "fail"
        X.append(np.array(f_basis(*args), dtype=float))
        Y.append(got.ravel())
    X, Y = np.array(X), np.array(Y)
    coef = np.linalg.lstsq(X, Y, rcond=None)[0]
    coef = np.vectorize(lambda v: float(sp.Rational(v).limit_denominator(12)))(coef)
    if np.max(np.abs(X @ coef - Y)) > 1e-7:  # nije u bazi, npr. student radi u stupnjevima
        return None, "fail"
    entries = [sum(sp.nsimplify(c) * b for c, b in zip(coef[:, e], basis)) for e in range(9)]
    return sp.Matrix(3, 3, entries), "ok"


class _SymModule:
    # podmece se umjesto np/math: polja su dtype=object, cos/sin idu preko sympy-a
    _MAKE = {"array", "asarray", "zeros", "ones", "empty", "eye", "identity", "full",
             "zeros_like", "ones_like"}
    _FUNCS = {"cos": sp.cos, "sin": sp.sin, "tan": sp.tan, "sqrt": sp.sqrt,
              "arctan2": sp.atan2, "atan2": sp.atan2, "radians": sp.rad, "deg2rad": sp.rad,
              "degrees": sp.deg, "rad2deg": sp.deg}

    def __init__(self, real):
        self._real = real
        self.pi = sp.pi

    def __getattr__(self, name):
        real = getattr(self._real, name)
        if name in self._FUNCS:
            f = self._FUNCS[name]

            def elementwise(*a):
                if any(isinstance(v, np.ndarray) for v in a):
                    return np.vectorize(f, otypes=[object])(*a)
                return f(*a)
            return elementwise
        if name in self._MAKE and self._real is np:
            def make(*a, **kw):
                kw["dtype"] = object
                return real(*a, **kw)
            return make
        return real


def _symbolic_call(student, fits, fn, *args):
    import math

    def as_fn(M, syms):
        return lambda *a: np.array(M.subs(dict(zip(syms, a)), simultaneous=True).tolist(),
                                   dtype=object)

    patch = {name: as_fn(M, _FIT[name][0]) for name, M in fits.items()}
    patch.update({k: _SymModule(v) for k, v in vars(student).items() if v is np or v is math})
    saved = {k: getattr(student, k) for k in patch}
    try:
        for k, v in patch.items():
            setattr(student, k, v)
        return fn(*args)
    finally:
        for k, v in saved.items():
            setattr(student, k, v)


def student_symbolic(student, robot):
    """Vraca (A, T_0e, J, napomena_A, napomena_T, napomena_J, (status_J, poruka_J)).
    Gdje ne ide, referentni oblik."""
    A_ref, T_ref, J_ref = robot.symbolic_A(), robot.symbolic_T_total(), robot.symbolic_J()
    if robot.dim != 2:
        return A_ref, T_ref, J_ref, "", "", "", ("ok", "")

    def note(name, st):
        why = f"{name} još nije gotova" if st == "todo" else f"ne može se izvesti iz tvoje {name}"
        return f"opći oblik: referentni ({why})"

    fits, status = {}, {}
    for name, (syms, basis, sample) in _FIT.items():
        fn = getattr(student, name, None)
        M, status[name] = _fit_matrix_fn(fn, syms, basis, sample) if fn else (None, "todo")
        if M is not None:
            fits[name] = M

    if "link_matrix" in fits:
        A = [fits["link_matrix"].subs({_x: r.x, _t: r.theta}, simultaneous=True)
             .applyfunc(sp.simplify) for r in robot.rows]
        note_A = ""
    else:
        A, note_A = A_ref, note("link_matrix", status["link_matrix"])

    def derive(name, fn, ref, num_args, sym_args):
        # fn sa simbolima, pa provjera da daje iste brojeve kao studentski kod
        st = _try(fn, *num_args(robot.q_home))[1]
        if st == "ok":
            try:
                M = sp.Matrix(np.asarray(_symbolic_call(student, fits, fn, *sym_args),
                                         dtype=object).tolist())
                M = M.applyfunc(lambda e: sp.simplify(sp.nsimplify(e, rational=True)))
                f_M = sp.lambdify(robot._syms, M, "numpy")
                rng = np.random.default_rng(3)
                for _ in range(3):
                    q = [rng.uniform(r.q_min, r.q_max) for r in robot.var_rows]
                    exp = np.array(f_M(*robot._args(q)), dtype=float)
                    got = _as_array(_try(fn, *num_args(q))[0], exp.shape)
                    if got is None or exp.shape != ref.shape or not np.allclose(got, exp, atol=1e-7):
                        raise ValueError
                return M, ""
            except Exception:
                pass
        return ref, note(name, st)

    rows_sym = [(r.x, r.theta) for r in robot.rows]
    T, note_T = derive("forward_kinematics", lambda rows: student.forward_kinematics(rows)[-1],
                       T_ref, lambda q: (robot.link_rows(q),), (rows_sym,))
    J, st_J, msg_J = student_jacobian(student, robot, T)
    if J is None:
        J, note_J = J_ref, f"opći oblik: referentni ({msg_J})"
    else:
        note_J = ""
    return A, T, J, note_A, note_T, note_J, (st_J, msg_J)


# provjere za checklistu u GUI-ju
@dataclass
class Check:
    name: str
    status: str  # ok / fail / todo


def _check(student, name, cases):
    """cases: lista (argumenti, ocekivani rezultat)."""
    fn = getattr(student, name, None)
    if fn is None:
        return Check(name, "todo")
    for args, exp in cases:
        out, st = _try(fn, *args)
        if st != "ok":
            return Check(name, st)
        exp = np.asarray(exp, dtype=float)
        got = _as_array(out, exp.shape)
        if got is None or not np.allclose(got, exp, atol=1e-9):
            return Check(name, "fail")
    return Check(name, "ok")


def run_checks(student):
    rng = np.random.default_rng(1)
    vec = lambda: rng.uniform(-1, 1, 2)

    fk = []
    for robot in ROBOTS.values():
        if robot.dim == 2:
            for _ in range(3):
                q = [rng.uniform(r.q_min, r.q_max) for r in robot.var_rows]
                fk.append(((robot.link_rows(q),), robot.reference_T(q)))
    Ts = [ROBOTS["RRR"].reference_T(rng.uniform(-3, 3, 3))[-1] for _ in range(4)]
    links = [(0.0, 0.0)] + [(rng.uniform(-1, 1), rng.uniform(-3, 3)) for _ in range(5)]

    cases = {
        "rot": [((t,), ref_rot(t)) for t in (0.0, 0.3, -1.2, 2.5, np.pi / 2, -np.pi / 2)],
        "transl": [((x, y), ref_transl(x, y)) for x, y in (vec() for _ in range(4))],
        "link_matrix": [((x, t), ref_link(x, t)) for x, t in links],
        "forward_kinematics": fk,
        "tcp_position": [((T,), T[:2, 2]) for T in Ts],
        "tcp_angle": [((T,), np.arctan2(T[1, 0], T[0, 0])) for T in Ts],
        "inverse_transform": [((T,), np.linalg.inv(T)) for T in Ts],
        "transform_point": [((T, p), (T @ np.append(p, 1.0))[:2]) for T, p in
                            ((T, vec()) for T in Ts)],
    }
    checks = [_check(student, name, c) for name, c in cases.items()]
    checks.insert([c.name for c in checks].index("inverse_transform"),
                  _check_jacobian(student, rng))
    return checks


def _check_jacobian(student, rng):
    """jacobian(T, q) na simbolickom T_0e svakog 2D robota, brojcano prema referentnom J."""
    for robot in ROBOTS.values():
        if robot.dim != 2:
            continue
        J, st, _ = student_jacobian(student, robot, robot.symbolic_T_total())
        if J is None:
            return Check("jacobian", st)
        f_J = sp.lambdify(robot._syms, J, "numpy")
        for _ in range(3):
            q = [rng.uniform(r.q_min, r.q_max) for r in robot.var_rows]
            got = _as_array(_try(f_J, *robot._args(q))[0], (3, len(q)))
            if got is None or not np.allclose(got, robot.reference_J(q), atol=1e-9):
                return Check("jacobian", "fail")
    return Check("jacobian", "ok")
