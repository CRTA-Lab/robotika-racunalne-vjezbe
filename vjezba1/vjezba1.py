"""
Vjezba 1 - matrice transformacije, direktna kinematika i jakobijan

Dovrsi funkcije s TODO i pokreni: python vjezba1.py
Nakon izmjene zatvori prozor i pokreni ponovno. lab_gui.py i lab_core.py ne mijenjaj.

Clanak: A_i = Transl(x_i, 0) * Rot(theta_i)
R zglob: theta_i = q_i,  T zglob: x_i = q_i
T_0i = A_1 * A_2 * ... * A_i
Npr. RR: A_1 = Rot(q1), A_2 = Transl(L1, 0) * Rot(q2), A_3 = Transl(L2, 0)

Kutovi u radijanima, duljine u metrima. Matrice se mnoze s @.
"""

import sys

import numpy as np
import sympy as sp

import lab_gui


# Zadatak 1
def rot(theta):
    # TODO 1.1
    raise NotImplementedError


def transl(x, y):
    # TODO 1.2
    raise NotImplementedError


# Zadatak 2
def link_matrix(x, theta):
    # TODO 2
    raise NotImplementedError


# Zadatak 3
def forward_kinematics(rows):
    # rows = [(x_1, theta_1), (x_2, theta_2), ...]
    # vraca [T_01, T_02, ..., T_0e]
    # TODO 3
    raise NotImplementedError


# Zadatak 4
def tcp_position(T):
    # TODO 4.1
    raise NotImplementedError


def tcp_angle(T):
    # TODO 4.2
    raise NotImplementedError


# Zadatak 5
# J = [dx/dq1  dx/dq2 ...]
#     [dy/dq1  dy/dq2 ...]
#     [dphi/dq1 dphi/dq2 ...]
# T je sympy matrica, q lista simbola
#
# HINT: koristi sympy funkcije atan2 i diff,
# istrazi kako se koriste
def jacobian(T, q):
    raise NotImplementedError  # TODO 5, obrisi ovaj redak

    x = ...  # TODO 5.1
    y = ...
    phi = ...

    J = sp.zeros(3, len(q))
    for j in range(len(q)):
        J[0, j] = ...  # TODO 5.2
        J[1, j] = ...
        J[2, j] = ...
    return J


# bonus
def inverse_transform(T):
    # inverz matrice T = [R p; 0 1], ali bez np.linalg.inv
    # R je rotacija pa je R^-1 = R^T, a inverz je
    # T^-1 = [R' -R'*p; 0 1]   (R' je R transponirano)
    # TODO B1
    raise NotImplementedError


def transform_point(T, p):
    # p = (x, y) je tocka zadana u KS clanka cija je matrica T
    # vrati istu tocku zadanu u baznom KS, (x, y)
    # TODO B2
    raise NotImplementedError


if __name__ == "__main__":
    lab_gui.run(sys.modules[__name__])
