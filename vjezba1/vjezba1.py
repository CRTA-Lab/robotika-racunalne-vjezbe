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
'''
Napisite funkciju koja vraca 2D homogenu matricu transformacija (np.array) koja radi rotaciju za kut theta
'''
def rot(theta):
    raise NotImplementedError # TODO 1.1 obrisi ovaj redak

'''
Napisite funkciju koja vraca 2D homogenu matricu transformacija (np.array) koja radi translaciju za vektor (x, y)
'''
def transl(x, y):
    raise NotImplementedError # TODO 1.2 obrisi ovaj redak


# Zadatak 2
'''
Napisite funkciju koja vraca umnozak homogenih matrica transormacije iz prvog zadatka,
radimo s apsolutnim koordinatama, prvo djeluje matrica koja rotira za theta pa matrica koja translatira za L u smjeru nove x-osi
'''
def link_matrix(L, theta):
    raise NotImplementedError # TODO 2 obrisi ovaj redak


# Zadatak 3
'''
Napisite funkciju koja vraca listu matrica transformacija. Ulaz je lista parova oblika (L, theta) koji govori da je iduci koordinatni
sustav rotiran za theta i translatiran za L u odnosu na prethodni.

U izlaznoj listi potrebno je appendati transformacije na sljedeci nacin, prvi element je transformacija iz 0->1, drugi element je transformacija 0->2, treci 0->3 itd.

Hint: (primijetite da k-ti element izlazne liste ovisi o (k-1) elementu), (koristite for petlju)
'''
def forward_kinematics(rows):
    # rows = [(L_1, theta_1), (L_2, theta_2), ...]
    # vraca [T_01, T_02, ..., T_0e]
    raise NotImplementedError # TODO 3 obrisi ovaj redak


# Zadatak 4
'''
Ulaz - homogena matrica transformacije iz 0->(ee)
Izlaz - pozicija izvrsnog clana (x, y) kao np.array
'''
def tcp_position(T):
    raise NotImplementedError    # TODO 4.1 obrisi ovaj redak


'''
Ulaz - homogena matrica transformacije iz 0->(ee)
Izlaz - orijentacija izvrsnog clana (u nasem slucaju kut koji cini s x-osi koordinatnog sustava baze)\
Hint: (morate iz rotacijskog dijela matrice transformacije izracunati kut)
'''
def tcp_angle(T):
    raise NotImplementedError  # TODO 4.2 obrisi ovaj redak


# Zadatak 5
'''
Potrebno je simbolicki izracunati Jakobijan, ulaz je homogena matrica transformacije iz 0->(ee) u obliku simbolicke sympy matrice te lista simbola q.

Izlazni Jakobijan mora biti isto sympy matrica.

Trebate popuniti funckiju ispod

'''
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
