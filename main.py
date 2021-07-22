"""Resolve um simplex da entrada padrao"""
from simplex import ler_tableaux, simplex, base
from numpy import round

if __name__ == "__main__":
    n_res, n_var, tableaux = ler_tableaux()
    oq, ot, sol, cert = simplex(
        tableaux, n_res, n_var, n_var, base(tableaux, n_res, n_var))
    print(oq)
    if oq == "ilimitada":
        last_cert = n_var - n_res
    else:
        last_cert = n_res
    if ot is not None:
        print(f"{round(ot, 7):.7f}".strip("-"))
    if sol is not None:
        for i in sol[:n_var-n_res]:
            print(f"{round(i, 7):.7f}".strip("-"), end=' ')
        print()
    if cert is not None:
        for i in cert[:last_cert]:
            print(f"{round(i, 7):.7f}".strip("-"), end=' ')
        print()
