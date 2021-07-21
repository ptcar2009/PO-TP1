"""Resolve um simplex da entrada padrao"""
from simplex import ler_tableaux, simplex, base

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
        print(int(ot))
    if sol is not None:
        for i in sol[:n_var-n_res]:
            print(int(i), end=' ')
        print()
    if cert is not None:
        for i in cert[:last_cert]:
            print(int(i), end=' ')
        print()
