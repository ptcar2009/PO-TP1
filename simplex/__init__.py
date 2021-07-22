"""Pacote implementa o metodo simplex"""
import sys
from typing import List, Tuple, Union

import numpy as np


def pivot(
    tableaux: np.ndarray,
    coluna: int,
    linha: int,
):
    """
    Pivoteia uma linha em um tableaux

    :param tableaux np.ndarray: Tableaux que sera pivoteado
    :param coluna int: Coluna correspondente a variavel que sera pivoteada
    :param linha int: Linha que sera pivoteada
    """

    tableaux[linha] /= tableaux[linha, coluna]
    tableaux[:linha] -= (tableaux[:linha, coluna] / tableaux[linha, coluna]
                         )[:, np.newaxis] * tableaux[linha, ]
    tableaux[linha + 1:] -= (tableaux[linha + 1:, coluna] / tableaux[linha, coluna]
                             )[:, np.newaxis] * tableaux[linha, ]


def build_tableaux(
    coefficients: List[np.ndarray],
    restrictions: List[int],
    costs: np.ndarray,
    n_res: int,
    n_var: int
) -> np.ndarray:
    """
    Constroi um tableaux a partir da matriz de coeficientes, de custos e de restricoes.

    :param coefficients List[np.ndarray]: Matriz de coeficientes.
    :param restrictions List[int]: Vetor de restricoes
    :param costs np.ndarray: Vetor de custos
    :param n_res int: Numero de restricoes
    :param n_var int: Numero de variaveis
    :rtype np.ndarray: Tableaux completo, extendido pelo metodo VEROtm
    """
    t = np.zeros((n_res + 1, n_res + n_var + 1))
    t[1:, :n_var] = np.array(coefficients)
    t[1:, -1] = np.array(restrictions)
    t[0, :n_var] = -costs
    t[1:, n_var:-1] = np.identity(n_res)
    return t


def teste_inviavel(
    tableaux: np.ndarray,
) -> bool:
    """
    Testa se um tableaux esta em uma configuracao inviavel para a PL.

    :param tableaux np.ndarray: Tableaux a ser testado
    :param n_res int: Numero de restricoes
    :rtype bool: Se e ou nao inviavel
    """
    for line in tableaux[1:]:
        if (line[:-1] <= 0).all() != (line[-1] <= 0):
            return True
    return False


def base_mask(b, size):
    mask = np.zeros(size)
    b = b.copy()
    b = b[b < size]
    mask[b] = 1
    return mask.astype('bool')


def solucao(
    tableaux: np.ndarray,
    n_res: int,
    n_var: int,
    b: np.ndarray
) -> np.ndarray:
    """
    Recupera a solucao de um tableaux.

    :param tableaux np.ndarray: Tableaux que tera a solucao recuperada.
    :param n_res int: Numero de restricoes.
    :param n_var int: Numero de variaveis.
    :rtype np.ndarray: Solucao atual do tableaux.
    """
    solution = np.zeros(n_var)
    for line in tableaux[1:]:
        solution += line[:n_var] * base_mask(b, n_var) * line[-1]

    return solution


def base(
    tableaux: np.ndarray,
    n_res: int,
    n_var: int
):
    """
    Recupera uma base atual do tableaux

    :param tableaux np.ndarray: Tableaux do qual a base sera recuperada
    :param n_res int: Numero de restricoes
    :param n_var int: Numero de variaveis
    """
    res = tableaux[1:, :-1]
    c = tableaux[0, :-1]
    base_atual = np.zeros(n_res)
    cur = 0
    for index in range(n_var):
        if c[index] == 0 and res[res[:, index] == 1, cur].shape[0] == 1:
            base_atual[cur] = index
            cur += 1
            if cur == n_res:
                break
    return base_atual.astype('int')


def get_vero(tableaux, n_res, n_var):
    vero = np.zeros((n_res + 1, n_var + n_res + 1))
    vero[:, n_res:] = tableaux
    vero[1:, :n_res] = np.identity(n_res)
    return vero

def simplex_(
    vero: np.ndarray,
    n_res: int,
    n_var: int,
    costs: int,
    b: np.ndarray,
) -> Tuple[str, Union[int, None], Union[np.ndarray, None], np.ndarray, np.ndarray, np.ndarray]:

    def t():
        return vero[:, n_res:]

    while True:
        if (-t()[0, :-1] <= 0).all():
            return "otima", t()[0, -1], solucao(t(), n_res, n_var, b),  vero[0, :n_res], vero, b

        k = (np.arange(costs)[t()[0, :-1] < 0])[0]
        if (t()[1:, k] <= 0).all():
            cert = np.zeros(n_var)
            cert[b[b < n_var]] = t()[1:, k]
            cert *= -1
            cert[k] = 1
            return "ilimitada", None,  solucao(t(), n_res, n_var, b), cert[:n_var], vero, b

        line = t()[1:, k]
        indices = np.arange(line.shape[0])[line > 0] + 1
        index = indices[np.argmin((t()[indices, -1]/line[indices - 1]))]
        cur = t()[index][b] != 0
        b[cur] = k

        pivot(vero, k + n_res, index)

def simplex(
    tableaux: np.ndarray,
    n_res: int,
    n_var: int,
    costs: int,
    b: np.ndarray,
) -> Tuple[str, Union[int, None], Union[np.ndarray, None], np.ndarray]:
    """
    Soluciona um simplex a partir de um tableaux em formato canonico.

    :param tableaux np.ndarray: Tableaux em forma canonica
    :param n_res int: Numero de restricoes
    :param n_var int: Numero de variaveis
    :param costs int: Vetor de custos
    :param b np.ndarray: base atual
    :rtype Tuple[str, Union[int, None], Union[np.ndarray, None], np.ndarray]: Condicao da solucao atual, seu valor, e certificado
    """
    aux = get_vero(tableaux, n_res,  n_var).astype("float")
    aux[aux[:, -1] < 0] *= -1
    aux = simplex_aux(aux, n_res)
    aux[0] -= aux[1:].sum(axis=0)
    _, ot, __, cert, aux, b = simplex_(aux, n_res, n_var +
                                n_res, n_var + n_res, np.arange(n_res)  + n_var)
    if ot is not None and np.round(ot, 7) < 0:
        return "inviavel", None, None, cert
    vero = get_vero(tableaux, n_res, n_var)
    vero[:, :n_res + n_var] = aux[:, :n_res + n_var]
    vero[0, n_res:] = tableaux[0] + aux[0, :n_res].T@tableaux[1:]
    vero[:, -1] = aux[:, -1]

    for i in b:
        pivot(vero, i + n_res,np.arange(n_res)[vero[1:,  n_res + i] != 0][0] + 1 )
    return simplex_(vero, n_res, n_var, costs, b)[:-2]

def simplex_aux(tableaux, n_res):
    new_tableaux = np.zeros((tableaux.shape[0], tableaux.shape[1] + n_res))
    new_tableaux[1:, :tableaux.shape[1] - 1] = tableaux[1:, :-1]
    new_tableaux[0, tableaux.shape[1]-1:-1] = 1
    new_tableaux[1:, tableaux.shape[1]-1:-1] = np.identity(n_res)
    new_tableaux[:, -1] = tableaux[:, -1]
    return new_tableaux



def ler_tableaux() -> Tuple[int, int, np.ndarray]:
    """
    Le um tableaux da entrada padrao.

    :rtype Tuple[int, int, np.ndarray]: O numero de restricoes, o numero de variaveis e o tableaux construido.
    """
    coeficientes: List[np.ndarray] = []
    restricoes: List[int] = []
    custos: np.ndarray = np.zeros(1)
    n_res = n_var = 0
    for cur, line in enumerate(sys.stdin):
        if cur == 0:
            n_res, n_var = map(int, line.split())
        elif cur == 1:
            custos = np.array(list(map(int, line.split())))
        else:
            cur_line = list(map(int, line.split()))
            coeficientes += [np.array(cur_line[:-1])]
            restricoes += [cur_line[-1]]

    tableaux = build_tableaux(coeficientes, restricoes, custos, n_res, n_var)
    return n_res, n_res+n_var, tableaux.astype("float")
