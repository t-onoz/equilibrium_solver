from typing import NamedTuple, Any
import numpy as np

def _svd_tol(M):
    M = np.asarray(M, dtype=float)
    # Standard scale-aware tolerance
    eps = np.finfo(float).eps
    s = np.linalg.svd(M, compute_uv=False)
    return max(M.shape) * eps * (s[0] if s.size else 0.0)

def _nullspace(M, *, tol=None):
    """Return N whose columns form an orthonormal basis of Null(M)."""
    M = np.asarray(M, dtype=float)
    U, s, Vt = np.linalg.svd(M, full_matrices=True)
    if tol is None:
        tol = _svd_tol(M)
    r = int(np.sum(s > tol))
    return Vt[r:].T  # (n, n-r)

def _left_nullspace(M, *, tol=None):
    """Return L whose columns form an orthonormal basis of {y | y^T M = 0}."""
    return _nullspace(np.asarray(M, dtype=float).T, tol=tol)  # (m, m-rank)


class KmapResult(NamedTuple):
    Aprime: np.ndarray
    T: np.ndarray
    L: np.ndarray
    info: dict[str, Any]

def precompute_Aprime_and_Kmap_checked(
    A,
    free_idx,
    *,
    enforce_rank_match=True,   # True: p must equal nullity(A). False: allow p <= nullity(A)
    validate_free_idx=True,    # check rank(S @ Null(A)) == p
    rank_tol=None,             # passed to matrix_rank if not None
    ns_tol=None,               # SVD tolerance for (left-)nullspace if not None
    pinv_rcond=None,           # passed to pinv
    numerical_sanity_check=True,
    sanity_tol=1e-9,
):
    """
    Av = K を満たす v を、自由変数 v' を用いて v = A' v' + T K と表すための Aprime と T を事前計算する。
    例えば、Aを化学反応の係数行列、vを成分活量の対数、Kを反応の平衡定数の対数と考えると、
    v'は自由に選べる成分活量の対数、A'は自由変数v'からvへの寄与を表す行列、
    TKは自由に選べる成分活量を用いた各成分の全生成定数ベクトルとして解釈できる。

    さらに、KがAv=Kの形で実現可能かどうかを判断するための、Aの左零空間の基底Lも計算して返す。

        - A: (m, n) の行列
        - free_idx: vのどの成分を自由変数v'とするかのインデックス。長さp。
        - enforce_rank_match: Trueなら、p == nullity(A) を要求する。Falseなら p <= nullity(A) を許す。
        - validate_free_idx: Trueなら、free_idxが本当に自由であることを検査する。enforce_rank_match=Falseの場合は特に重要。
        - rank_tol, ns_tol, pinv_rcond: 内部で使う関数に渡す数値的な許容値。必要に応じて調整。
        - numerical_sanity_check: Trueなら、AprimeとTが正しく機能するか、数値的なサニティチェックを行う。失敗したら例外を投げる。
        - sanity_tol: 数値的なサニティチェックの許容値。

        返り値:
        - Aprime: (n, p) の行列で、v'からvへの寄与を表す。
        - T: (n, m) の行列で、Kからvへの寄与を表す。
        - L: (m, q) の行列で、Aの左零空間の基底。KがAv=Kの形で実現可能かどうかは、||L^T K||が十分小さいかで判断できる。
        - info: 計算に関する情報を含む辞書。必要に応じて拡張可能。
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape

    free_idx = list(free_idx)
    if len(free_idx) != len(set(free_idx)):
        raise ValueError("free_idx contains duplicates.")
    if any(i < 0 or i >= n for i in free_idx):
        raise ValueError("free_idx contains out-of-range indices.")
    p = len(free_idx)

    # Rank / nullity
    rankA = np.linalg.matrix_rank(A, tol=rank_tol) if rank_tol is not None else np.linalg.matrix_rank(A)
    nullity = n - rankA

    if enforce_rank_match and p != nullity:
        raise ValueError(
            "Rank/DOF mismatch: expected p == nullity(A) = n-rank(A) "
            f"= {n}-{rankA} = {nullity}, but got p={p}."
        )
    if (not enforce_rank_match) and p > nullity:
        raise ValueError(f"Too many free components: p={p} > nullity(A)={nullity}.")

    # Selection matrix S (S v = v[free_idx])
    S = np.zeros((p, n), dtype=float)
    for row, j in enumerate(free_idx):
        S[row, j] = 1.0

    # Validate: rank(S @ Null(A)) == p
    rank_SN = None
    if validate_free_idx and p > 0:
        N = _nullspace(A, tol=ns_tol)          # (n, nullity)
        M = S @ N                               # (p, nullity)
        rank_SN = np.linalg.matrix_rank(M, tol=rank_tol) if rank_tol is not None else np.linalg.matrix_rank(M)
        if rank_SN < p:
            raise ValueError(
                "free_idx is not truly free: cannot realize arbitrary v' while keeping Av=K. "
                f"rank(S @ Null(A))={rank_SN} < p={p}."
            )

    # Precompute pseudoinverse mapping
    B = np.vstack([A, S])                       # (m+p, n)
    Bp = np.linalg.pinv(B, rcond=pinv_rcond)     # (n, m+p)

    T = Bp[:, :m]        # K -> K'
    Aprime = Bp[:, m:]   # v' -> contribution

    # Left nullspace basis to test feasibility of each K: L^T K = 0
    L = _left_nullspace(A, tol=ns_tol)  # (m, q)

    info = {
        "m": m, "n": n, "p": p,
        "rank_A": int(rankA),
        "nullity_A": int(nullity),
        "rank_SNullA": None if rank_SN is None else int(rank_SN),
        "free_idx": free_idx,
        "enforce_rank_match": enforce_rank_match,
        "validate_free_idx": validate_free_idx,
    }

    # Optional numerical sanity checks (does not depend on K)
    if numerical_sanity_check and p > 0:
        # Expect: S @ Aprime ≈ I, and A @ Aprime ≈ 0 (changing v' should not change K)
        I = np.eye(p)
        rS = np.linalg.norm(S @ Aprime - I)
        rA = np.linalg.norm(A @ Aprime)
        if rS > 10 * sanity_tol or rA > 10 * sanity_tol:
            raise ValueError(
                "Numerical sanity check failed: "
                f"||S Aprime - I||={rS:.3e}, ||A Aprime||={rA:.3e}. "
                "Try adjusting tolerances or scaling."
            )
        info["sanity_checked"] = True

    return KmapResult(Aprime, T, L, info)
