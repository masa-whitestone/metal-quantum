"""
metalq/backends/cpu/adjoint.py - Adjoint Differentiation

可逆な adjoint 法による勾配計算 (PennyLane-Lightning 方式)。
parameter-shift の 2p 回の全回路実行に対し、

    forward 1 回 (フュージョン経路) + λ = H|ψ> 1 回 (全項 1 パス)
    + 逆順スイープ (ψ と λ をブロック単位でまとめて巻き戻す)

で全パラメータの勾配が得られる。

数式: E(θ) = <ψ|H|ψ>, ψ = U_L ... U_1 |0>。ゲート j のパラメータ v に
ついて dE/dv = 2 Re <λ_j | dU_j/dv | ψ_{j-1}> ここで
ψ_{j-1} = U_{j-1}...U_1|0>, λ_j = U_{j+1}†...U_L† H ψ。逆順に
ψ ← U_j†ψ, λ ← U_j†λ と巻き戻せば両者が同時に得られる。

**微分は ψ_{j-1} ではなく ψ_j 上で取る**:
    <λ_j| dU_j |ψ_{j-1}> = <λ_j| (dU_j U_j†) |ψ_j>
なので、小行列 Dm = dU U† を先に作っておけば「巻き戻す前のタイル」
だけで縮約でき、ψ を巻き戻す前後で 2 度触る必要がなくなる。

**ブロック融合 + 1 カーネル/ブロック**: 逆順ゲート列を「qubit の
和集合が K 個以下」のブロックにまとめ、ブロックごとに 1 つの Numba
カーネルで (a) 微分の縮約 (b) ψ ← U†ψ (c) λ ← U†λ を回す。ゲート
あたり 3 カーネル・6 パスだったものがブロックあたり 1 カーネル・
4 パス (ψ と λ の読み書き) になる。カーネルはタイル (bits[] を自由に
した 2^K 振幅) の**チャンク**を prange で分け、チャンクの内側で
ゲートを回すので、ブロック内の後続ゲートは L2 に載ったままの
チャンクを触り直すだけで済む (statevector の置換はしない;
fusion.py の GEMM 経路とはそこが違う)。ブロック分割の前に
fusion._reorder_for_locality で可換ゲート (disjoint qubit) を詰め直す
(依存関係を保つ並べ替えは各ゲート位置の ψ_j / λ_j を変えないので、
微分値は不変)。

**単項式ゲートの 1 乗算経路**: 各行の非ゼロが 1 つだけの行列
(対角: rz/cz/rzz/cp/ccz…、置換: x/cx/swap/ccx…) は
out[r] = ph[r] * in[col[r]] と書けるので、振幅あたり 2^kg 回では
なく 1 回の複素乗算で適用できる。_DERIV_SUPPORTED のゲートは
生成子が Pauli 積なので Dm = dU U† も必ず単項式になり、微分の縮約も
同じ経路に乗る。単項式判定は行列値から一般に行う (ゲート名の
特別扱いを増やさない)。

ψ は fusion の lazy permutation レイアウトのまま扱う (qubit q は
インデックスの bit pos_of[q])。canonical 復元パスが 1 つ減り、
λ = Hψ のマスクも pos_of で組める。

回路レベルのパラメータ θ_i とゲートパラメータ v_g は
ParameterExpression.grad() の連鎖律で結び、dE/dθ_i = Σ_g (dv_g/dθ_i)
dE/dv_g とする (線形式のみ; 非線形や未対応ゲートは
AdjointUnsupported を投げて parameter-shift にフォールバック)。

λ は精度のため statevector の dtype に依らず complex128 で保持する
(部分和も float64)。
"""
from typing import List, TYPE_CHECKING

import numpy as np

from ...parameter import Parameter, ParameterExpression, is_parameterized
from .statevector import get_gate_matrix
from .expectation import (_term_masks, apply_hamiltonian_multi,
                          get_num_threads)

if TYPE_CHECKING:
    from ...circuit import Circuit
    from ...spin import Hamiltonian
    from .backend import CPUBackend

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


class AdjointUnsupported(Exception):
    """Raised when the circuit cannot be differentiated by the adjoint
    method (caller should fall back to parameter-shift)."""


# 逆順スイープのブロックあたり最大 qubit 数 K。タイルは 2^K 振幅で、
# statevector の置換はせずビットストライドで触る。大きいほどブロック数
# = パス数は減るが、タイルあたりのメモリストリーム数が 2^K x 2 本
# (ψ と λ) に増え、連続ラン長 (= 2^(ブロック最下位ビット)) も短くなる。
# 実測 (4 コア Xeon / L3 260MB / Linux, 3 層 RY/RZ+CX ansatz,
# OPENBLAS_NUM_THREADS=1, complex128 の勾配全体):
#   n=12: K=2 4.6 / K=3 4.1 / K=4 3.5 ms   (Python 側の組み立てが支配的)
#   n=16: K=2 24.9 / K=3 24.9 / K=4 24.8 ms
#   n=18: K=2 96.0 / K=3 92.8 / K=4 91.5 ms
#   n=20: K=2 307 / K=3 313 / K=4 365 ms   (K=4 でストリーム数が効く)
# 全サイズで最良かその誤差内に収まる K=3 を採る。
BLOCK_QUBITS = 3

# 逆順スイープのチャンク最小タイル数 (2 のべき)。チャンクはブロック内の
# 全ゲートに再訪されるので L2 に載る大きさで切りたいが、小さすぎると
# prange の分割オーバーヘッドが勝つ。
_MIN_TILES_PER_CHUNK = 8

# 1 スレッドあたりのチャンク数 (expectation.py と同じ方針)。
_CHUNKS_PER_THREAD = 64


class AdjointStats:
    """Per-process counters for the adjoint reverse sweep."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.n_kernels = 0      # Numba kernel launches (λ build + blocks)
        self.n_blocks = 0       # fused unwind blocks
        self.n_gates = 0        # gates unwound
        self.n_passes = 0.0     # statevector-sized read/write units

    def as_dict(self):
        return {'n_kernels': self.n_kernels, 'n_blocks': self.n_blocks,
                'n_gates': self.n_gates, 'n_passes': self.n_passes}


stats = AdjointStats()


# ============================================================================
# Gate derivative matrices
# ============================================================================

_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_P1Q = {'x': _X, 'y': _Y, 'z': _Z}

# 単一パラメータの exp 形ゲートのみ対応 (u2/u3/r は fallback)
_DERIV_SUPPORTED = frozenset({
    'rx', 'ry', 'rz', 'p', 'u1',
    'crx', 'cry', 'crz', 'cp', 'cu1',
    'rxx', 'ryy', 'rzz', 'rzx',
})


def _deriv_matrix(name: str, params: List) -> np.ndarray:
    """dU/dv for a bound single-parameter gate.

    U(v) = exp(-i v G / 2) 形は dU = -i/2 G U。位相ゲート
    (p/cp: U = exp(i v |1..1><1..1|)) は dU = i P U。
    """
    U = get_gate_matrix(name, params)
    if name in ('rx', 'ry', 'rz'):
        return -0.5j * (_P1Q[name[1]] @ U)
    if name in ('crx', 'cry', 'crz'):
        G = np.zeros((4, 4), dtype=np.complex128)
        G[2:, 2:] = _P1Q[name[2]]
        return -0.5j * (G @ U)
    if name in ('rxx', 'ryy', 'rzz', 'rzx'):
        G = np.kron(_P1Q[name[1]], _P1Q[name[2]])
        return -0.5j * (G @ U)
    if name in ('p', 'u1'):
        return 1j * (np.diag([0.0, 1.0]).astype(np.complex128) @ U)
    if name in ('cp', 'cu1'):
        return 1j * (np.diag([0.0, 0.0, 0.0, 1.0]).astype(np.complex128) @ U)
    raise AdjointUnsupported(name)


# Dm はゲート名だけで決まる (下記) のでキャッシュする。
_DM_CACHE = {}


def _unwind_deriv_matrix(name: str, params: List) -> np.ndarray:
    """Dm = (dU/dv) U†, i.e. the derivative evaluated at ψ_j.

    <λ_j| dU |ψ_{j-1}> = <λ_j| dU U† |ψ_j> なので、この小行列を先に
    作っておけば「巻き戻す前のタイル」だけで縮約できる。

    _DERIV_SUPPORTED のゲートはすべて 1 パラメータの指数族
    U(v) = exp(v A) (A は定数: 回転なら -i G / 2、位相ゲートなら i Π)
    なので dU/dv = A U、したがって Dm = A U U† = A はパラメータ値に
    依存しない。値は既存の _deriv_matrix から一般に計算し (ゲートごと
    の特殊化を増やさない)、名前でキャッシュする。パラメータ非依存性は
    tests/test_adjoint_cpu.py で固定している。
    """
    dm = _DM_CACHE.get(name)
    if dm is None:
        U = get_gate_matrix(name, params)
        dm = _deriv_matrix(name, params) @ np.conj(U).T
        _DM_CACHE[name] = dm
    return dm


# ============================================================================
# Numba kernels
# ============================================================================
# 1) fused block unwind (hot path)
# 2) per-gate kernels: in-place k-qubit gate application (k = 1, 2, 3),
#    bra-op-ket contraction and single-term Pauli accumulation. These are
#    the reference implementation the block kernel is tested against
#    (see tests/test_adjoint_cpu.py) and the Hamiltonian fallback.

if HAS_NUMBA:

    @jit(nopython=True, cache=True, fastmath=True)
    def _unwind_apply_chunk(arr, mat, ph, src, roff, mono, kg, subbase,
                            nsub, b_start, nrun, R, mask, notmask,
                            use_mask_adv):
        """arr <- U† arr on one tile chunk (ψ or λ; dtype-specialized).

        行列要素・オフセット・位相はループの外でスカラーに巻き上げる。
        ``mono`` が真なら U† は単項式 (各行に非ゼロ 1 つ) なので
        out[r] = ph[r] * in[src[r]] の 1 複素乗算/振幅で済む
        (対角ゲート rz/cz/rzz/cp、置換ゲート x/cx/swap/ccx ...)。
        インデックスは連続ラン (長さ R) で回し、ラン間だけマスク付き
        インクリメントで飛ばす。
        """
        o1 = roff[1]
        o2 = roff[2]
        o3 = roff[3]
        q0 = src[0]
        q1 = src[1]
        q2 = src[2]
        q3 = src[3]
        p0 = ph[0]
        p1 = ph[1]
        p2 = ph[2]
        p3 = ph[3]
        m00 = mat[0, 0]
        m01 = mat[0, 1]
        m10 = mat[1, 0]
        m11 = mat[1, 1]
        m02 = mat[0, 2]
        m03 = mat[0, 3]
        m12 = mat[1, 2]
        m13 = mat[1, 3]
        m20 = mat[2, 0]
        m21 = mat[2, 1]
        m22 = mat[2, 2]
        m23 = mat[2, 3]
        m30 = mat[3, 0]
        m31 = mat[3, 1]
        m32 = mat[3, 2]
        m33 = mat[3, 3]
        dg = 1 << kg
        # code: 0 = mono 1q, 1 = mono 2q, 2 = dense 1q, 3 = dense 2q,
        #       4 = generic (3-qubit, mono or dense)
        if kg == 1:
            code = 0 if mono else 2
        elif kg == 2:
            code = 1 if mono else 3
        else:
            code = 4
        sp = np.empty(8, dtype=arr.dtype)
        for si in range(nsub):
            sb = subbase[si]
            base = b_start
            for _run in range(nrun):
                ob = base + sb
                for d in range(R):
                    b = ob + d
                    if code == 0:
                        a0 = arr[b + q0]
                        a1 = arr[b + q1]
                        arr[b] = p0 * a0
                        arr[b + o1] = p1 * a1
                    elif code == 1:
                        a0 = arr[b + q0]
                        a1 = arr[b + q1]
                        a2 = arr[b + q2]
                        a3 = arr[b + q3]
                        arr[b] = p0 * a0
                        arr[b + o1] = p1 * a1
                        arr[b + o2] = p2 * a2
                        arr[b + o3] = p3 * a3
                    elif code == 2:
                        a0 = arr[b]
                        a1 = arr[b + o1]
                        arr[b] = m00 * a0 + m01 * a1
                        arr[b + o1] = m10 * a0 + m11 * a1
                    elif code == 3:
                        a0 = arr[b]
                        a1 = arr[b + o1]
                        a2 = arr[b + o2]
                        a3 = arr[b + o3]
                        arr[b] = m00 * a0 + m01 * a1 + m02 * a2 + m03 * a3
                        arr[b + o1] = (m10 * a0 + m11 * a1 + m12 * a2
                                       + m13 * a3)
                        arr[b + o2] = (m20 * a0 + m21 * a1 + m22 * a2
                                       + m23 * a3)
                        arr[b + o3] = (m30 * a0 + m31 * a1 + m32 * a2
                                       + m33 * a3)
                    else:
                        # generic arity (3-qubit gates). Every source
                        # amplitude is read before any is overwritten
                        # (the gate permutes within the sub-tile).
                        if mono:
                            for r in range(dg):
                                sp[r] = arr[b + src[r]]
                            for r in range(dg):
                                arr[b + roff[r]] = ph[r] * sp[r]
                        else:
                            for j in range(dg):
                                sp[j] = arr[b + roff[j]]
                            for r in range(dg):
                                acc = mat[r, 0] * sp[0]
                                for cc in range(1, dg):
                                    acc += mat[r, cc] * sp[cc]
                                arr[b + roff[r]] = acc
                if use_mask_adv:
                    base = (((base + R - 1) | mask) + 1) & notmask
                else:
                    base += R

    @jit(nopython=True, cache=True, fastmath=True)
    def _unwind_deriv_chunk(psi, lam, dmat, dph, dsrc, roff, mono, kg,
                            subbase, nsub, b_start, nrun, R, mask, notmask,
                            use_mask_adv):
        """(re, im) of <λ| Dm |ψ> over one tile chunk, float64 accumulation.

        _DERIV_SUPPORTED のゲートはすべて生成子が Pauli 積なので
        Dm = dU U† は単項式になり、振幅あたり 1 複素乗算で縮約できる
        (dense 経路は 2^kg 乗算)。
        """
        o1 = roff[1]
        o2 = roff[2]
        o3 = roff[3]
        q0 = dsrc[0]
        q1 = dsrc[1]
        q2 = dsrc[2]
        q3 = dsrc[3]
        p0 = dph[0]
        p1 = dph[1]
        p2 = dph[2]
        p3 = dph[3]
        dg = 1 << kg
        if mono:
            code = 0 if kg == 1 else 1
        else:
            code = 2
        re = 0.0
        im = 0.0
        sp = np.empty(8, dtype=psi.dtype)
        for si in range(nsub):
            sb = subbase[si]
            base = b_start
            for _run in range(nrun):
                ob = base + sb
                for d in range(R):
                    b = ob + d
                    if code == 0:
                        z0 = p0 * psi[b + q0]
                        z1 = p1 * psi[b + q1]
                        l0 = lam[b]
                        l1 = lam[b + o1]
                        re += (l0.real * z0.real + l0.imag * z0.imag
                               + l1.real * z1.real + l1.imag * z1.imag)
                        im += (l0.real * z0.imag - l0.imag * z0.real
                               + l1.real * z1.imag - l1.imag * z1.real)
                    elif code == 1:
                        z0 = p0 * psi[b + q0]
                        z1 = p1 * psi[b + q1]
                        z2 = p2 * psi[b + q2]
                        z3 = p3 * psi[b + q3]
                        l0 = lam[b]
                        l1 = lam[b + o1]
                        l2 = lam[b + o2]
                        l3 = lam[b + o3]
                        re += (l0.real * z0.real + l0.imag * z0.imag
                               + l1.real * z1.real + l1.imag * z1.imag
                               + l2.real * z2.real + l2.imag * z2.imag
                               + l3.real * z3.real + l3.imag * z3.imag)
                        im += (l0.real * z0.imag - l0.imag * z0.real
                               + l1.real * z1.imag - l1.imag * z1.real
                               + l2.real * z2.imag - l2.imag * z2.real
                               + l3.real * z3.imag - l3.imag * z3.real)
                    else:
                        for j in range(dg):
                            sp[j] = psi[b + roff[j]]
                        for r in range(dg):
                            z = dmat[r, 0] * sp[0]
                            for cc in range(1, dg):
                                z += dmat[r, cc] * sp[cc]
                            lr = lam[b + roff[r]]
                            re += lr.real * z.real + lr.imag * z.imag
                            im += lr.real * z.imag - lr.imag * z.real
                if use_mask_adv:
                    base = (((base + R - 1) | mask) + 1) & notmask
                else:
                    base += R
        return re, im

    @jit(nopython=True, parallel=True, cache=True)
    def _unwind_block_numba(psi, lam, bits, subbase, rowoff, mono_src,
                            mats_p, mats_l, dmats, mph_p, mph_l, dsrc, dph,
                            arity, kind, dkind, has_deriv, do_lam,
                            ntiles, nchunks, chunk, out):
        """Unwind a whole block of gates over ψ and λ in one kernel launch.

        タイル t は statevector インデックスのうち bits[] のビットを
        自由にした 2^k 振幅。prange はタイルの**チャンク**にかけ
        (微分和は Numba parfor がスカラーしか畳めないので
        (nchunks, ngates, 2) に出す)、チャンクの中でゲートを順に

            (a) has_deriv なら <λ_j| Dm |ψ_j> を縮約 (巻き戻す前)
            (b) ψ ← U† ψ
            (c) λ ← U† λ   (do_lam のとき)

        と処理する。ゲートをチャンクの内側に置くと、ブロック内の後続
        ゲートは L2 に載ったままのチャンクを触り直すだけで済み、
        statevector パスがブロックあたり 1 回にまとまる (ゲートあたり
        3 カーネル・6 パスから)。(a)(b)(c) を別ループに割ったのは
        どれもチャンク内で閉じていて順序が保たれるため (微分は
        巻き戻す前の ψ_j / λ_j で評価される)。

        mats_p / mph_p は psi の dtype、mats_l / mph_l / dmats は
        complex128 なので complex64 の ψ と complex128 の λ を
        混在させられる。
        """
        k = bits.size
        ngates = arity.size
        mask = np.int64(0)
        for i in range(k):
            mask |= np.int64(1) << bits[i]
        notmask = ~mask
        R0 = np.int64(1) << bits[0]
        for c in prange(nchunks):
            t_lo = np.int64(c) * chunk
            span = min(chunk, ntiles - t_lo)
            R = R0
            if R > span:
                R = span
            use_mask_adv = R == R0
            nrun = span // R
            # チャンク先頭のタイル基底 (bits[] の位置に 0 を挿入)
            b_start = np.int64(t_lo)
            for i in range(k):
                p = bits[i]
                b_start = (((b_start >> p) << (p + 1))
                           | (b_start & ((np.int64(1) << p) - 1)))

            for g in range(ngates):
                kg = arity[g]
                nsub = np.int64(1) << (k - kg)
                if has_deriv[g] != 0:
                    re, im = _unwind_deriv_chunk(
                        psi, lam, dmats[g], dph[g], dsrc[g], rowoff[g],
                        dkind[g] != 0, kg, subbase[g], nsub, b_start, nrun,
                        R, mask, notmask, use_mask_adv)
                    out[c, g, 0] = re
                    out[c, g, 1] = im
                _unwind_apply_chunk(
                    psi, mats_p[g], mph_p[g], mono_src[g], rowoff[g],
                    kind[g] != 0, kg, subbase[g], nsub, b_start, nrun, R,
                    mask, notmask, use_mask_adv)
                if do_lam[g] != 0:
                    _unwind_apply_chunk(
                        lam, mats_l[g], mph_l[g], mono_src[g], rowoff[g],
                        kind[g] != 0, kg, subbase[g], nsub, b_start, nrun,
                        R, mask, notmask, use_mask_adv)

    @jit(nopython=True, parallel=True, cache=True)
    def _apply_1q(sv, g, t):
        """sv <- (g on qubit t) sv, in place. g index: standard 2x2."""
        step = np.int64(1) << t
        half = sv.size >> 1
        for ii in prange(half):
            i = np.int64(ii)
            b = ((i >> t) << (t + 1)) | (i & (step - 1))
            i1 = b + step
            a0 = sv[b]
            a1 = sv[i1]
            sv[b] = g[0, 0] * a0 + g[0, 1] * a1
            sv[i1] = g[1, 0] * a0 + g[1, 1] * a1

    @jit(nopython=True, parallel=True, cache=True)
    def _apply_2q(sv, g, p1, p0):
        """sv <- g sv, in place. g row index = 2*bit(p1) + bit(p0)
        (p1 = qubits[0] = MSB of the gate index)."""
        lo = min(p0, p1)
        hi = max(p0, p1)
        s0 = np.int64(1) << p0
        s1 = np.int64(1) << p1
        mlo = (np.int64(1) << lo) - 1
        mhi = (np.int64(1) << hi) - 1
        quarter = sv.size >> 2
        for ii in prange(quarter):
            i = np.int64(ii)
            b = ((i >> lo) << (lo + 1)) | (i & mlo)
            b = ((b >> hi) << (hi + 1)) | (b & mhi)
            i1 = b + s0
            i2 = b + s1
            i3 = b + s0 + s1
            a0 = sv[b]
            a1 = sv[i1]
            a2 = sv[i2]
            a3 = sv[i3]
            sv[b] = g[0, 0] * a0 + g[0, 1] * a1 + g[0, 2] * a2 + g[0, 3] * a3
            sv[i1] = g[1, 0] * a0 + g[1, 1] * a1 + g[1, 2] * a2 + g[1, 3] * a3
            sv[i2] = g[2, 0] * a0 + g[2, 1] * a1 + g[2, 2] * a2 + g[2, 3] * a3
            sv[i3] = g[3, 0] * a0 + g[3, 1] * a1 + g[3, 2] * a2 + g[3, 3] * a3

    @jit(nopython=True, parallel=True, cache=True)
    def _apply_3q(sv, g, p2, p1, p0):
        """sv <- g sv, in place. g row index = 4*bit(p2)+2*bit(p1)+bit(p0)
        (p2 = qubits[0] = MSB)."""
        lo = min(p0, min(p1, p2))
        hi = max(p0, max(p1, p2))
        mid = (p0 + p1 + p2) - lo - hi
        s0 = np.int64(1) << p0
        s1 = np.int64(1) << p1
        s2 = np.int64(1) << p2
        mlo = (np.int64(1) << lo) - 1
        mmid = (np.int64(1) << mid) - 1
        mhi = (np.int64(1) << hi) - 1
        eighth = sv.size >> 3
        for ii in prange(eighth):
            i = np.int64(ii)
            b = ((i >> lo) << (lo + 1)) | (i & mlo)
            b = ((b >> mid) << (mid + 1)) | (b & mmid)
            b = ((b >> hi) << (hi + 1)) | (b & mhi)
            a0 = sv[b]
            a1 = sv[b + s0]
            a2 = sv[b + s1]
            a3 = sv[b + s1 + s0]
            a4 = sv[b + s2]
            a5 = sv[b + s2 + s0]
            a6 = sv[b + s2 + s1]
            a7 = sv[b + s2 + s1 + s0]
            for r in range(8):
                acc = (g[r, 0] * a0 + g[r, 1] * a1 + g[r, 2] * a2
                       + g[r, 3] * a3 + g[r, 4] * a4 + g[r, 5] * a5
                       + g[r, 6] * a6 + g[r, 7] * a7)
                idx = b
                if r & 1:
                    idx += s0
                if r & 2:
                    idx += s1
                if r & 4:
                    idx += s2
                sv[idx] = acc

    @jit(nopython=True, parallel=True, cache=True)
    def _brk_1q(lam, psi, g, t):
        """(re, im) of <lam| (g on qubit t) |psi>, float64 accumulation."""
        step = np.int64(1) << t
        half = psi.size >> 1
        re = 0.0
        im = 0.0
        for ii in prange(half):
            i = np.int64(ii)
            b = ((i >> t) << (t + 1)) | (i & (step - 1))
            i1 = b + step
            a0 = psi[b]
            a1 = psi[i1]
            m0 = g[0, 0] * a0 + g[0, 1] * a1
            m1 = g[1, 0] * a0 + g[1, 1] * a1
            l0 = lam[b]
            l1 = lam[i1]
            re += (np.float64(l0.real) * np.float64(m0.real)
                   + np.float64(l0.imag) * np.float64(m0.imag)
                   + np.float64(l1.real) * np.float64(m1.real)
                   + np.float64(l1.imag) * np.float64(m1.imag))
            im += (np.float64(l0.real) * np.float64(m0.imag)
                   - np.float64(l0.imag) * np.float64(m0.real)
                   + np.float64(l1.real) * np.float64(m1.imag)
                   - np.float64(l1.imag) * np.float64(m1.real))
        return re, im

    @jit(nopython=True, parallel=True, cache=True)
    def _brk_2q(lam, psi, g, p1, p0):
        """(re, im) of <lam| g |psi> for a 2-qubit operator."""
        lo = min(p0, p1)
        hi = max(p0, p1)
        s0 = np.int64(1) << p0
        s1 = np.int64(1) << p1
        mlo = (np.int64(1) << lo) - 1
        mhi = (np.int64(1) << hi) - 1
        quarter = psi.size >> 2
        re = 0.0
        im = 0.0
        for ii in prange(quarter):
            i = np.int64(ii)
            b = ((i >> lo) << (lo + 1)) | (i & mlo)
            b = ((b >> hi) << (hi + 1)) | (b & mhi)
            i1 = b + s0
            i2 = b + s1
            i3 = b + s0 + s1
            a0 = psi[b]
            a1 = psi[i1]
            a2 = psi[i2]
            a3 = psi[i3]
            for r in range(4):
                m = g[r, 0] * a0 + g[r, 1] * a1 + g[r, 2] * a2 + g[r, 3] * a3
                if r == 0:
                    l = lam[b]
                elif r == 1:
                    l = lam[i1]
                elif r == 2:
                    l = lam[i2]
                else:
                    l = lam[i3]
                re += (np.float64(l.real) * np.float64(m.real)
                       + np.float64(l.imag) * np.float64(m.imag))
                im += (np.float64(l.real) * np.float64(m.imag)
                       - np.float64(l.imag) * np.float64(m.real))
        return re, im

    @jit(nopython=True, parallel=True, cache=True)
    def _accumulate_pauli(psi, out, x_mask, sign_mask, w):
        """out += w * Q psi where (Q psi)[i] = sgn(i^x) psi[i^x],
        sgn(j) = (-1)^parity(j & sign_mask)."""
        for ii in prange(psi.size):
            i = np.int64(ii)
            j = i ^ x_mask
            v = np.int64(j & sign_mask)
            v ^= v >> 32
            v ^= v >> 16
            v ^= v >> 8
            v ^= v >> 4
            v ^= v >> 2
            v ^= v >> 1
            sgn = 1.0 - 2.0 * np.float64(v & 1)
            out[i] = out[i] + w * sgn * psi[j]


def _apply_gate_matrix(sv: np.ndarray, mat: np.ndarray, bits: List[int]):
    """Apply a bound gate matrix to sv in place (1-3 qubits).

    ``bits`` are index bit positions (= qubit indices on a canonical
    layout, pos_of[q] on a permuted one).
    """
    g = np.ascontiguousarray(mat).astype(sv.dtype, copy=False)
    if len(bits) == 1:
        _apply_1q(sv, g, bits[0])
    elif len(bits) == 2:
        _apply_2q(sv, g, bits[0], bits[1])
    elif len(bits) == 3:
        _apply_3q(sv, g, bits[0], bits[1], bits[2])
    else:
        raise AdjointUnsupported(f"{len(bits)}-qubit gate")


def _bra_op_ket(lam: np.ndarray, psi: np.ndarray, mat: np.ndarray,
                bits: List[int]) -> complex:
    g = np.ascontiguousarray(mat)
    if len(bits) == 1:
        re, im = _brk_1q(lam, psi, g, bits[0])
    elif len(bits) == 2:
        re, im = _brk_2q(lam, psi, g, bits[0], bits[1])
    else:
        raise AdjointUnsupported(f"{len(bits)}-qubit derivative gate")
    return complex(re, im)


def _apply_hamiltonian(psi: np.ndarray, hamiltonian,
                       num_qubits: int) -> np.ndarray:
    """λ = H_eff |psi> in complex128, one read+write pass per term.

    項ごとの積み上げ実装 (``apply_hamiltonian_multi`` の 1 パス版に
    置き換えられた参照実装; テストで比較する)。

    E = Σ_t Re(c_t) Re<P_t> の勾配に効くのは各項のエルミート部分。
    P_t = w Q (w = i^{n_Y} × 合成位相, Q はマスク作用素) に対し
    Q† = (-1)^{n_Y} Q なのでエルミート部分は h Q,
    h = (w + conj(w) (-1)^{n_Y}) / 2。
    """
    lam = np.zeros(psi.size, dtype=np.complex128)
    for term in hamiltonian.terms:
        x_mask, sign_mask, n_y, phase = _term_masks(term)
        w = (1j ** (n_y & 3)) * phase
        h = (w + np.conj(w) * (-1.0) ** (n_y & 1)) / 2.0
        scalar = complex(term.coeff.real * h)
        if scalar == 0:
            continue
        _accumulate_pauli(psi, lam, np.int64(x_mask), np.int64(sign_mask),
                          scalar)
    return lam


# ============================================================================
# Reverse-sweep block planning
# ============================================================================

class _Block:
    """One fused unwind step: gates sharing <= K index bits."""

    __slots__ = ('bits', 'gate_idx')

    def __init__(self, bits, gate_idx):
        self.bits = bits            # sorted ascending index bit positions
        self.gate_idx = gate_idx    # gate indices, in unwind order


def _plan_reverse_blocks(gates, max_block: int) -> List[_Block]:
    """Reverse the gate list, repack commuting gates, cut into blocks.

    ``_reorder_for_locality`` は per-qubit 依存 DAG のトポロジカル順序
    を返す (同じ qubit に触るゲートの相対順序は保存)。逆順ゲート列に
    かけると「巻き戻し順の入れ替え」になるが、入れ替わるのは disjoint
    qubit のゲート同士だけなので各ゲートの ψ_j / λ_j (したがって
    微分値) は変わらない。
    """
    from .fusion import _reorder_for_locality

    rev = list(range(len(gates)))[::-1]
    items = [(None, list(gates[i].qubits), False) for i in rev]
    seq = [rev[o] for o in _reorder_for_locality(items, max_block)]

    blocks: List[_Block] = []
    cur: List[int] = []
    union: set = set()
    for gi in seq:
        qs = set(gates[gi].qubits)
        if cur and len(union | qs) > max_block:
            blocks.append(_Block(sorted(union), cur))
            cur, union = [], set()
        cur.append(gi)
        union |= qs
    if cur:
        blocks.append(_Block(sorted(union), cur))
    return blocks


def _monomial(mat: np.ndarray):
    """[(col, phase), ...] if every row of ``mat`` has a single nonzero,
    else None.

    対角ゲート (rz/cz/rzz/cp/ccz...) と置換ゲート (x/cx/swap/ccx...)、
    そして _DERIV_SUPPORTED の全 Dm = dU U† (生成子が Pauli 積なので
    必ず単項式) がこれに当たる。(M v)[r] = phase_r * v[col_r] と書ける
    ので、カーネルは振幅あたり 1 回の複素乗算で済ませられる。

    判定は厳密ゼロ比較 (誤検出しない側に倒す): 数値的に極小なだけの
    非ゼロ要素は「単項式ではない」と判断され、dense 経路に落ちる
    だけで結果は変わらない。2x2/4x4/8x8 の小行列なので NumPy の
    呼び出しコストを避けて素の Python で走査する (1 評価あたり
    ゲート数 x 2 回呼ばれる)。
    """
    out = []
    for row in mat.tolist():
        col = -1
        for j, v in enumerate(row):
            if v != 0:
                if col >= 0:
                    return None
                col = j
        out.append((0, 0j) if col < 0 else (col, row[col]))
    return out


class _BlockStructure:
    """Parameter-independent layout of one fused unwind block.

    ゲート名・qubit 列と pos_of だけで決まる (行列値を見ない) ので、
    同じ回路構造なら再利用できる: VQE/QAOA は同じ回路をパラメータ値
    だけ変えて何度も微分するため、小さい n ではこの組み立てが
    カーネル本体より高くつく。
    """

    __slots__ = ('gate_idx', 'bits', 'subbase', 'rowoff', 'arity', 'dgs')

    def __init__(self, gate_idx, bits, subbase, rowoff, arity, dgs):
        self.gate_idx = gate_idx
        self.bits = bits
        self.subbase = subbase
        self.rowoff = rowoff
        self.arity = arity
        self.dgs = dgs


def _block_structure(block: _Block, gates, pos_of) -> _BlockStructure:
    """Build the index tables for one block (no gate matrices).

    ``subbase[g, si]`` is the statevector offset of the si-th sub-tile
    base of gate g (the block bits it does not act on) and
    ``rowoff[g, j]`` the statevector offset of the gate matrix's row j,
    so amplitude j of gate g in sub-tile si of tile base b lives at
    ``b + subbase[g, si] + rowoff[g, j]``.
    """
    bits = sorted(pos_of[q] for q in block.bits)
    k = len(bits)
    dim = 1 << k
    loc_of_bit = {b: i for i, b in enumerate(bits)}

    ng = len(block.gate_idx)
    arity = np.zeros(ng, dtype=np.int64)
    rowoff = np.zeros((ng, 8), dtype=np.int64)
    subbase = np.zeros((ng, dim), dtype=np.int64)
    dgs = []

    for g, gi in enumerate(block.gate_idx):
        gate = gates[gi]
        kg = len(gate.qubits)
        if kg > 3:
            raise AdjointUnsupported(f"{kg}-qubit gate")
        dg = 1 << kg
        arity[g] = kg
        dgs.append(dg)

        # gpos[t]: 行列インデックスの bit t (LSB=0) が載るタイル内位置。
        # get_gate_matrix は qubits[0] を MSB とする規約。
        gpos = [loc_of_bit[pos_of[gate.qubits[kg - 1 - t]]]
                for t in range(kg)]
        gmask = 0
        for q in gpos:
            gmask |= 1 << q
        for j in range(dg):
            o = 0
            for t in range(kg):
                if (j >> t) & 1:
                    o |= 1 << bits[gpos[t]]
            rowoff[g, j] = o
        nb = 0
        for bl in range(dim):
            if bl & gmask:
                continue
            o = 0
            for i in range(k):
                if (bl >> i) & 1:
                    o |= 1 << bits[i]
            subbase[g, nb] = o
            nb += 1

    return _BlockStructure(block.gate_idx, np.asarray(bits, dtype=np.int64),
                           subbase, rowoff, arity, dgs)


def _pack_block(struct: _BlockStructure, gates, dcoeffs, psi_dtype,
                skip_last_lam: bool):
    """Fill the per-evaluation gate matrices for one block.

    ``kind[g]`` / ``dkind[g]`` are 1 when U† / Dm is monomial (one
    nonzero per row) and the kernel can take the 1-multiply path;
    ``mono_src[g, r]`` / ``dsrc[g, r]`` are then the statevector offsets
    the row reads from.

    Returns the kernel arguments plus ``deriv_gates``: the (kernel slot,
    gate index) pairs whose derivative the kernel fills.
    """
    ng = len(struct.gate_idx)
    rowoff = struct.rowoff
    mats_p = np.zeros((ng, 8, 8), dtype=psi_dtype)
    mats_l = np.zeros((ng, 8, 8), dtype=np.complex128)
    dmats = np.zeros((ng, 8, 8), dtype=np.complex128)
    mono_src = np.zeros((ng, 8), dtype=np.int64)
    mph_p = np.zeros((ng, 8), dtype=psi_dtype)
    mph_l = np.zeros((ng, 8), dtype=np.complex128)
    dsrc = np.zeros((ng, 8), dtype=np.int64)
    dph = np.zeros((ng, 8), dtype=np.complex128)
    kind = np.zeros(ng, dtype=np.uint8)
    dkind = np.zeros(ng, dtype=np.uint8)
    has_deriv = np.zeros(ng, dtype=np.uint8)
    do_lam = np.ones(ng, dtype=np.uint8)
    deriv_gates = []

    for g, gi in enumerate(struct.gate_idx):
        gate = gates[gi]
        dg = struct.dgs[g]
        if dg == 8 and dcoeffs[gi]:
            # 3-qubit rotations are not in _DERIV_SUPPORTED; the kernel's
            # generic branch has no derivative path.
            raise AdjointUnsupported(gate.name)
        mat = get_gate_matrix(gate.name, gate.params)
        if mat.shape[0] != dg:
            raise AdjointUnsupported(gate.name)
        udag = mat.conj().T
        mono = _monomial(udag)
        if mono is None:
            mats_p[g, :dg, :dg] = udag
            mats_l[g, :dg, :dg] = udag
        else:
            # 単項式経路では密行列は読まれないので詰めない。
            kind[g] = 1
            cols = [c for c, _ in mono]
            phases = [ph for _, ph in mono]
            mph_p[g, :dg] = phases
            mph_l[g, :dg] = phases
            mono_src[g, :dg] = rowoff[g, cols]

        if dcoeffs[gi]:
            dm = _unwind_deriv_matrix(gate.name, gate.params)
            dmono = _monomial(dm)
            if dmono is None:
                dmats[g, :dg, :dg] = dm
            else:
                dkind[g] = 1
                dph[g, :dg] = [ph for _, ph in dmono]
                dsrc[g, :dg] = rowoff[g, [c for c, _ in dmono]]
            has_deriv[g] = 1
            deriv_gates.append((g, gi))

    if skip_last_lam:
        # 最後に巻き戻すゲートの後で λ は二度と読まれない。
        do_lam[ng - 1] = 0

    return (struct.bits, struct.subbase, rowoff, mono_src, mats_p, mats_l,
            dmats, mph_p, mph_l, dsrc, dph, struct.arity, kind, dkind,
            has_deriv, do_lam, deriv_gates)


# 構造キャッシュ (回路構造 + レイアウト -> ブロック構造)。FusionPlan の
# キャッシュと同じ動機: パラメータだけ変えた再評価で作り直さない。
_STRUCT_CACHE = {}
_STRUCT_CACHE_MAX = 32


def _reverse_blocks_cached(gates, pos_of, max_block):
    key = (max_block, tuple(pos_of),
           tuple((g.name, tuple(g.qubits)) for g in gates))
    structs = _STRUCT_CACHE.get(key)
    if structs is None:
        structs = [_block_structure(b, gates, pos_of)
                   for b in _plan_reverse_blocks(gates, max_block)]
        if len(_STRUCT_CACHE) >= _STRUCT_CACHE_MAX:
            _STRUCT_CACHE.pop(next(iter(_STRUCT_CACHE)))
        _STRUCT_CACHE[key] = structs
    return structs


def _tile_chunk_plan(ntiles: int):
    """(nchunks, chunk) in tiles; chunk is a power of two.

    ブロック内の全ゲートがチャンクを触り直すので L2 に載る大きさで
    切る。2 のべきに丸めるのは、カーネルの連続ランループが
    「チャンクはラン長 2^bits[0] の倍数か、その中に収まる」ことを
    前提にするため。
    """
    target = max(1, _CHUNKS_PER_THREAD * max(1, get_num_threads()))
    chunk = _MIN_TILES_PER_CHUNK
    while chunk * target < ntiles:
        chunk <<= 1
    chunk = min(chunk, ntiles)
    return -(-ntiles // chunk), chunk


# ============================================================================
# Adjoint gradient driver
# ============================================================================

def _analyze_parameters(circuit: 'Circuit'):
    """Map each non-barrier gate to its d(gate param)/d(theta_i) entries.

    Returns a list aligned with the non-barrier gate sequence; each
    element is a list of (theta_index, weight). Raises
    AdjointUnsupported for gates the adjoint method cannot handle.
    """
    plist = circuit.parameters
    pindex = {p: k for k, p in enumerate(plist)}
    dcoeffs = []
    for gate in circuit._gates:
        if gate.name == 'barrier':
            continue
        entries = []
        has_free = any(is_parameterized(p) for p in gate.params)
        if has_free:
            if gate.name not in _DERIV_SUPPORTED or len(gate.params) != 1:
                raise AdjointUnsupported(gate.name)
            pv = gate.params[0]
            if isinstance(pv, Parameter):
                entries.append((pindex[pv], 1.0))
            elif isinstance(pv, ParameterExpression):
                for p in pv.parameters:
                    try:
                        w = pv.grad(p)
                    except Exception as exc:
                        raise AdjointUnsupported(str(exc))
                    if w != 0.0:
                        entries.append((pindex[p], float(w)))
        dcoeffs.append(entries)
    return dcoeffs, len(plist)


def adjoint_energy_and_gradient(backend: 'CPUBackend',
                                circuit: 'Circuit',
                                hamiltonian: 'Hamiltonian',
                                params: List[float],
                                max_block: int = None):
    """<H> and dE/dθ for all θ in ~2 circuit applications.

    Returns (energy, grads) with ``grads`` aligned with
    ``circuit.parameters``. Raises AdjointUnsupported if the circuit is
    outside the adjoint method's reach (caller falls back to
    parameter-shift).
    """
    if not HAS_NUMBA:
        raise AdjointUnsupported("numba required")

    from ...spin import PauliTerm, Hamiltonian as _H
    if isinstance(hamiltonian, PauliTerm):
        hamiltonian = _H([hamiltonian])

    # パラメータ解析は前もって (未対応回路で forward を無駄にしない)。
    dcoeffs, num_params = _analyze_parameters(circuit)
    grads = np.zeros(num_params)

    bound = circuit.bind_parameters(list(params))
    n = circuit.num_qubits
    gates = [g for g in bound._gates if g.name != 'barrier']

    # Forward (fused). レイアウトは復元せず pos_of のまま扱う:
    # ゲートは bit pos_of[q] に、λ のマスクも pos_of で組める。
    # statevector バッファは呼び出し元と共有されない契約なので
    # in-place で巻き戻してよい。
    psi, pos_of = backend._statevector_layout(bound)
    energy, lam = apply_hamiltonian_multi(psi, hamiltonian, n, pos_of)
    stats.n_kernels += 1
    stats.n_passes += 2.0

    if num_params == 0:
        return energy, grads

    max_block = BLOCK_QUBITS if max_block is None else max_block
    max_block = max(1, min(max_block, n))
    blocks = _reverse_blocks_cached(gates, pos_of, max_block)

    ntiles_cache = {}
    for bi, block in enumerate(blocks):
        packed = _pack_block(block, gates, dcoeffs, psi.dtype,
                             skip_last_lam=(bi == len(blocks) - 1))
        (bits, subbase, rowoff, mono_src, mats_p, mats_l, dmats, mph_p,
         mph_l, dsrc, dph, arity, kind, dkind, has_deriv, do_lam,
         deriv_gates) = packed

        ntiles = psi.size >> bits.size
        plan = ntiles_cache.get(ntiles)
        if plan is None:
            plan = _tile_chunk_plan(ntiles)
            ntiles_cache[ntiles] = plan
        nchunks, chunk = plan
        out = np.zeros((nchunks, len(block.gate_idx), 2))

        _unwind_block_numba(psi, lam, bits, subbase, rowoff, mono_src,
                            mats_p, mats_l, dmats, mph_p, mph_l, dsrc, dph,
                            arity, kind, dkind, has_deriv, do_lam,
                            np.int64(ntiles), nchunks, np.int64(chunk), out)
        stats.n_kernels += 1
        stats.n_blocks += 1
        stats.n_gates += len(block.gate_idx)
        stats.n_passes += 4.0

        if deriv_gates:
            part = out.sum(axis=0)
            for g, gi in deriv_gates:
                val = 2.0 * part[g, 0]
                for pi, w in dcoeffs[gi]:
                    grads[pi] += w * val

    return energy, grads


def adjoint_gradient(backend: 'CPUBackend',
                     circuit: 'Circuit',
                     hamiltonian: 'Hamiltonian',
                     params: List[float]) -> np.ndarray:
    """dE/dθ for all θ (see ``adjoint_energy_and_gradient``)."""
    return adjoint_energy_and_gradient(backend, circuit, hamiltonian,
                                       params)[1]


def adjoint_gradient_reference(backend: 'CPUBackend',
                               circuit: 'Circuit',
                               hamiltonian: 'Hamiltonian',
                               params: List[float]) -> np.ndarray:
    """Per-gate bra-op-ket reference implementation (slow).

    ブロック融合版の検証用に残してある元アルゴリズム: ゲートごとに
    ψ ← U†ψ、<λ|dU|ψ_{j-1}> の 1 パス縮約、λ ← U†λ。
    """
    if not HAS_NUMBA:
        raise AdjointUnsupported("numba required")

    from ...spin import PauliTerm, Hamiltonian as _H
    if isinstance(hamiltonian, PauliTerm):
        hamiltonian = _H([hamiltonian])

    dcoeffs, num_params = _analyze_parameters(circuit)
    grads = np.zeros(num_params)
    if num_params == 0:
        return grads

    bound = circuit.bind_parameters(list(params))
    n = circuit.num_qubits
    gates = [g for g in bound._gates if g.name != 'barrier']

    psi = backend.statevector(bound)
    lam = _apply_hamiltonian(psi, hamiltonian, n)

    for j in range(len(gates) - 1, -1, -1):
        gate = gates[j]
        mat = get_gate_matrix(gate.name, gate.params)
        udag = np.conj(mat).T
        _apply_gate_matrix(psi, udag, gate.qubits)   # ψ -> ψ_{j-1}
        if dcoeffs[j]:
            dU = _deriv_matrix(gate.name, gate.params)
            val = 2.0 * _bra_op_ket(lam, psi, dU, gate.qubits).real
            for pi, w in dcoeffs[j]:
                grads[pi] += w * val
        if j > 0:
            _apply_gate_matrix(lam, udag, gate.qubits)

    return grads
