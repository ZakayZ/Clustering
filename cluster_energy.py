import functools
import itertools
import math
import numpy as np
import masstable
from scipy.special import erf

from dataclasses import dataclass


# -----------------------------------------------------------------------------
# Nucleon cloud (replaces notebook ``Event`` / ``Particle``)
# -----------------------------------------------------------------------------


def _invariant_mass_rows(p4: np.ndarray) -> np.ndarray:
    """Per-row invariant mass; ``p4`` shape ``(K, 4)`` with columns E, px, py, pz."""
    E = p4[:, 0]
    px = p4[:, 1]
    py = p4[:, 2]
    pz = p4[:, 3]
    s = E * E - px * px - py * py - pz * pz
    s = np.maximum(s, 0.0)
    return np.sqrt(s)


@dataclass
class NucleonCloud:
    """
    One event worth of nucleons: Cartesian positions, four-momenta, and species.

    ``pos`` shape ``(N, 3)`` in fm. ``four_momentum`` shape ``(N, 4)`` with rows
    ``(E, p_x, p_y, p_z)`` (energy and three-momentum in **MeV** and **MeV/c**).
    ``is_proton`` shape ``(N,)`` (neutron if False).
    """

    pos: np.ndarray
    four_momentum: np.ndarray
    is_proton: np.ndarray

    def __post_init__(self) -> None:
        self.pos = np.asarray(self.pos, dtype=np.float64)
        self.four_momentum = np.asarray(self.four_momentum, dtype=np.float64)
        self.is_proton = np.asarray(self.is_proton, dtype=bool)
        n = int(self.pos.shape[0])
        if self.pos.ndim != 2 or self.pos.shape[1] != 3:
            raise ValueError("pos must have shape (N, 3)")
        if self.four_momentum.shape != (n, 4):
            raise ValueError("four_momentum must have shape (N, 4)")
        if self.is_proton.shape != (n,):
            raise ValueError("is_proton must have shape (N,)")

    @classmethod
    def from_numpy(
        cls,
        pos: np.ndarray,
        mom: np.ndarray,
        is_proton: np.ndarray,
    ) -> "NucleonCloud":
        pos = np.asarray(pos, dtype=np.float64)
        mom = np.asarray(mom, dtype=np.float64)
        is_proton = np.asarray(is_proton, dtype=bool)

        pos3 = pos[:, 1:4].copy() if pos.shape[1] == 4 else pos.copy()
        return cls(pos=pos3, four_momentum=mom, is_proton=is_proton)


@dataclass(frozen=True)
class ClusterEnergyResult:
    """UrQMD-style cluster decomposition (energies in MeV)."""

    A: int
    Z: int
    internal_kinetic: float
    pair_potential: float
    binding_prior: float
    total_energy: float


def _pair_relative_momentum_matrix(mom3: np.ndarray) -> np.ndarray:
    """``|p_i - p_j|`` for every pair; shape ``(n, n)``, symmetric."""
    dp = mom3[:, None, :] - mom3[None, :, :]
    return np.linalg.norm(dp, axis=2)


def _spatial_distance_matrix(pos_c: np.ndarray) -> np.ndarray:
    """``|r_i - r_j|`` for every pair; shape ``(n, n)``, symmetric."""
    dr = pos_c[:, None, :] - pos_c[None, :, :]
    return np.linalg.norm(dr, axis=2)


# -----------------------------------------------------------------------------
# Nuclear binding for the macroscopic prior: tables first, SEMF fallback (MeV)
# -----------------------------------------------------------------------------

M_P = 938.2723
M_N = 939.5656

_MASS_TABLE_NAMES: tuple[str, ...] = ("AME2012all",)


@functools.lru_cache(maxsize=None)
def _table(name: str):
    return masstable.Table(name)


def binding_energy_liquid_drop(a: int, z: int) -> float:
    """SEMF binding energy in MeV (used only when nuclide is absent from mass tables)."""
    if a <= 0 or z < 0 or z > a:
        raise ValueError(f"invalid (A, Z)=({a}, {z})")
    n = a - z
    if a == 1:
        return 0.0
    a_v, a_s, a_c, a_a = 15.75, 17.8, 0.711, 23.7
    vol = a_v * a
    surf = a_s * (a ** (2.0 / 3.0))
    coul = a_c * z * (z - 1.0) / (a ** (1.0 / 3.0))
    asym = a_a * ((n - z) ** 2) / a
    if a % 2 == 1:
        delta = 0.0
    elif z % 2 == 0:
        delta = 12.0 / math.sqrt(a)
    else:
        delta = -12.0 / math.sqrt(a)
    return vol - surf - coul - asym + delta


def binding_energy_from_tables(a: int, z: int) -> float | None:
    """Binding energy in MeV from ``masstable`` if ``(Z, N)`` is in the table."""
    n = a - z
    key = (z, n)
    for name in _MASS_TABLE_NAMES:
        t = _table(name)
        if key in t.binding_energy.df.index:
            return float(t.binding_energy.df.loc[key])
    return None


def get_mass(a: int, z: int) -> float:
    """
    Ground-state mass: ``Z M_p + N M_n - B`` with ``B`` from tables if present,
    else :func:`binding_energy_liquid_drop` (MeV).
    """
    n = a - z
    b = binding_energy_from_tables(a, z) or binding_energy_liquid_drop(a, z)
    return z * M_P + n * M_N - b


def nuclear_binding_energy(a: int, z: int) -> float:
    return z * M_P + (a - z) * M_N - get_mass(a, z)


def binding_prior(a: int, z: int) -> float:
    """Macroscopic binding prior (MeV), subtracted in :func:`cluster_energy`."""
    if a < 1 or z < 0 or z > a:
        raise ValueError(f"invalid (A, Z)=({a}, {z})")

    if a == 1:
        return 0.0

    return nuclear_binding_energy(a, z)


# -----------------------------------------------------------------------------
# UrQMD hard EoS parameters (Table 3.1, no Pauli by default), energies in MeV
# -----------------------------------------------------------------------------

URQMD_ALPHA_FM_INV2 = 0.25
URQMD_T1_FM3 = -7264.04
URQMD_TGAMMA_FM6 = 87.65
URQMD_V0YUK_FM = -0.498
URQMD_GAMMA_Y_FM = 1.4
URQMD_E2_FM = 1.44

URQMD_P_REL_MAX = 2000.0

URQMD_USE_PAULI_APPROX = False
URQMD_V0PAU = 98.95
URQMD_Q0_FM = 2.16
URQMD_P0 = 120.0

URQMD_USE_SK3 = False


def cluster_internal_kinetic(cloud: NucleonCloud, cluster: list[int]) -> float:
    idx = np.asarray(cluster, dtype=np.int64)
    p4 = cloud.four_momentum[idx]
    masses = _invariant_mass_rows(p4)
    momenta = p4[:, 1:4]
    total_mass = float(np.sum(masses))
    p_cm = np.sum(momenta, axis=0) / total_mass
    q = momenta - masses[:, None] * p_cm[None, :]
    q2 = np.sum(q * q, axis=1)
    e_rel = np.sqrt(masses * masses + q2) - masses
    return float(np.sum(e_rel))


def _urqmd_sk2_pair_energy_vec(r: np.ndarray) -> np.ndarray:
    alpha = URQMD_ALPHA_FM_INV2
    return URQMD_T1_FM3 * (alpha / math.pi) ** 1.5 * np.exp(-alpha * r * r)


def _urqmd_yukawa_pair_energy_vec(r: np.ndarray) -> np.ndarray:
    out = np.zeros_like(r, dtype=np.float64)
    mask = r > 1.0e-12
    if not np.any(mask):
        return out
    r_m = r[mask]
    alpha = URQMD_ALPHA_FM_INV2
    gamma_y = URQMD_GAMMA_Y_FM
    v0 = URQMD_V0YUK_FM
    pref = v0 * (1.0 / (2.0 * r_m)) * np.exp(1.0 / (4.0 * alpha * gamma_y * gamma_y))
    a = 1.0 / (2.0 * gamma_y * math.sqrt(alpha))
    b = math.sqrt(alpha) * r_m
    term1 = np.exp(-r_m / gamma_y) * (1.0 - erf(a - b))
    term2 = np.exp(+r_m / gamma_y) * (1.0 - erf(a + b))
    out[mask] = pref * (term1 - term2)
    return out


def _urqmd_coulomb_pair_energy_vec(z_prod: np.ndarray, r: np.ndarray) -> np.ndarray:
    out = np.zeros_like(r, dtype=np.float64)
    mask = (r > 1.0e-12) & (z_prod > 0.0)
    if not np.any(mask):
        return out
    r_m = r[mask]
    z_m = z_prod[mask]
    alpha = URQMD_ALPHA_FM_INV2
    out[mask] = (z_m * URQMD_E2_FM / r_m) * erf(np.sqrt(alpha) * r_m)
    return out


def _urqmd_pauli_pair_energy_vec(
    r: np.ndarray, p_rel: np.ndarray, same_isospin: np.ndarray
) -> np.ndarray:
    if not URQMD_USE_PAULI_APPROX:
        return np.zeros_like(r)
    alpha = URQMD_ALPHA_FM_INV2
    q0 = URQMD_Q0_FM
    p0 = URQMD_P0
    pref = (
        URQMD_V0PAU
        * (1.0 / (p0 * q0)) ** 3
        * (1.0 + 1.0 / (2.0 * alpha * q0 * q0)) ** (-1.5)
    )
    expo = np.exp(
        -alpha * r * r / (2.0 * alpha * q0 * q0 + 1.0) - (p_rel * p_rel) / (2.0 * p0 * p0)
    )
    return pref * expo * same_isospin


def urqmd_pair_energies_upper_triangle(
    pos_c: np.ndarray, mom3: np.ndarray, is_proton_c: np.ndarray
) -> np.ndarray:
    """
    Pair energies for all unique pairs in a cluster (upper triangle), shape ``(P,)``
    with ``P = n*(n-1)//2``, same order as ``np.triu_indices(n, k=1)``.
    """
    n = pos_c.shape[0]
    dr = pos_c[:, None, :] - pos_c[None, :, :]
    r = np.linalg.norm(dr, axis=2)
    dp = mom3[:, None, :] - mom3[None, :, :]
    p_rel = np.linalg.norm(dp, axis=2)
    iu, ju = np.triu_indices(n, k=1)
    r_p = r[iu, ju]
    p_rel_p = p_rel[iu, ju]
    zl = is_proton_c.astype(np.float64)
    z_prod = zl[iu] * zl[ju]
    same_iso = (is_proton_c[iu] == is_proton_c[ju]).astype(np.float64)

    e_sk2 = _urqmd_sk2_pair_energy_vec(r_p)
    e_yuk = _urqmd_yukawa_pair_energy_vec(r_p)
    e_coul = _urqmd_coulomb_pair_energy_vec(z_prod, r_p)
    e_pau = _urqmd_pauli_pair_energy_vec(r_p, p_rel_p, same_iso)
    e = e_sk2 + e_yuk + e_coul + e_pau
    return np.where(p_rel_p < URQMD_P_REL_MAX, e, 0.0)


def _urqmd_sk3_triplet_from_pair_values(
    pij: float,
    pik: float,
    pjk: float,
    rij: float,
    rik: float,
    rjk: float,
) -> float:
    if max(pij, pik, pjk) >= URQMD_P_REL_MAX:
        return 0.0
    alpha = URQMD_ALPHA_FM_INV2
    pref = URQMD_TGAMMA_FM6 * (4.0 * alpha**2 / (3.0 * math.pi**2)) ** 1.5
    term_i = math.exp(-alpha * (rij * rij + rik * rik))
    term_j = math.exp(-alpha * (rij * rij + rjk * rjk))
    term_k = math.exp(-alpha * (rik * rik + rjk * rjk))
    return pref * (term_i + term_j + term_k) / 3.0


def cluster_AZ(cloud: NucleonCloud, cluster: list[int]) -> tuple[int, int]:
    a = len(cluster)
    z = sum(1 for i in cluster if cloud.is_proton[i])
    return a, z


def cluster_pair_energy(cloud: NucleonCloud, cluster: list[int]) -> float:
    idx = np.asarray(cluster, dtype=int)
    pos_c = cloud.pos[idx]
    mom3 = cloud.four_momentum[idx, 1:4].copy()
    isp = cloud.is_proton[idx]
    return float(np.sum(urqmd_pair_energies_upper_triangle(pos_c, mom3, isp)))


def cluster_triplet_energy(cloud: NucleonCloud, cluster: list[int]) -> float:
    if not URQMD_USE_SK3:
        return 0.0
    idx = np.asarray(cluster, dtype=int)
    n = int(idx.size)
    if n < 3:
        return 0.0
    pos_c = cloud.pos[idx]
    mom3 = cloud.four_momentum[idx, 1:4]
    p_rel = _pair_relative_momentum_matrix(mom3)
    r = _spatial_distance_matrix(pos_c)
    total = 0.0
    for la, lb, lc in itertools.combinations(range(n), 3):
        total += _urqmd_sk3_triplet_from_pair_values(
            p_rel[la, lb],
            p_rel[la, lc],
            p_rel[lb, lc],
            r[la, lb],
            r[la, lc],
            r[lb, lc],
        )
    return total


def cluster_energy(cloud: NucleonCloud, cluster: list[int]) -> ClusterEnergyResult:
    """UrQMD cluster energy with macroscopic binding :func:`binding_prior` (MeV)."""
    a, z = cluster_AZ(cloud, cluster)
    tint = cluster_internal_kinetic(cloud, cluster)
    vpair = cluster_pair_energy(cloud, cluster)
    vtrip = cluster_triplet_energy(cloud, cluster)
    urqmd_total = tint + vpair + vtrip
    b_prior = binding_prior(a, z)
    total = urqmd_total - b_prior
    return ClusterEnergyResult(
        A=a,
        Z=z,
        internal_kinetic=tint,
        pair_potential=vpair + vtrip,
        binding_prior=b_prior,
        total_energy=total,
    )


def partition_loss(cloud: NucleonCloud, partition: list[list[int]]) -> float:
    """Sum of :func:`cluster_energy` ``total_energy`` over clusters (MeV)."""
    total = 0.0
    for c in partition:
        total += cluster_energy(cloud, c).total_energy
    return total


def partition_loss_numpy(
    pos: np.ndarray,
    mom: np.ndarray,
    is_proton: np.ndarray,
    partition: list[list[int]],
) -> float:
    """:func:`partition_loss` on numpy arrays (MeV).

    ``pos`` is ``(N, 3)`` fm ``(x,y,z)`` or ``(N, 4)`` ``(t, x, y, z)`` (fm/c, fm). Four-momenta
    rows are ``(E, px, py, pz)`` in MeV and MeV/c.
    """
    return partition_loss(NucleonCloud.from_numpy(pos, mom, is_proton), partition)
