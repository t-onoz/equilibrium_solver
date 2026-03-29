from __future__ import annotations
import typing as t
import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import optimize

from eq_solver.system import System, Constraint, Phase, Species

logger = logging.getLogger(__name__)

def fischer_burmeister(a, b, eps=0.):
    """complementary function
    f(a, b) = 0 ⇔ (a>=0 & b=0) or (a=0 & b>0)"""
    return np.sqrt(a*a + b*b + eps**2) - (a + b)

@dataclass(frozen=True)
class Conditions:
    values: npt.NDArray[np.float64]
    system: System
    q_ref: float = field(init=False)   # charge when only DIRECT species exist

    def __post_init__(self):
        n_val = len(self.values)
        n_cpt = len(self.system.components)
        if n_cpt != n_val:
            raise ValueError(f"length of values ({n_val}) must be equal to number of components ({n_cpt})")
        # calculation of q_ref
        cpt_charge = self.system.spc_charge[self.system.cpt_base_idx]
        _m = self.system.cpt_cstr == Constraint.DIRECT
        q_vec = cpt_charge[_m] * self.values[_m]
        q_pos = q_vec[q_vec > 0].sum()
        q_neg = -q_vec[q_vec < 0].sum()
        q_ref = np.min([q_pos, q_neg])
        if np.isclose(q_pos, q_neg):
            q_ref *= 1 - 1e-8
        object.__setattr__(self, 'q_ref', q_ref)

    @classmethod
    def from_dict(cls, system: System, conditions: dict[str, float]) -> Conditions:
        values = np.zeros_like(system.cpt_cstr, dtype=float)
        for i, cpt in enumerate(system.components):
            if cpt.constraint == Constraint.CHARGE:
                continue
            try:
                values[i] = conditions[cpt.name]
            except KeyError:
                raise ValueError(f"missing required component: {cpt.name}")
        return cls(values, system)


@dataclass
class FitFunc:
    system: System
    cond: Conditions

    @t.overload
    def __call__(self, x: t.Any, return_all_spc: t.Literal[False] = False) \
        -> np.ndarray: ...

    @t.overload
    def __call__(self, x: t.Any, return_all_spc: t.Literal[True] = True) \
        -> tuple[np.ndarray, pd.DataFrame]: ...

    def __call__(self, x: Sequence[float] | npt.NDArray[np.float64], return_all_spc=False) \
        -> np.ndarray | tuple[np.ndarray, pd.DataFrame]:
        """function to optimize.

        Variable (x) layout depends on activity_model:
            if activity_model == 'none':
                x = [log a_ind, n_solid]
            else:
                x = [log a_ind, n_solid, ionic_strength]

            Use `pprint(system.specs)` to determine the expected size.

        return_all_spc: if False, returns a residual vector.
                        if True, returns a tuple of (residual vector, DataFrame),
                        where DataFrame consists of "concentration" and "activity" columns.
        Notes:
            - activities are in log10 scale
            - aqueous species concentrations are derived via a = γ c
        """
        system = self.system
        cond = self.cond
        if len(x) != system.specs.n_vars:
            raise ValueError(f'len(x) must be {system.specs.n_vars}, but got x={repr(x)}.')

        # unpack x
        log_a_independent, n_solid, I = self._unpack_x(x)

        # determine γ (activity coefficient) based on the activity model
        if system.activity_model == 'none':
            spc_gamma = np.ones_like(system.spc_logK, dtype=np.float64)
        else:
            spc_gamma = system.gamma(I)
        cpt_gamma = spc_gamma[system.cpt_base_idx]

        # initialize log activities of the components
        cpt_log_a = np.zeros_like(system.cpt_cstr, dtype=float)
        # activity = γ * c for DIRECT components
        _m = system.cpt_cstr == Constraint.DIRECT
        with np.errstate(divide='ignore'):
            cpt_log_a[_m] = np.maximum(np.log10(cpt_gamma[_m] * cond.values[_m]), -9999)
        # for TOTAL or CHARGE components, log(activity) are taken from x.
        cpt_log_a[system.cpt_cstr != Constraint.DIRECT] = log_a_independent

        spc_log_a = system.stoichiometry_matrix @ cpt_log_a + system.spc_logK

        # initialize concentration of species
        spc_c = np.zeros_like(spc_log_a)
        # for liquid phase, c = a / γ
        _m = system.spc_phase == Phase.LIQUID
        spc_c[_m] = 10. ** spc_log_a[_m] / spc_gamma[_m]
        # for solid phases, c values are taken from x.
        spc_c[system.spc_phase == Phase.SOLID] = n_solid
        # c vales of gaseous molecules should be kept 0,
        # because we can't determine gas volume.

        # r0: total mole conservation
        _m = system.cpt_cstr == Constraint.TOTAL
        _c = spc_c @ system.stoichiometry_matrix[:, _m]
        _c0 = cond.values[_m]
        with np.errstate(divide='ignore'):
            r0 = np.where(_c0 == 0, _c, np.log10(_c0) - np.log10(_c))

        # r1: solid-liquid equilibrium
        # described using "complementary conditions"
        _m = system.spc_phase == Phase.SOLID
        # calculate scaling factor for solid species
        _c = np.where(system.cpt_cstr == Constraint.TOTAL, cond.values, np.nan)
        with np.errstate(divide='ignore', invalid='ignore'), warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
            _v = np.nanmin(np.abs(_c / system.stoichiometry_matrix), axis=1)
        _v = np.where(np.isnan(_v) | (_v == 0), 1.0, _v)
        r1 = fischer_burmeister(
            spc_c[_m] / _v[_m],  # mole of solid phase
            -spc_log_a[_m],  # degree of "undersaturation"
        )

        # r2: charge conservation
        if system.has_charge_cstr:
            q_vec = system.spc_charge * spc_c
            q_pos = q_vec[q_vec > 0].sum()
            q_neg = -q_vec[q_vec < 0].sum()
            r2 = [np.log10(q_pos - cond.q_ref) - np.log10(q_neg - cond.q_ref)]
        else:
            r2 = []

        # r3: ionic strength
        if system.activity_model == 'none':
            r3 = []
        else:
            r3 = [np.log10(system.ionic_strength(spc_c)) - np.log10(I+1e-100)]

        r = np.concatenate((r0, r1, r2, r3))
        logger.debug('x=%s', x)
        logger.debug('cpt_log_a=%s', cpt_log_a)
        logger.debug('spc_log_a=%s', spc_log_a)
        logger.debug('spc_c=%s', spc_c)
        logger.debug('r0=%s', r0)
        logger.debug('r1=%s', r1)
        logger.debug('qdiff=%s', r2)
        logger.debug('Idiff=%s', r3)
        if return_all_spc:
            res = pd.DataFrame({'spc_c': spc_c, 'spc_a': 10.**spc_log_a},
                               index=[sp.name for sp in system.species])
            return r, res
        return r

    def jac(self, x: npt.ArrayLike, dx: float = 1e-3) -> npt.NDArray[np.float64]:
        """Calculate the Jacobian matrix of the function."""
        x = np.asarray(x)
        J = np.zeros((len(x), len(x)))
        f0 = self(x)
        for i in range(len(x)):
            x_dx = np.copy(x)
            x_dx[i] += dx
            f = self(x_dx)
            J[:, i] = (f - f0) / dx
        return J

    def generate_initial_points(
            self,
            random_points: int = 100,
            random_seed: np.random.Generator | int | None = None,
            condnum_threshold: float | int = 1000.0
    ) -> list[npt.NDArray[np.float64]]:
        """Obtain candidates for initial points, order of which is:
        (1) points with cond(J) <= threshold, in ascending order of rmse.
        (2) points with cond(J) > threshold, in ascending order of rmse.

        Returns:
            list[npt.NDArray[np.float64]]: list of candidate points
        """
        rng = np.random.default_rng(random_seed)

        xs: list[npt.NDArray[np.float64]] = []
        rmses: list[float] = []
        condnums: list[float] = []
        for _ in range(random_points):
            x = rng.uniform(low=self.system.bounds_lower, high=self.system.bounds_upper)
            rmse = np.sqrt(np.average(self(x)**2))
            J = self.jac(x, dx=1e-3)
            xs.append(x)
            rmses.append(rmse)
            try:
                cn = np.linalg.cond(J)
            except np.linalg.LinAlgError:
                logger.info('Failed to calculate condition number.\n  Conditions: %s',
                            self.cond,
                            exc_info=True)
                cn = float('inf')
            condnums.append(cn)

        idx_sorted = np.argsort(rmses)
        l1 = [xs[i] for i in idx_sorted if condnums[i] < condnum_threshold]
        l2 = [xs[i] for i in idx_sorted if condnums[i] >= condnum_threshold]
        return l1 + l2

    def solve(
            self,
            x0: npt.NDArray[np.float64] | None = None,
            max_retries: int = 10,
            rmse_threshold: float = 1e-6,
            random_seed: np.random.Generator | int | None=None,
    ) -> SolverResults:
        """
        Solve the equilibrium problem.

        The solver tries multiple initial guesses until the RMSE of the residual
        falls below `rmse_threshold`.

        If x0 is provided, it is tried first.
        Otherwise, initial guesses are generated automatically.

        random_seed controls reproducibility of initial guesses
        """

        def gen_x0(x0_):
            random_points = max(max_retries, 100)
            if x0_ is not None:
                yield x0_
            rng = np.random.default_rng(random_seed)
            yield from self.generate_initial_points(random_seed=rng, random_points=random_points)

        rmse = float('inf')
        for retries, x0 in enumerate(gen_x0(x0)):
            sol = optimize.least_squares(self, x0=x0,
                                         bounds=(self.system.bounds_lower, self.system.bounds_upper),
                                         method='trf', ftol=1e-12, xtol=1e-12, gtol=1e-12,
                                         max_nfev=self.system.specs.n_vars * 200)
            rmse = np.sqrt(np.average(sol.fun**2))
            if rmse < rmse_threshold:
                break  # acceptable solution found
            if retries >= max_retries:
                break
        if rmse >= rmse_threshold:
            logger.warning(
                'Poor convergence after %d retries (rmse=%g)\n   Conditions: %s',
                retries, rmse, self.cond
            )
        _, df = self(sol.x, return_all_spc=True)
        return SolverResults(
            sol=sol,
            f=self,
            spc_c=df['spc_c'].to_numpy(),
            spc_a=df['spc_a'].to_numpy(),
            retries=retries
        )

    def _unpack_x(self, x: np.ndarray):
        num_loga = self.system.specs.n_loga
        num_solid = self.system.specs.n_solid
        loga, n_solid = x[:num_loga], x[num_loga:num_loga + num_solid]
        if len(x) > num_loga + num_solid:
            I = x[-1]
        else:
            I = float('nan')
        return loga, n_solid, I

@dataclass(frozen=True)
class SolverResults:
    sol: optimize.OptimizeResult = field(repr=False)
    f: FitFunc = field(repr=False)
    spc_c: npt.NDArray[np.float64]
    spc_a: npt.NDArray[np.float64]
    retries: int

    ionic_strength: float = field(init=False)
    rmse: float = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'rmse', np.sqrt(np.average(self.sol.fun**2)))
        object.__setattr__(self, 'ionic_strength', self.f.system.ionic_strength(self.spc_c))

    def total_conc_in_liquid(self, cpt_name: str) -> float:
        system = self.f.system
        try:
            i = [cpt.name for cpt in system.components].index(cpt_name)
        except IndexError:
            raise ValueError(f'{cpt_name} not found in the system.')
        _m = self.f.system.spc_phase == Phase.LIQUID
        return self.spc_c[_m] @ system.stoichiometry_matrix[_m, i]

    def distribution(self, cpt_name: str, relative=False) -> dict[Species, float]:
        system = self.f.system
        try:
            i = [cpt.name for cpt in system.components].index(cpt_name)
        except IndexError:
            raise ValueError(f'{cpt_name} not found in the system.')
        _m = system.stoichiometry_matrix[:, i].ravel() != 0
        _c = self.spc_c[_m] * system.stoichiometry_matrix[_m, i]
        if relative:
            _c = _c / np.nansum(_c) * 100
        return dict(zip(np.array(system.species)[_m], _c))

    def pH(self) -> float:
        for i, s in enumerate(self.f.system.species):
            if s.name in ['H+', 'H⁺', 'H^+', 'H^{+}']:
                return -np.log10(self.spc_a[i])
        raise ValueError('H+ is not found in the system.')

def solve_for_conditions(
    system: System,
    cond_list: t.Iterable[Conditions],
    rmse_thresh=1e-6,
    random_seed=None
    ) -> list[SolverResults]:
    """
    Solves equilibrium problems for a sequence of conditions.

    The solution from the previous condition is reused as the initial guess
    for the next one (warm start), if convergence was successful.
    """
    x0: np.ndarray | None = None
    results: list[SolverResults] = []

    for i, cond in enumerate(cond_list):
        f = FitFunc(system, cond)
        s = f.solve(x0=x0, rmse_threshold=rmse_thresh, random_seed=random_seed)
        results.append(s)
        logger.info("Step %d: rmse=%g", i, s.rmse)
        if s.rmse < rmse_thresh:
            x0 = s.sol.x.copy()
        else:
            x0 = None

    return results
