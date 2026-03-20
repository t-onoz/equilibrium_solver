"""Generic equilibrium solver for aqueous systems.
Supports aqueous spieces and pure solid phases.
Gas phase, solid solution, and activity coefficients are not supported yet."""
from __future__ import annotations
import logging
import typing as t
from enum import IntEnum
from dataclasses import dataclass, field
from collections.abc import Sequence
import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from scipy import optimize
from IPython.display import display

import activity_models as am
from linalg_tools import precompute_Aprime_and_Kmap_checked
from _types import Float

logger = logging.getLogger(__name__)
_handler = logging.StreamHandler()

class Cstr(IntEnum):
    """componentの制約条件を表す列挙型。TOTALは全濃度指定、DIRECTは直接濃度指定、CHARGEは電荷保存条件を表す。"""
    CHARGE = 0
    TOTAL = 1
    DIRECT = 2

class Phase(IntEnum):
    """平衡種の相の種類を表す列挙型。"""
    SOLID = 0
    LIQUID = 1
    GAS = 2

@dataclass(frozen=True)
class SolverResults:
    sol: optimize.OptimizeResult = field(repr=False)
    f: FitFunc = field(repr=False)
    spc_c: npt.NDArray[np.float64]
    spc_a: npt.NDArray[np.float64]
    retries: int
    rmse: float = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'rmse', np.sqrt(np.average(self.sol.fun**2)))

    def total_conc_in_liquid(self, cpt_name: str) -> float:
        system = self.f.system
        try:
            i = np.nonzero(system.cpt_names == cpt_name)[0]
        except IndexError:
            raise ValueError(f'{cpt_name} not found in the system.')
        _m = self.f.system.spc_phase == Phase.LIQUID
        return self.spc_c[_m] @ system.composition_matrix[_m, i]

    def distribution(self, cpt_name: str, relative=False) -> dict[str, float]:
        system = self.f.system
        try:
            i = np.nonzero(system.cpt_names == cpt_name)[0]
        except IndexError:
            raise ValueError(f'{cpt_name} not found in the system.')
        _m = system.composition_matrix[:, i].ravel() != 0
        _c = self.spc_c[_m] * system.composition_matrix[_m, i]
        if relative:
            _c = _c / np.nansum(_c) * 100
        return dict(zip(system.spc_names[_m], _c))

    def pH(self) -> float:
        for i, s in enumerate(self.f.system.spc_names):
            if s in ['H+', 'H⁺', 'H^+', 'H^{+}']:
                return -np.log10(self.spc_a[i])
        raise ValueError('H+ is not found in the system.')


def fischer_burmeister(a, b, eps=0.):
    """a >= 0, b >= 0, かつ ab = 0 のとき0になる関数。
    epsは平滑化定数。"""
    return np.sqrt(a*a + b*b + eps**2) - (a + b)

def vant_Hoff(logK, H, T, T_ref=298.15):
    """温度変化に伴う平衡定数の変化を計算する関数。
    logK: 平衡定数の対数 (log10)
    H: 反応エンタルピー (kJ/mol)
    T: 計算したい温度 (K)
    T_ref: Kが与えられた温度 (K)"""
    R = 8.3145e-3  # kJ/(mol*K)
    K = 10 ** logK
    return np.log10(K * np.exp(-H/R * (1/T - 1/T_ref)))


@dataclass
class EqSystem:
    """平衡モデルをベクトル、行列として格納するためのクラス。"""
    # cpt: component, spc: speciesの略
    # componentとは、平衡系を構成する基本的な化学成分のこと。例えば、水溶液中の平衡系なら、H+, OH-, Na+などがcomponentになる。
    # speciesとは、平衡系に存在する全ての化学種のこと。例えば、水溶液中の平衡系なら、H+, OH-, Na+などがspeciesになる。
    activity_model: t.Literal['debye_huckel', 'ex_debye_huckel', 'davies', 'truesdell_jones', 'none']
    cpt_names: npt.NDArray[np.str_]
    cpt_base: npt.NDArray[np.str_]
    cpt_cstr: npt.NDArray[np.uint8]
    cpt_charge: npt.NDArray[np.integer]
    spc_names: npt.NDArray[np.str_]
    spc_phase: npt.NDArray[np.uint8]

    # spc_logK: speciesの活量の対数を求めるための定数項。
    # spc_logK[i]は、species iの活量の対数を求めるときに、組成行列とcomponentの活量の対数からさらに足す定数項。
    spc_logK: npt.NDArray[np.float64]

    composition_matrix: npt.NDArray[np.number]
    temperature_K: float = 298.15
    has_charge_cstr: bool = field(init=False)
    n_loga_free: int = field(init=False)
    n_phase_amount: int = field(init=False)
    n_ionic_strength: int = field(init=False)

    # gamma_a, gamma_b are used to determine activity coefficient.
    spc_gamma_a: npt.NDArray[np.float64] = field(default_factory=lambda: np.asarray(0.0))
    spc_gamma_b: npt.NDArray[np.float64] = field(default_factory=lambda: np.asarray(0.0))

    spc_charge: npt.NDArray[np.number] = field(init=False)

    def __post_init__(self):
        self.has_charge_cstr = Cstr.CHARGE in self.cpt_cstr

        self.n_loga_free = np.sum(self.cpt_cstr != Cstr.DIRECT)
        self.n_phase_amount = np.sum(self.spc_phase == Phase.SOLID)
        self.n_ionic_strength = 0 if self.activity_model == 'none' else 1

        self.spc_charge = self.composition_matrix @ self.cpt_charge
        if self.temperature_K < 273 or self.temperature_K > 374:
            t_deg = self.temperature_K - 273.15
            logger.warning('temperature is set to %s degC. Is it really correct?', t_deg)
        # validations
        for i, name in enumerate(self.spc_names):
            if self.spc_phase[i] == Phase.LIQUID:
                continue
            if np.abs(self.spc_charge[i]) > 1e-5:
                raise ValueError(f'gas/solid species ({name}) has charge {self.spc_charge[i]}. Charged gas/solid is not supported.')

            idx_cpt = [j for j, v in enumerate(self.composition_matrix[i, :]) if np.abs(v) > 1e-5]
            if np.any(self.cpt_cstr[idx_cpt] == Cstr.TOTAL) and self.spc_phase[i] == Phase.GAS:
                raise ValueError(f'Cstr. related to gas species ({name}) must not be TOTAL.')
        if self.activity_model != 'none' and not self.has_charge_cstr:
            raise ValueError('Please set a charge constraint when using activity models.')


    def gamma(self, I: float) -> npt.NDArray[np.float64]:
        z = self.spc_charge
        g = np.ones_like(self.spc_names, dtype=np.float64)

        # liquid phase
        _m = self.spc_phase == Phase.LIQUID
        if self.activity_model == 'none':
            g_liq: Float = 1.0
        elif self.activity_model == 'debye_huckel':
            g_liq = am.debye_huckel(I, z[_m], self.temperature_K)
        elif self.activity_model == 'ex_debye_huckel':
            g_liq = am.ex_debye_huckel(I, z[_m], self.temperature_K, self.spc_gamma_a[_m])
        elif self.activity_model == 'davies':
            g_liq = am.davies(I, z[_m], self.temperature_K)
        elif self.activity_model == 'truesdell_jones':
            g_liq = am.truesdell_jones(I, z[_m], self.temperature_K, self.spc_gamma_a[_m], self.spc_gamma_b[_m])
        else:
            raise ValueError(f'unknown activity_model: {self.activity_model}')
        g[_m] = g_liq

        return g


    @property
    def n_vars(self):
        return self.n_loga_free + self.n_phase_amount + self.n_ionic_strength

    @classmethod
    def from_yaml(cls, path, temperature=298.15, display_info=False) -> EqSystem:
        """Generates EqSystem object from YAML files."""
        with open(path, 'r', encoding='utf-8') as fp:
            config = yaml.safe_load(fp)
        return cls.from_config(config, temperature, display_info)

    @classmethod
    def from_config(cls, config: dict, temperature=298.15, display_info=False) -> EqSystem:
        """Generates EqSystem object from dictionary.
        For contents of `config`, refer to YAML files in the `examples` folder."""
        for e in config['equilibria']:
            _required = ['name', 'logK', 'composition']
            _optional = ['DH']
            for r in _required:
                if r not in e:
                    raise ValueError(f'missing "{r}" in {e}')
            for key in e:
                if key not in _required + _optional:
                    raise ValueError(f'unknown key "{key}" in {e}')
        df_cpt = pd.DataFrame.from_records(config['components'])
        df_spc = pd.DataFrame.from_records(config['species']).fillna(0.0)
        df_eq = pd.DataFrame.from_records(config['equilibria'])
        df_compo = pd.json_normalize(df_eq['composition'].to_list()).fillna(0.0)
        activity_model = config['activity_model']
        if activity_model not in am.functions:
            raise ValueError(f'unknown activity model: {activity_model}')

        l = [name for name in df_cpt['base'] if name not in df_spc['name'].to_list()]
        if l:
            raise ValueError(f'species {l} found in "components," but not in "species."')
        l = [name for name in df_compo.columns if name not in df_spc['name'].to_list()]
        if l:
            raise ValueError(f'species {l} found in "equilibria," but not in "species."')

        if display_info:
            display(f"activity model: {activity_model}")
            display("---------- components ----------")
            display(df_cpt)
            display("---------- species ----------")
            display(df_spc)
            display("---------- equilibria ----------")
            display(pd.concat([df_eq, df_compo], axis='columns'))

        if len(df_spc) != len(df_cpt) + len(df_eq):
            raise ValueError(
                f'expected n(spc) = n(cpt) + n(eq). got {len(df_spc)} != {len(df_cpt)} + {len(df_eq)}'
                )
        df_eq['logK_orig'] = df_eq['logK']
        df_eq['logK'] = vant_Hoff(df_eq['logK'], df_eq['DH'], temperature, config['temperature'])

        cpt_names = df_cpt['name'].to_numpy(dtype=np.str_)
        cpt_base = df_cpt['base'].to_numpy(dtype=np.str_)
        cpt_cstr = np.array(
            [getattr(Cstr, v.upper()) for v in df_cpt['constraint'].values],
            dtype=np.uint8
        )
        cpt_charge = df_cpt['charge'].to_numpy()

        spc_names = df_spc['name'].to_numpy(dtype=np.str_)
        for n in spc_names:
            if n not in df_compo.columns:
                df_compo[n] = 0

        A = df_compo[spc_names].to_numpy(dtype=np.float64)
        free_idx = [spc_names.tolist().index(s) for s in df_cpt['base']]
        r = precompute_Aprime_and_Kmap_checked(A, free_idx)

        spc_phase = np.array([
            getattr(Phase, str(ph).upper()) for ph in df_spc['phase']
        ], dtype=np.uint8)

        spc_logK = np.round(r.T @ df_eq['logK'].to_numpy(dtype=np.float64), 3)

        composition_matrix = np.round(r.Aprime, 3)

        if display_info:
            display('---------- species from base components ----------')
            _df = pd.DataFrame(composition_matrix, index=spc_names, columns=df_cpt['base'].to_list())
            _df['logK'] = spc_logK
            display(_df)

        s = EqSystem(
            activity_model=activity_model,
            cpt_names=cpt_names,
            cpt_base=cpt_base,
            cpt_cstr=cpt_cstr,
            spc_names=spc_names,
            cpt_charge=cpt_charge,
            spc_phase=spc_phase,
            spc_logK=spc_logK,
            composition_matrix=composition_matrix,
            temperature_K=temperature,
        )
        if 'a' in df_spc.columns:
            s.spc_gamma_a = df_spc['a'].to_numpy()
        if 'b' in df_spc.columns:
            s.spc_gamma_b = df_spc['b'].to_numpy()
        return s

    def make_f(self, cond: Conditions):
        """フィッティング用の関数を生成する。"""
        return FitFunc(system=self, cond=cond)

    def get_arg_names(self) -> list[str]:
        """Return what each element of x vector means."""
        result = []
        for i in range(self.n_loga_free):
            cpt_base = self.cpt_base[self.cpt_cstr != Cstr.DIRECT][i]
            result.append(f'Log a({cpt_base})')
        for i in range(self.n_phase_amount):
            phase_name = self.spc_names[self.spc_phase == Phase.SOLID][i]
            result.append(f'mole({phase_name})')
        for i in range(self.n_ionic_strength):
            result.append('I')
        return result

    def get_return_names(self) -> list[str]:
        """Return what each element of return values means."""
        result = []

        _m = self.cpt_cstr == Cstr.TOTAL
        n_total = np.sum(_m, dtype=int)
        for i in range(n_total):
            cpt_name = self.cpt_names[_m][i]
            result.append(f'{cpt_name} mole conservation')

        _m = self.spc_phase == Phase.SOLID
        n_solid = np.sum(_m, dtype=int)
        for i in range(n_solid):
            sld_name = self.spc_names[_m][i]
            result.append(f'solid-liquid eq. of {sld_name}')

        if self.has_charge_cstr:
            result.append(f'charge conservation')

        if self.activity_model != 'none':
            result.append(f'self-consistency of I')

        return result


@dataclass
class FitFunc:
    """平衡計算のフィッティング関数を定義するクラス。
    EqSystem.make_fで生成される関数はこのクラスのインスタンスになる。"""
    system: EqSystem
    cond: Conditions
    q_ref: float = field(init=False)
    bounds_lower: t.Sequence[float] = field(init=False, repr=False)
    bounds_upper: t.Sequence[float] = field(init=False, repr=False)
    arg_names: tuple[str, ...] = field(init=False)
    return_names: tuple[str, ...] = field(init=False)

    def __post_init__(self):
        # q_ref calculation
        q_vec = self.system.cpt_charge * self.cond.cpt_c_direct
        q_pos = np.sum(q_vec[q_vec > 0])
        q_neg = -1 * np.sum(q_vec[q_vec < 0])
        self.q_ref = np.min([q_pos, q_neg])
        if q_pos == q_neg:
            self.q_ref *= 1 - 1e-8

        n0 = self.system.n_loga_free
        n1 = self.system.n_phase_amount + self.system.n_ionic_strength
        self.bounds_lower = (-100.,) * n0 + (0.,) * n1
        self.bounds_upper = (2.,) * n0 + (10.,) * n1
        self.arg_names = tuple(self.system.get_arg_names())
        self.return_names = tuple(self.system.get_return_names())

    @t.overload
    def __call__(self, x: t.Any, return_all_spc: t.Literal[False] = False) \
        -> npt.NDArray[np.float64]: ...

    @t.overload
    def __call__(self, x: t.Any, return_all_spc: t.Literal[True] = True) \
        -> tuple[npt.NDArray[np.float64], pd.DataFrame]: ...

    def __call__(self, x: Sequence[float] | npt.NDArray[np.float64], return_all_spc=False) \
        -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], pd.DataFrame]:
        """function to optimize.

        Args:
            x : first `n_loga_free` elements represent log(activity) of free (TOTAL or CHARGE) components.
                next `n_phase_amount` elements represent mole amount of solids.
                if activity_model != 'none', additional 1 element represents ionic strength.
            return_all_spc: if False, returns a residual vector.
                            if True, returns a tuple of (residual vector, DataFrame),
                            where DataFrame consists of "concentration" and "activity" columns.
        """
        system = self.system
        cond = self.cond

        if len(x) != system.n_vars:
            raise ValueError(f'len(x) must be {system.n_vars}, but got {repr(x)}.')

        # determine γ (activity coefficient) based on the activity model
        spc_gamma = system.gamma(0.0 if system.activity_model == 'none' else x[-1])
        _l = list(system.spc_names)
        cpt_gamma = spc_gamma[[_l.index(n) for n in system.cpt_base]]

        # initialize log activities of the components
        cpt_log_a = np.zeros_like(system.cpt_base, dtype=float)
        # activity = γ * c for DIRECT components
        _cpt_a_direct = (cpt_gamma * cond.cpt_c_direct)[system.cpt_cstr == Cstr.DIRECT]
        with np.errstate(divide='ignore'):
            cpt_log_a[system.cpt_cstr == Cstr.DIRECT] = np.maximum(np.log10(_cpt_a_direct), -9999)
        # for TOTAL or CHARGE components, log(activity) are taken from x.
        cpt_log_a[system.cpt_cstr != Cstr.DIRECT] = x[:system.n_loga_free]

        spc_log_a = system.composition_matrix @ cpt_log_a + system.spc_logK

        # initialize concentration of species
        spc_c = np.zeros_like(spc_log_a)
        # for liquid phase, c = a / γ
        _m = system.spc_phase == Phase.LIQUID
        spc_c[_m] = 10. ** spc_log_a[_m] / spc_gamma[_m]
        # for solid phases, c values are taken from x.
        spc_c[system.spc_phase == Phase.SOLID] = \
            x[system.n_loga_free:] if system.activity_model == 'none' else x[system.n_loga_free:-1]
        # c vales of gaseous molecules should be kept 0,
        # because we can't determine gas phase volume.

        # r0: total mole conservation
        _c_total = spc_c @ system.composition_matrix[:, system.cpt_cstr == Cstr.TOTAL]
        _c0 = cond.cpt_c_total[system.cpt_cstr == Cstr.TOTAL]
        with np.errstate(divide='ignore'):
            r0 = np.where(
                _c0 == 0,
                _c_total,
                np.log10(_c0) - np.log10(_c_total)
            )

        # r1: solid-liquid equilibrium
        # described using "complementary conditions" (c_solid > 0 & a(solid)=1, or c_solid == 0 & a(solid) < 1)
        _m = system.spc_phase == Phase.SOLID
        r1 = fischer_burmeister(
            spc_c[_m],  # mole of solid phase
            -spc_log_a[_m],  # degree of "undersaturation"
        )

        # r2: charge conservation
        if system.has_charge_cstr:
            q_vec = system.spc_charge * spc_c
            q_pos = np.sum(q_vec[q_vec > 0]) - self.q_ref
            q_neg = np.sum(q_vec[q_vec < 0]) * (-1.0) - self.q_ref
            r2 = [np.log10(q_pos) - np.log10(q_neg)]
        else:
            r2 = []

        # r3: ionic strength
        if system.activity_model == 'none':
            r3 = []
        else:
            _m = system.spc_phase == Phase.LIQUID
            r3 = [np.log10(np.sum(0.5 * spc_c[_m] * system.spc_charge[_m]**2)) - np.log10(x[-1]+1e-100)]

        r = np.concatenate((r0, r1, r2, r3))
        logger.debug('x=%s', x)
        logger.debug('cpt_log_a=%s', cpt_log_a)
        logger.debug('spc_log_a=%s', spc_log_a)
        logger.debug('spc_c=%s', spc_c)
        logger.debug('r0=%s', r0)
        logger.debug('r1=%s', r1)
        logger.debug('qdiff=%s', r2)
        if return_all_spc:
            res = pd.DataFrame({'spc_c': spc_c, 'spc_a': 10.**spc_log_a}, index=system.spc_names)
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
            x = rng.uniform(low=self.bounds_lower, high=self.bounds_upper)
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
            rmse_threshold: float = 1e-6,
            random_seed: np.random.Generator | int | None=None,
    ) -> SolverResults:
        """Solves equilibrium problems for a list of different conditions.
            When RMSE is lower than `rmse_thresh` (default: 1e-6),
            retries optimization with different initial points.
            `rng` (Generator, optional) is used when making initial point candidates."""

        def gen_x0(x0_):
            if x0_ is not None:
                yield x0_
            rng = np.random.default_rng(random_seed)
            yield from self.generate_initial_points(random_seed=rng)

        rmse = float('inf')
        retries = -1
        for x0 in gen_x0(x0):
            retries += 1
            sol = optimize.least_squares(self, x0=x0, bounds=(self.bounds_lower, self.bounds_upper),
                                         method='trf', ftol=1e-12, xtol=1e-12, gtol=1e-12,
                                         max_nfev=self.system.n_vars * 200)
            rmse = np.sqrt(np.average(sol.fun**2))
            if rmse < rmse_threshold:
                break
        else:
            logger.warning('Poor convergence! (rmse=%s)\n   Conditions: %s', rmse, self.cond)
        assert retries >= 0

        _, df = self(sol.x, return_all_spc=True)
        return SolverResults(
            sol=sol,
            f=self,
            spc_c=df['spc_c'].to_numpy(),
            spc_a=df['spc_a'].to_numpy(),
            retries=retries
        )

@dataclass(frozen=True)
class Conditions:
    """Constraint conditions for equilibrium solver.
    There are two types of constraints.
        - "direct": mole of a base species of component
        - "total": total mole of a component
    Examples:
        - direct: pCO2 in atm, [H+] in mol/L
        - total: formal mole of Ca2+ (Ca2+ + [Ca(OH)]+ + Ca(OH)2(s) + ...)
        """
    system: EqSystem = field(repr=False)
    cpt_c_direct: npt.NDArray[np.float64]
    cpt_c_total: npt.NDArray[np.float64]

    @classmethod
    def from_dict(cls, system: EqSystem, concentrations: dict[str, float]) -> Conditions:
        """Generate Conditions object from dictionary.
        `concentrations` is a dictionary of {cpt_name: constraint_value}.
        """
        for i, name in enumerate(system.cpt_names[system.cpt_cstr != Cstr.CHARGE]):
            if name not in concentrations:
                raise ValueError(
                    f'component "{name}" is missing in dict.'
                    )
        cpt_c_total = np.full_like(system.cpt_names, np.nan, dtype=float)
        cpt_c_direct = np.full_like(system.cpt_names, np.nan, dtype=float)
        for i, n in enumerate(system.cpt_names):
            if n in concentrations:
                if system.cpt_cstr[i] == Cstr.DIRECT:
                    cpt_c_direct[i] = concentrations[n]
                if system.cpt_cstr[i] == Cstr.TOTAL:
                    cpt_c_total[i] = concentrations[n]
        return cls(cpt_c_direct=cpt_c_direct, cpt_c_total=cpt_c_total, system=system)

    def __post_init__(self):
        assert len(self.system.cpt_names) == len(self.cpt_c_total)
        assert len(self.system.cpt_names) == len(self.cpt_c_direct)

def solve_for_conditions(
    system: EqSystem,
    cond_list: t.Iterable[Conditions],
    rmse_thresh=1e-6,
    rng=None
    ) -> list[SolverResults]:
    """Solves equilibrium problems for a list of different conditions.
    To accelerate convergence, the solution obtained is used as an initial point for the next step."""
    x0: np.ndarray | None = None
    results: list[SolverResults] = []

    for cond in cond_list:
        f = system.make_f(cond)
        s = f.solve(x0=x0, rmse_threshold=rmse_thresh, random_seed=rng)
        results.append(s)
        if s.rmse < rmse_thresh:
            x0 = s.sol.x
        else:
            x0 = None

    return results

