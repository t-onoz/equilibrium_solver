from __future__ import annotations
import typing as t
from collections.abc import Sequence
import pandas as pd

from eq_solver.system import Constraint
from eq_solver.solver import SolverResults


def _header(r: SolverResults) -> dict[str, t.Any]:
    h = {}
    for i, cpt in enumerate(r.f.system.components):
        if cpt.constraint == Constraint.TOTAL:
            h[f'cond.total.{cpt.name}'] = r.f.cond.values[i]
        if r.f.system.cpt_cstr[i] == Constraint.DIRECT:
            h[f'cond.direct.{cpt.name}'] = r.f.cond.values[i]
    h['sol.rmse'] = r.rmse
    h['sol.retries'] = r.retries
    h['sol.nfev'] = r.sol.nfev
    try:
        h['pH'] = r.pH()
    except ValueError:
        pass
    return h

def _c_species(r: SolverResults) -> dict[str, t.Any]:
    return dict(zip([sp.name for sp in r.f.system.species], r.spc_c))

def _a_species(r: SolverResults) -> dict[str, t.Any]:
    return dict(zip([sp.name for sp in r.f.system.species], r.spc_a))

def _c_total_aq(r: SolverResults) -> dict[str, t.Any]:
    return {
        cpt.name: r.total_conc_in_liquid(cpt.name) for cpt in r.f.system.components
    }

def get_total_aq(x: SolverResults | Sequence[SolverResults]) -> pd.DataFrame:
    """calculates total dissolved concentration of each component"""
    records = []
    l = [x] if isinstance(x, SolverResults) else x
    for r in l:
        d = _header(r)
        d.update(_c_total_aq(r))
        records.append(d)
    return pd.DataFrame.from_records(records)

def get_distribution(
        x: SolverResults | Sequence[SolverResults],
        cpt_name: str,
        relative: bool = False,
) -> pd.DataFrame:
    """calculates distribution of concentration of a given component (including solid phases)"""
    records =[]
    l = [x] if isinstance(x, SolverResults) else x
    for r in l:
        d = _header(r)
        d.update({sp.name: val for sp, val in
                  r.distribution(cpt_name, relative=relative).items()})
        records.append(d)
    return pd.DataFrame.from_records(records)

def get_summary(x: SolverResults | Sequence[SolverResults]) -> pd.DataFrame:
    records = []
    l = [x] if isinstance(x, SolverResults) else x
    for r in l:
        d = _header(r)
        d['ionic strength'] = r.ionic_strength
        d.update({f'c_total.{key}': val for key, val in _c_total_aq(r).items()})
        d.update({f'c.{key}': val for key, val in _c_species(r).items()})
        d.update({f'a.{key}': val for key, val in _a_species(r).items()})
        records.append(d)
    return pd.DataFrame.from_records(records)
