from __future__ import annotations
import typing as t
import pandas as pd
from collections.abc import Sequence
from eq_solver import SolverResults, Cstr

__all__ = ['get_summary', 'get_all', 'get_distribution']

def _header(r: SolverResults) -> dict[str, t.Any]:
    h = {}
    for i, cpt_name in enumerate(r.f.system.cpt_names):
        if r.f.system.cpt_cstr[i] == Cstr.TOTAL:
            h[f'cond.total.{cpt_name}'] = r.f.cond.cpt_c_total[i]
        if r.f.system.cpt_cstr[i] == Cstr.DIRECT:
            h[f'cond.direct.{cpt_name}'] = r.f.cond.cpt_c_direct[i]
    h['sol.rmse'] = r.rmse
    h['sol.retries'] = r.retries
    h['sol.nfev'] = r.sol.nfev
    try:
        h['pH'] = r.pH()
    except ValueError:
        pass
    return h

def _c_species(r: SolverResults) -> dict[str, t.Any]:
    return dict(zip(r.f.system.spc_names, r.spc_c))

def _a_species(r: SolverResults) -> dict[str, t.Any]:
    return dict(zip(r.f.system.spc_names, r.spc_a))

def _c_total_aq(r: SolverResults) -> dict[str, t.Any]:
    return {
        f'total {cpt_name}(aq)': r.total_conc_in_liquid(cpt_name) for cpt_name in r.f.system.cpt_names
    }

@t.overload
def get_summary(x: SolverResults) -> pd.Series: ...

@t.overload
def get_summary(x: Sequence[SolverResults]) -> pd.DataFrame: ...

def get_summary(x: SolverResults | Sequence[SolverResults]) -> pd.Series | pd.DataFrame:
    records = []
    l = [x] if isinstance(x, SolverResults) else x
    for r in l:
        d = _header(r)
        d.update(_c_total_aq(r))
        records.append(d)
    if isinstance(x, SolverResults):
        return pd.Series(records[0])
    return pd.DataFrame.from_records(records)


@t.overload
def get_all(x: SolverResults) -> pd.Series: ...

@t.overload
def get_all(x: Sequence[SolverResults]) -> pd.DataFrame: ...

def get_all(x: SolverResults | Sequence[SolverResults]) -> pd.Series | pd.DataFrame:
    records = []
    l = [x] if isinstance(x, SolverResults) else x
    for r in l:
        d = _header(r)
        d.update({f'c.{k}': v for k, v in _c_species(r).items()})
        d.update({f'a.{k}': v for k, v in _a_species(r).items()})
        records.append(d)
    if isinstance(x, SolverResults):
        return pd.Series(records[0])
    return pd.DataFrame.from_records(records)

@t.overload
def get_distribution(x: SolverResults, cpt_name: str, relative: bool=False) -> pd.Series: ...

@t.overload
def get_distribution(x: Sequence[SolverResults], cpt_name: str, relative: bool=False) -> pd.DataFrame: ...

def get_distribution(x: SolverResults | Sequence[SolverResults], cpt_name: str, relative: bool=False):
    l = [x] if isinstance(x, SolverResults) else x
    records = []
    for r in l:
        d = _header(r)
        d.update(r.distribution(cpt_name, relative=relative))
        records.append(d)
    if isinstance(x, SolverResults):
        return pd.Series(records[0])
    return pd.DataFrame.from_records(records)
