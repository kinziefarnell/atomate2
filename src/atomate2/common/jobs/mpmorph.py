"""Define utility functions for amorphous structure equilibration.

This file generalizes the MPMorph workflows of
https://github.com/materialsproject/mpmorph
originally written in atomate for VASP only to a more general
code agnostic form.

For information about the current flows, contact:
- Bryant Li (@BryantLi-BLI)
- Aaron Kaplan (@esoteric-ephemera)
- Max Gallant (@mcgalcode)
"""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from jobflow import Job
from pymatgen.core import Composition, Molecule, Structure
from pymatgen.io.packmol import PackmolBoxGen
from scipy import stats

if TYPE_CHECKING:
    from pymatgen.core import Element, Species

_DEFAULT_AVG_VOL_FILE = Path("~/.cache/atomate2").expanduser() / "db_avg_vols.json.gz"
if not _DEFAULT_AVG_VOL_FILE.parents[0].exists():
    os.makedirs(_DEFAULT_AVG_VOL_FILE.parents[0], exist_ok=True)
_DEFAULT_AVG_VOL_URL = "https://figshare.com/ndownloader/files/49704288"


from jobflow import JobConfig, job


def _trajectory_and_units(md_job_output):
    """Return (trajectory, energy_key, stress_in_kbar).

    Stress is already in kbar for VASP and for ASE when using emmet AtomTrajectory;
    otherwise (ASE pymatgen Trajectory) stress is in eV/Å³.
    """
    try:
        traj = md_job_output.vasp_objects["trajectory"]
        return traj, "e_wo_entrp", True
    except Exception:
        traj = md_job_output.objects["trajectory"]
        try:
            from emmet.core.trajectory import AtomTrajectory
            in_kbar = isinstance(traj, AtomTrajectory)
        except ImportError:
            in_kbar = False
        return traj, "energy", in_kbar


def _stress_to_3x3_kbar(stress, in_kbar: bool):
    """Return 3x3 stress in kbar. Uses ASE convention: Voigt eV/Å³ * (-10/GPa) -> kbar."""
    from ase.stress import voigt_6_to_full_3x3_stress
    from ase.units import GPa

    arr = np.asarray(stress, dtype=float)
    if arr.shape == (6,):
        if in_kbar:
            return voigt_6_to_full_3x3_stress(arr)
        # Voigt stress in eV/Å³ -> 3x3 in kbar (same factor as ASE schema / MP convention)
        return voigt_6_to_full_3x3_stress(arr * (-10 / GPa))
    if arr.shape == (3, 3):
        if in_kbar:
            return arr
        return arr * (-10 / GPa)  # eV/Å³ -> kbar
    raise ValueError(f"Stress must be Voigt (6,) or 3x3, got shape {arr.shape}")


@job()
def extract_trajectory_frames(md_job_output, converge_check=False):
    """Extract trajectory-averaged energy, temperature, pressure, and stress.

    Pressure and stress are returned in kbar (VASP/emmet already in kbar; ASE
    eV/Å³ converted with -10/GPa factor, MP convention) so the convergence
    threshold can be specified in kbar.
    """
    trajectory, energy_name, stress_in_kbar = _trajectory_and_units(md_job_output)
    length = len(trajectory)
    num_last_frames = int(length * 0.10) + 1
    if converge_check:
        num_last_frames = length

    frames = trajectory.frame_properties
    frame_slice = frames[-num_last_frames:]

    stresses_kbar = [
        _stress_to_3x3_kbar(f["stress"], stress_in_kbar) for f in frame_slice
    ]
    pressures_kbar = [np.trace(s) / 3 for s in stresses_kbar]

    trajectory_data = {
        "energy": np.mean([f[energy_name] for f in frame_slice]),
        "temperature": np.mean([f["temperature"] for f in frame_slice]),
        "pressure": np.mean(pressures_kbar),
        "stress": np.mean(stresses_kbar, axis=0),
    }

    if converge_check:
        num_atoms = len(trajectory[0].species)
        energies = [f[energy_name] for f in frame_slice]
        norm_energies = np.array(energies) / num_atoms
        mu, _ = stats.norm.fit(norm_energies)
        half = len(norm_energies) // 2
        mu1, _ = stats.norm.fit(norm_energies[:half])
        mu2, _ = stats.norm.fit(norm_energies[half:])
        trajectory_data["ionic"] = np.abs((mu2 - mu1) / mu)

    return trajectory_data


@job(config=JobConfig(expose_store=True, resolve_references=False))
def md_summary_from_uuid(
    md_job_uuid: str,
    converge_check: bool = False,
    *,
    _md_state_ref=None,
):
    """
    Summarize an MD TaskDoc in the store by UUID using trajectory averages.

    This job:
    - Loads the full MD task document from the current JobStore using `md_job_uuid`.
    - Reuses `extract_trajectory_frames` logic on that document to compute
      average energy, temperature, pressure (kbar), stress, and ionic metric.
    - Returns a small summary dict with those values plus:
      - `md_output_uuid`: the UUID of the full MD TaskDoc in the store.
      - `trajectory_metadata`: basic info like n_frames and n_sites.

    The optional keyword-only argument ``_md_state_ref`` can be used to pass a
    small piece of the MD job output (e.g. ``md_job.output.state``) purely to
    establish a jobflow dependency. It is intentionally unused inside this job
    and, with ``resolve_references=False``, will not be resolved.
    """
    from jobflow import CURRENT_JOB

    if md_job_uuid is None:
        raise ValueError("md_summary_from_uuid requires the MD job UUID string.")
    md_job_uuid = str(md_job_uuid)

    store = getattr(CURRENT_JOB, "store", None)
    if store is None:
        raise RuntimeError(
            "md_summary_from_uuid requires a JobStore; run with a manager that "
            "provides a store and expose_store=True."
        )

    # load=True so the store inlines blob data (e.g. trajectory) from additional_stores["data"],
    # replacing blob_uuid refs with the actual serialized objects for MontyDecoder
    full_doc = store.get_output(md_job_uuid, load=True)
    if full_doc is None:
        raise ValueError(f"No MD task document found in store for uuid {md_job_uuid!r}")

    # Decode Monty-serialized dict (and nested trajectory) to objects
    from monty.serialization import MontyDecoder

    full_doc = MontyDecoder().process_decoded(full_doc)

    # If still a dict, coerce to ForceFieldTaskDocument for _trajectory_and_units / extract_trajectory_frames
    if isinstance(full_doc, dict):
        from atomate2.forcefields.schemas import ForceFieldTaskDocument

        full_doc = ForceFieldTaskDocument.model_validate(full_doc)

    # Reuse underlying extract_trajectory_frames function (original, undecorated)
    summary = extract_trajectory_frames.original(
        full_doc, converge_check=converge_check
    )

    # Basic trajectory metadata (frame count, site count)
    traj, _, _ = _trajectory_and_units(full_doc)
    n_frames = len(traj)
    n_sites = len(traj[0].species) if n_frames else None

    summary["md_output_uuid"] = md_job_uuid
    summary["trajectory_metadata"] = {
        "n_frames": n_frames,
        "n_sites": n_sites,
    }
    return summary


#def optimize_vol(p0, v0, p1, v1, rescale_scheme):
def optimize_vol(volumes, pressures, rescale_scheme):
    # code taken from old mpmorph

    p1 = pressures[-1]
    v1 = volumes[-1]
    if rescale_scheme == "thermo" or len(volumes) == 1:
        beta = 5e-7
        target_pressure = 0
        vol_change = np.exp(-beta * (target_pressure - p1))
        return v1 * vol_change
    
    p0 = pressures[-2]
    v0 = volumes[-2]

    if rescale_scheme == "linear":
        new_volume = ((v1 * p0) - (p1 * v0)) / (p0 - p1)
        return new_volume
    
    if rescale_scheme == "poly":
         if len(volumes) == 2:
             eqs = np.poly1d(np.polyfit(volumes, pressures, 1))
         else:
             eqs = np.poly1d(np.polyfit(volumes, pressures, 2))


"""
@job
def convergence_check(md_job_output):
    if md_job_output.vasp_objects:
        traj = md_job_output.vasp_objects['trajectory']
    elif md_job_output.ase_objects:
        traj = md_job_output.ase_objects['trajectory']
    
    # so you have your trajectory
    convergence_data = {"pressure": 0,
                        "ionic": }

    # extract the pressure from the trajectory - average pressure over run
    pressures = []



    working_outputs["avg-pressure"] = np.mean(pressures)

    # extract the energy from the trajectory

    # calculate min, max, and average energy 
    np.max(energies)
    np.min(energies)
    np.mean(energies)

    # figure out how they do this second check in old mpmorph
    working_outputs["energy_diff"] = max - min

    # figure out what you need for third check in mpmorph
        
    return working_outputs

"""


def _get_average_volumes_file(
    chunk_size: int = 2048, timeout: float = 60
) -> pd.DataFrame:
    """
    Retrieve stored average volume data from figshare if needed.

    Parameters
    ----------
    chunk_size : int = 2048
        Chunk size for downloading from figshare
    timeout : float = 60
        Timeout time in seconds to wait for the request to resolve
    """
    if not _DEFAULT_AVG_VOL_FILE.exists():
        import requests  # type: ignore[import-untyped]

        stream_data = requests.get(_DEFAULT_AVG_VOL_URL, stream=True, timeout=timeout)
        with open(str(_DEFAULT_AVG_VOL_FILE), "wb") as file:
            file.writelines(stream_data.iter_content(chunk_size=chunk_size))

    return pd.read_json(_DEFAULT_AVG_VOL_FILE, orient="split")


def get_average_volume_from_mp_api(
    composition: Composition, mp_api_key: str | None = None
) -> float:
    """
    Get the average volume per atom for a given composition from the Materials Project.

    This function will make API calls to the Materials Project.
    Check Materials Project API documentation for more
    information https://next-gen.materialsproject.org/api.

    Parameters
    ----------
    composition : Composition
        The target composition.
    mp_api_key : str or None
        The user's MP API key.

    Returns
    -------
    float
        The average volume per atom for the composition in Angstrom^3.
    """
    from mp_api.client import MPRester

    with MPRester(api_key=mp_api_key) as mpr:
        comp_entries = mpr.get_entries(composition.reduced_formula, inc_structure=True)

    vols = [
        entry.structure.volume / entry.structure.num_sites for entry in comp_entries
    ]

    if not vols:
        # Find all Materials project entries containing the elements in the
        # desired composition to estimate starting volume.
        with MPRester() as mpr:
            _entries = mpr.get_entries_in_chemsys(
                [str(el) for el in composition.elements], inc_structure=True
            )

        # Only take entries with at least two elements in common with target composition
        entries = [
            entry
            for entry in _entries
            if len(set(composition).intersection(set(entry.structure.composition))) > 1
        ]

        vols = [entry.structure.volume / entry.structure.num_sites for entry in entries]

    # Fallback: mix atomic volume by relative weight in composition
    if not vols:
        by_comp: dict[Element | Species, list[float]] = {
            ele: [] for ele in composition.elements
        }
        for entry in _entries:
            if len(entry.composition.elements) == 1:
                by_comp[entry.composition.elements[0]].append(
                    entry.structure.volume / entry.structure.num_sites
                )
        vols = [
            coeff * np.mean(by_comp[ele]) / composition.num_atoms
            for ele, coeff in composition.items()
        ]

        if any(not v for v in by_comp.values()):
            raise ValueError(
                "No unary data for "
                f"{', '.join(str(k) for k, v in by_comp.items() if not v)}."
            )

    return np.mean(vols)


def get_average_volume_from_db_cached(
    composition: Composition,
    db_name: str,
    cache_file: pd.DataFrame | None = None,
    ignore_oxi_states: bool = True,
) -> float:
    """
    Get the average volume per atom for a given composition from cached data.

    This function uses cached data to accelerate the volume/atom search.

    Parameters
    ----------
    composition : Composition
        The target composition.
    db_name : str
        Name of the database to pull data from.
    cache_file : pandas DataFrame or None (default)
        DataFrame containing cached volumes.
        Should match the format of the data in _DEFAULT_AVG_VOL_FILE,
        and have the following columns:
            "chem_env", "avg_vol", "count", "with_oxi", "source"
    ignore_oxi_states : bool = True
        Whether to ignore oxidation state data.

    Returns
    -------
    float
        The average volume per atom for the composition.
    """
    avg_vols = cache_file or _get_average_volumes_file()

    avg_vols = avg_vols[avg_vols["source"] == db_name]
    return get_average_volume_from_database(
        composition,
        avg_vols=avg_vols,
        ignore_oxi_states=ignore_oxi_states,
    )


def get_average_volume_from_mp(
    composition: Composition, use_cached: bool = True, **kwargs
) -> float:
    """
    Get the average volume per atom for a given composition from MP data.

    This function will either make MP API calls or used cached data for
    the search.

    Parameters
    ----------
    composition : Composition
        The target composition.
    use_cached : bool = True
        Whether to use cached MP data (True) or make calls to the MP API (False)
    **kwargs : kwargs to pass to the volume/atom search functions, see
        `get_average_volume_from_db_cached`,
        `get_average_volume_from_mp_api`
        for specific kwargs.

    Returns
    -------
    float
        The average volume per atom for the composition.
    """
    if use_cached:
        return get_average_volume_from_db_cached(composition, db_name="mp", **kwargs)
    return get_average_volume_from_mp_api(composition, **kwargs)


def _get_chem_env_key_from_composition(
    composition: Composition, ignore_oxi_states: bool = True
) -> str:
    """
    Get chemical environment as a string for ICSD avg volume determination.

    Parameters
    ----------
    composition : .Composition
        Structure composition
    ignore_oxi_states : bool = True
        Whether to ignore oxidation states assigned to sites in the structure,
        both in the input composition and ICSD structures.

        Note that 0+ / 0- oxidation states are treated identically even
        when ignore_oxi_states = False.

    Returns
    -------
    Chemical environment returned as a dunder-separated string,
    such as "Ag+__Cu2+__N5+__O2-"
    """
    comp = composition
    if ignore_oxi_states:
        comp = comp.remove_charges()
    chem_env = "__".join(sorted(set(comp.as_dict())))
    for char in ["+", "-"]:
        chem_env = chem_env.replace(f"0{char}", "")
    return chem_env


def get_average_volume_from_database(
    composition: Composition,
    avg_vols: pd.DataFrame,
    ignore_oxi_states: bool = True,
) -> float:
    """
    Get average volume for a chemical environment from ICSD data.

    The ICSD data is for "reasonable", ordered, experimental inorganic solids.

    Parameters
    ----------
    composition : .Composition
        Structure composition
    avg_vols : pandas .DataFrame
        Chemical environment data for a given database.
        Should have the following columns:
            "chem_env", "avg_vol", "count", "with_oxi"
    ignore_oxi_states : bool = True
        Whether to ignore oxidation states assigned to sites in the structure,
        both in the input composition and ICSD structures.

        Note that 0+ / 0- oxidation states are treated identically even
        when ignore_oxi_states = False.

    Returns
    -------
    Average volume as a float
    """
    from itertools import combinations

    def get_entry_from_dict(chem_env: str) -> dict | None:
        data = avg_vols[avg_vols["chem_env"] == chem_env]
        data = data[
            (
                data["with_oxi"]
                if (not ignore_oxi_states and len(data[data["with_oxi"]]) > 0)
                else ~data["with_oxi"]
            )
        ]
        if len(data) > 0:
            return {k: data[k].squeeze() for k in ("avg_vol", "count")}
        return None

    full_chem_env_key = _get_chem_env_key_from_composition(
        composition, ignore_oxi_states=ignore_oxi_states
    )
    if (avg_vol := get_entry_from_dict(full_chem_env_key)) is not None:
        return avg_vol["avg_vol"]

    vols = []
    counts = 0
    for ielt in range(2, len(composition)):
        for combo in combinations(composition, ielt):
            chem_env_key = _get_chem_env_key_from_composition(
                Composition(dict.fromkeys(combo, 1)),
                ignore_oxi_states=ignore_oxi_states,
            )

            if (avg_vol := get_entry_from_dict(chem_env_key)) is not None:
                vols.append(avg_vol["avg_vol"] * avg_vol["count"])
                counts += avg_vol["count"]

    # Fallback, relative weight of monatomic volumes
    if counts == 0:
        by_comp = {ele: get_entry_from_dict(ele.name) for ele in composition.elements}
        if any(v is None for v in by_comp.values()):
            raise ValueError(
                "No unary data for "
                f"{', '.join(str(k) for k, v in by_comp.items() if v is None)}"
            )
        return (
            sum(coeff * by_comp[ele]["avg_vol"] for ele, coeff in composition.items())
            / composition.num_atoms
        )

    return sum(vols) / counts


def get_random_packed_structure(
    composition: Composition | str,
    polyhedras: list[Molecule] | None = None,
    target_atoms: int = 100,
    vol_multiply: float = 1.0,
    tol: float = 2.0,
    return_as_job: bool = False,
    vol_per_atom_source: float | str = "mp",
    pbc: bool = False,
    db_kwargs: dict | None = None,
    packmol_seed: int = 1,
    packmol_output_dir: str | Path | None = None,
) -> Structure | Job:
    """
    Generate a random packed structure with a target number of atoms and/or polyhedras.
    Polyhedra packing aims to follow Zachariasen's random network theory.

    Designed to make amorphous/glassy structures.
    Defaults to using cached MP data.

    Parameters
    ----------
    composition : Composition | str
        The composition of the target structure.
    polyhedras: list[Molecule] | None = None
        List of polyhedras to include in the target structure.
    target_atoms : int
        The target number of atoms in the structure.
    vol_multiply : float
        The factor to multiply the structure volume by.
    tol : float
        The tolerance to apply to the box size.
    return_as_job : bool
        Whether to return the structure as a jobflow job object.
    vol_per_atom_source : float | str
        If float - the volume per atom used to generate lattice size
        If str - "mp" to use the Materials Project API to estimate volume per atom.
        If str - "icsd" to use the ICSD database to estimate volume per atom.
    pbc : bool = False,
        Perserve periodic boundary condition effects when generating structure.
        Recommend set tol = 0 if pbc = True.
        Default to False; implemented in packmol > 20.15.0 only.
    db_kwargs : dict | None = None
        kwargs to pass to the volume-determining function.
    packmol_seed : int
        The seed to use for the packmol random number generator.
    packmol_output_dir : str | Path | None
        The directory to output the packmol files to. If None, a
        temporary directory is used and will be removed after.

    Returns
    -------
    Structure | Job
        The random packed structure.
    """
    if return_as_job:
        return Job(
            get_random_packed_structure,
            function_kwargs={
                "composition": composition,
                "polyhedras": polyhedras,
                "target_atoms": target_atoms,
                "vol_multiply": vol_multiply,
                "tol": tol,
                "return_as_job": False,
                "vol_per_atom_source": vol_per_atom_source,
                "pbc": False,
                "packmol_seed": packmol_seed,
            },
        )
    if isinstance(composition, str | dict):
        composition = Composition(composition)

    struct_db = (
        vol_per_atom_source.lower() if isinstance(vol_per_atom_source, str) else None
    )
    db_kwargs = db_kwargs or ({"use_cached": True} if struct_db == "mp" else {})

    if isinstance(vol_per_atom_source, float | int):
        vol_per_atom = vol_per_atom_source

    elif struct_db == "mp":
        vol_per_atom = get_average_volume_from_mp(composition, **db_kwargs)

    elif struct_db == "icsd":
        vol_per_atom = get_average_volume_from_db_cached(
            composition, db_name="icsd", **db_kwargs
        )

    else:
        raise ValueError(f"Unknown volume per atom source: {vol_per_atom_source}.")

    formula, _ = composition.get_integer_formula_and_factor()
    integer_composition = Composition(formula)
    full_cell_composition = integer_composition * np.ceil(
        target_atoms / integer_composition.num_atoms
    )

    # if polyhedras - finds the number of polyhedras that can pack into desired stiochemtry
    # remainder is filled with single atoms to match composition
    if polyhedras:
        polyhedra_total_comp = sum(
            [poly.composition for poly in polyhedras], start=Composition()
        )
        num_polyhedra_sites = int(
            min(
                [
                    full_cell_composition.as_dict()[el]
                    / polyhedra_total_comp.as_dict()[el]
                    for el in full_cell_composition.as_dict()
                    if el in polyhedra_total_comp.as_dict()
                ]
            )
        )
        atomic_site_composition = full_cell_composition - (
            polyhedra_total_comp * num_polyhedra_sites
        )
    else:
        atomic_site_composition = full_cell_composition

    supercell_composition = {
        str(el): int(atomic_site_composition.element_composition.get(el))
        for el in full_cell_composition
        if int(atomic_site_composition.element_composition.get(el)) != 0
    }

    with TemporaryDirectory() as tmpdir:
        molecules = []
        if polyhedras:
            for poly in polyhedras:
                xyz_file = f"{tmpdir}/{poly.composition.reduced_formula}.xyz"
                poly.to(xyz_file)
                molecules.append(
                    {
                        "name": poly.composition.reduced_formula,
                        "number": num_polyhedra_sites,
                        "coords": xyz_file,
                    }
                )

        for element, num_sites in supercell_composition.items():
            xyz_file = f"{tmpdir}/{element}.xyz"
            with open(xyz_file, "w+") as f:
                f.write("1\ncomment\n" + element + " 0.0 0.0 0.0\n")
            molecules.append({"name": element, "number": num_sites, "coords": xyz_file})

        box_scale = (vol_per_atom * full_cell_composition.num_atoms * vol_multiply) ** (
            1 / 3
        )
        box_lower_bound = tol / 2
        box_upper_bound = box_scale - tol / 2

        box_size = 3 * [box_lower_bound] + 3 * [box_upper_bound]

        # modify this to include additional packmol params
        # see PackmolBoxGen class in pymatgen.io.packmol for more details
        packmol_additional_params = (
            {"pbc": [" ".join(map(str, box_size)) + "\n"]} if pbc else {}
        )

        packmol_set = PackmolBoxGen(
            seed=packmol_seed,
            control_params=packmol_additional_params,
        ).get_input_set(molecules=molecules, box=box_size)
        packmol_output_dir = str(packmol_output_dir or tmpdir)
        packmol_set.write_input(directory=packmol_output_dir)
        packmol_set.run(path=packmol_output_dir)

        mol = Molecule.from_file(f"{packmol_output_dir}/packmol_out.xyz")

    return Structure(
        [[box_scale if i == j else 0.0 for j in range(3)] for i in range(3)],
        species=mol.species,
        coords=mol.cart_coords,
        coords_are_cartesian=True,
    )
