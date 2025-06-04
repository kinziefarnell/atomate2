"""Define code agnostic MPMorph flows.

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

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
from jobflow import Flow, Maker, Response, job

from atomate2.common.jobs.eos import MPMorphPVPostProcess, _apply_strain_to_structure

from atomate2.common.jobs.mpmorph import extract_trajectory_frames

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from jobflow import Job
    from pymatgen.core import Structure
    from typing_extensions import Self

    from atomate2.common.jobs.eos import EOSPostProcessor


@dataclass
class EquilibriumVolumeMaker(Maker):
    """
    Equilibrate structure using NVT + EOS fitting.

    Parameters
    ----------
    name : str = "Equilibrium Volume Maker"
        Name of the flow
    md_maker : Maker
        Maker to perform NVT MD runs
    postprocessor : atomate2.common.jobs.eos.EOSPostProcessor
        Postprocessing step to fit the EOS
    initial_strain : float | tuple[float,float] = 0.2
        Initial percentage linear strain to the apply to the structure
    min_strain : float, default = 0.5
        Minimum absolute percentage linear strain to apply to the structure
    max_attempts : int | None = 20
        Number of times to continue attempting to equilibrate the structure.
        If None, the workflow will not terminate if an equilibrated structure
        cannot be determined.
    """

    md_maker: Maker
    name: str = "Equilibrium Volume Maker"
    postprocessor: EOSPostProcessor = field(default_factory=MPMorphPVPostProcess)
    initial_strain: float | tuple[float, float] = 0.2
    min_strain: float = 0.5
    max_attempts: int | None = 20

    def __post_init__(self) -> None:
        """Ensure required class attributes are set."""
        if self.md_maker is None:
            raise ValueError("You must specify `md_maker` to use this flow.")

    @job
    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
        working_outputs: dict[str, Any] | None = None,
    ) -> Flow:
        """
        Run an NVT+EOS equilibration flow.

        Parameters
        ----------
        structure : Structure
            structure to equilibrate
        prev_dir : str | Path | None (default)
            path to copy files from
        working_outputs : dict or None
            contains the outputs of the flow as it recursively updates

        Returns
        -------
        .Flow, an MPMorph flow
        """
        if working_outputs is None:
            if isinstance(self.initial_strain, float | int):
                self.initial_strain = (
                    -abs(self.initial_strain),
                    abs(self.initial_strain),
                )
            elif (
                not isinstance(self.initial_strain, tuple | list | np.array)
                or len(self.initial_strain) != 2
            ):
                raise ValueError(
                    "`initial_strain` should either be a float, to set "
                    "a symmetric linear strain of ± `initial_strain`, or a two-element "
                    "tuple / list to explicitly set linear strain values, "
                    f"not {self.initial_strain}."
                )

            linear_strain = np.linspace(
                *self.initial_strain, self.postprocessor.min_data_points
            )
            working_outputs = {
                "relax": {
                    key: [] for key in ("energies", "volume", "stress", "pressure")
                }
            }

        else:
            # Fit EOS to running list of energies and volumes
            self.postprocessor.fit(working_outputs)
            working_outputs = dict(self.postprocessor.results)
            flow_output = {"working_outputs": working_outputs.copy(), "structure": None}
            for k in ("pressure", "energy"):
                working_outputs["relax"].pop(k, None)

            # Stop flow here if EOS cannot be fit
            if (v0 := working_outputs.get("V0")) is None:
                return Response(output=flow_output, stop_children=True)

            # Check if equilibrium volume is in range of attempted volumes
            v0_in_range = (
                (vmin := working_outputs.get("Vmin"))
                <= v0
                <= (vmax := working_outputs.get("Vmax"))
            )

            # Check if maximum number of refinement NVT runs is set,
            # and if so, if that limit has been reached
            max_attempts_reached = len(working_outputs["relax"]["volume"]) >= (
                (self.max_attempts or np.inf) + self.postprocessor.min_data_points
            )

            # Successful fit: return structure at estimated equilibrium volume
            if v0_in_range or max_attempts_reached:
                flow_output["structure"] = structure.copy()
                flow_output["structure"].scale_lattice(v0)  # type: ignore[attr-defined]
                return flow_output

            # Else, if the extrapolated equilibrium volume is outside the range of
            # fitted volumes, scale appropriately
            v_ref = vmax if v0 > vmax else vmin
            eps_0 = (v0 / v_ref) ** (1.0 / 3.0) - 1.0
            linear_strain = [np.sign(eps_0) * (abs(eps_0) + self.min_strain)]

        deformation_matrices = [np.eye(3) * (1.0 + eps) for eps in linear_strain]
        deformed_structures = _apply_strain_to_structure(
            structure, deformation_matrices
        )

        eos_jobs = []
        for index in range(len(deformation_matrices)):
            md_job = self.md_maker.make(
                structure=deformed_structures[index].final_structure,
                prev_dir=None,
            )
            relaxed_vol = len(working_outputs["relax"]["volume"])
            md_job.name = f"Equil Vol {md_job.name} {relaxed_vol + 1}"

            postprocess_job = extract_trajectory_frames(md_job.output)
            postprocess_job.name = f" process_traj_frames {relaxed_vol + 1}"

            working_outputs["relax"]["energies"].append(postprocess_job.output.energy)
            working_outputs["relax"]["volume"].append(md_job.output.structure.volume)
            working_outputs["relax"]["stress"].append(postprocess_job.output.stress)
            working_outputs["relax"]["pressure"].append(postprocess_job.output.pressure)
            eos_jobs.append(md_job)
            eos_jobs.append(postprocess_job)

        recursive = self.make(
            structure=structure,
            prev_dir=None,
            working_outputs=working_outputs,
        )

        new_eos_flow = Flow([*eos_jobs, recursive], output=recursive.output)

        return Response(replace=new_eos_flow, output=recursive.output)


# insert convergence maker here
@dataclass
class ConvergenceMDMaker():# put in appropriate args):
    # Add in appropriate notes on what this maker does
    # should take a Maker object that you will use to define the md runs
    """
    Converge structure using NVT + convergence runs

    Parameters
    name : str = "Convergence Maker"
        Name of the flow
    md_maker: Maker
        Maker to perform NVT MD runs
    rescale parameters: float? # TODO: figure out what should be included here
        Parameter to change volume by between pressure runs
    convergence parameters: dict
        parameters to identify convergence
    TODO: whatever else to add here, i.e. max energy runs
    """

    md_maker: Maker
    name: str = "Convergence Maker"
    rescale_params: float = 0.0000005 # "beta" parameter in old mpmorph, 
    convergence params: dict = {"density": 0.0005, "ionic": 0.0005} # TODO - check if correct val for density
    max_energy_runs: float = 3 

    # define post_init method
    # I'm not really sure why the post_init is there, but the other classes seem to have it
    def __post_init__(self) -> None:
        """Ensure required class attributes are set. """
        if self.md_maker is None:
            raise ValueError("You must specify 'md_maker' to use this flow.")

    # define make function
    @job
    make(
            self,
            structure,
            prev_dir,
            working_outputs
            )
    rescale_struct = False
    if working_outputs is None:
        # define working outputs
        working_outputs = {key: [] for key in ("pressure", "ionic")}
    else:
        # do check for pressure
        if working_outputs["pressure"] < pressure: # TODO add args somewhere that define pressure and ionic required for convergence
            converged_pressure = True
        else:
            converged_pressure = False
            rescale_struct = True

        # check for energy differences # TODO: check if you should be rescaling structure when energy is unconverged, no?
        if working_outputs["ionic"] < ionic:
            converged_ionic = True
        else:
            converged_ionic = False
        
        # TODO: third check in old mpmorph for "kinetic", idk

        if converged_pressure and converged_energy:
            # no more convergence runs


    #set up md run
    if rescale_struct:
        # need to rescale structure
        # set up some kind of deformation matrix with rescale params
        # apply that
        structure = deformed_structure
    # else, will just rerun same structure for longer
    self.md_maker.make(structure = structure, prev_dir = None)
    md_job.name = f"convergence run" # TODO: add something to indicate if pressure/ionic run

    post_process = extract_trajectory_frames(md_job.output, check_convergence = True)
    post_process.name = f"postprocess convergence run"
    # make extract trajectory frames to take md maker output

    working_outputs["pressure"] = post_process.output.pressure
    working_outputs["ionic"] = post_process.output.ionic

    # set up recursive call for function
    recursive = self.make(
            structure,
            prev_dir = None,
            working_outputs = working_outputs,
            )

    new_eos_flow = Flow([md_job, post_process, recursive], output = recursive.output)

    # return response which lists flow for md-jobs, convergence-check, and recursive make call
    return Response(replace = new_eos_flow, output = recursive.output)


    # you can define the make function with a job decorator and it will implicitly become a job object
    # inputs to make: structure, working outputs

        # make job to run MD for 1000 steps

        # make job that does checks on the MD - could add this in common/jobs/mpmorph.py, yes?

        # well, let's at least write the pseudocode for that job and can put it in the appropriate place as we figure out

        # take MD pressure - is it less than 5<
        # if less than 5, then that's good
        # if not less than 5, we need more convergence runs


        # does this need to be a job or is it possible to do the processing here?

        # because I notice that EquilibriumVolumeMaker has lots of processing and it's like a job in the flow?

        # some aspect of this should be recursive I think




@dataclass
class MPMorphMDMaker(Maker, metaclass=ABCMeta):
    """Base MPMorph flow for amorphous solid equilibration.

    This flow uses NVT molecular dynamics to:
    (1 - optional) Determine the equilibrium volume of an amorphous
        structure via EOS fit.
    (2 - optional) Quench the equilibrium volume structure from a higher
        temperature down to a lower desired "production" temperature.
    (3) Run a production, longer-time MD run in NVT.
        The production run can be broken up into smaller steps to
        ensure the simulation does not hit wall time limits.

    Check atomate2.vasp.flows.mpmorph for MPMorphVaspMDMaker

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    equilibrium_volume_maker : EquilibriumVolumeMaker
        MDMaker to generate the equilibrium volumer searcher
    production_md_maker : Maker
        MDMaker to generate the production run(s)
    quench_maker :  SlowQuenchMaker or FastQuenchMaker or None
        SlowQuenchMaker - MDMaker that quenches structure from
            high to low temperature
        FastQuenchMaker - DoubleRelaxMaker + Static that
            "quenches" structure at 0K
    """

    production_md_maker: Maker
    name: str = "Base MPMorph MD"
    equilibrium_volume_maker: Maker | None = None
    quench_maker: FastQuenchMaker | SlowQuenchMaker | None = None

    def __post_init__(self) -> None:
        """Ensure required class attributes are set."""
        if self.production_md_maker is None:
            raise ValueError("You must set `production_md_maker` to use this flow.")

    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
    ) -> Flow:
        """
        Create an MPMorph equilibration workflow.

        Converegence and quench steps are optional, and may be used
        to equilibrate the cell volume (useful for high temperature
        production runs of structures extracted from Materials Project)
        and to quench the structure from high to low temperature
        (e.g. amorphous structures), respectively.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.

        Returns
        -------
        Flow
            A flow containing series of molecular dynamics run (and relax+static).
        """
        flow_jobs = []

        if self.equilibrium_volume_maker is not None:
            convergence_flow = self.equilibrium_volume_maker.make(
                structure, prev_dir=prev_dir
            )
            flow_jobs.append(convergence_flow)

            # convergence_flow only outputs a structure
            structure = convergence_flow.output["structure"]

        self.production_md_maker.name = self.name + " production run"
        production_run = self.production_md_maker.make(
            structure=structure, prev_dir=prev_dir
        )
        flow_jobs.append(production_run)

        if self.quench_maker:
            quench_flow = self.quench_maker.make(
                structure=production_run.output.structure,
                prev_dir=production_run.output.dir_name,
            )
            flow_jobs += [quench_flow]

        return Flow(
            flow_jobs,
            output=production_run.output,
            name=self.name,
        )

    @classmethod
    @abstractmethod
    def from_temperature_and_steps(
        cls,
        temperature: float,
        n_steps_convergence: int = 5000,
        n_steps_production: int = 10000,
        end_temp: float | None = None,
        md_maker: Maker = None,
        quench_maker: FastQuenchMaker | SlowQuenchMaker | None = None,
    ) -> Self:
        """
        Create an MPMorph flow from a temperature and number of steps.

        This is a convenience class constructor. The user need only
        input the desired temperature and steps for convergence / production
        MD runs.

        Parameters
        ----------
        temperature : float
            The (starting) temperature
        n_steps_convergence : int = 5000
            The number of steps used in MD runs for equilibrating structures.
        n_steps_production : int = 10000
            The number of steps used in MD production runs.
        end_temp : float or None
            If a float, the temperature to ramp down to in the production run.
            If None (default), set to `temperature`.
        base_md_maker : Maker
            The Maker used to start MD runs.
        quench_maker : SlowQuenchMaker or FastQuenchMaker or None
            SlowQuenchMaker - MDMaker that quenches structure from
                high to low temperature
            FastQuenchMaker - DoubleRelaxMaker + Static that
                "quenches" structure at 0K
        """
        raise NotImplementedError


@dataclass
class FastQuenchMaker(Maker):
    """Fast quench flow from high temperature to 0K structures.

    Quenches a provided structure with a single (or double)
    relaxation and a static calculation at 0K.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker :  Maker
        Relax Maker
    relax_maker2 :  Maker or None
        Relax Maker for a second relaxation; useful for tighter convergence
    static_maker : Maker
        Static Maker
    """

    relax_maker: Maker
    static_maker: Maker
    name: str = "fast quench"
    relax_maker2: Maker | None = None

    def __post_init__(self) -> None:
        """Ensure required class attributes are set."""
        for attr in ("relax_maker", "static_maker"):
            if getattr(self, attr, None) is None:
                raise ValueError(
                    f"You must specify {attr} to use this flow. "
                    "Only `relax_maker2` is optional."
                )

    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
    ) -> Flow:
        """
        Create a fast quench flow with relax and static makers.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.

        Returns
        -------
        Flow
            A flow containing series of relax and static runs.
        """
        jobs: list[Job] = []

        relax1 = self.relax_maker.make(structure, prev_dir=prev_dir)
        jobs += [relax1]
        structure = relax1.output.structure
        prev_dir = relax1.output.dir_name

        if self.relax_maker2 is not None:
            relax1.name += " 1"
            relax2 = self.relax_maker2.make(structure, prev_dir=prev_dir)
            relax2.name += " 2"
            jobs += [relax2]
            structure = relax2.output.structure
            prev_dir = relax2.output.dir_name

        static = self.static_maker.make(structure, prev_dir=prev_dir)
        jobs += [static]
        return Flow(
            jobs,
            output=static.output,
            name=self.name,
        )


@dataclass
class SlowQuenchMaker(Maker, metaclass=ABCMeta):
    """Slow quench from high to low temperature structures.

    Quenches a provided structure with a molecular dynamics
    run from a desired high temperature to a desired low temperature.
    Flow creates a series of MD runs that holds at a certain temperature
    and initiates the following MD run at a lower temperature (step-wise
    temperature MD runs).

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    md_maker :  Maker | None = None
        Can only be an MDMaker or ForceFieldMDMaker.
        Defaults to None. If None, will not work. #WORK IN PROGRESS.
    quench_start_temperature : float = 3000
        Starting temperature for quench; default 3000K
    quench_end_temperature : float = 500
        Ending temperature for quench; default 500K
    quench_temperature_step : float = 500
        Temperature step for quench; default 500K drop
    quench_n_steps : int = 1000
        Number of steps for quench; default 1000 steps
    descent_method : str = "stepwise"
        Descent method for quench; default "stepwise".
        Others available: "linear with hold"
    """

    md_maker: Maker
    name: str = "slow quench"
    quench_start_temperature: float = 3000
    quench_end_temperature: float = 500
    quench_temperature_step: float = 500
    quench_n_steps: int = 1000
    descent_method: Literal["stepwise", "linear with hold"] = "stepwise"

    def __post_init__(self) -> None:
        """Ensure required class attributes are set."""
        if self.md_maker is None:
            raise ValueError("You must specify `md_maker` to use this flow.")

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Flow:
        """
        Create a slow quench flow with md maker.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.

        Returns
        -------
        Flow
            A flow containing series of relax and static runs.
        """
        md_jobs: list[Job] = []
        for temp in np.arange(
            self.quench_start_temperature,
            self.quench_end_temperature,
            -self.quench_temperature_step,
        ):
            prev_dir = (
                None
                if temp == self.quench_start_temperature
                else md_jobs[-1].output.dir_name
            )
            if self.descent_method == "stepwise":
                md_job = self.call_md_maker(
                    structure=structure,
                    temp=temp,
                    prev_dir=prev_dir,
                )

            elif (
                self.descent_method == "linear with hold"
            ):  # TODO: Work in Progress; needs testing
                md_job_linear = self.call_md_maker(
                    structure=structure,
                    temp=[temp, temp - self.quench_temperature_step],  # type: ignore[arg-type]
                    prev_dir=prev_dir,
                )

                md_job = self.call_md_maker(
                    structure=md_job_linear.output.structure,
                    temp=temp - self.quench_temperature_step,
                    prev_dir=md_job_linear.output.dir_name,
                )

                md_jobs.append(md_job_linear)

            md_jobs.append(md_job)

            structure = md_job.output.structure

        return Flow(
            md_jobs,
            output=md_jobs[-1].output,
            name=self.name,
        )

    @abstractmethod
    def call_md_maker(
        self,
        structure: Structure,
        temp: float,
        prev_dir: str | Path | None = None,
    ) -> Flow | Job:
        """Call MD maker for slow quench.

        To be implemented in subclasses.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object.
        temp : float
            The temperature in Kelvin.
        prev_dir : str or Path or None
            A previous calculation directory to copy output files from.

        Returns
        -------
        A slow quench .Flow or .Job
        """
        raise NotImplementedError
