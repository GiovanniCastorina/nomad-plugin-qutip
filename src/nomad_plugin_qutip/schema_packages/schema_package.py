from typing import (
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import (
        EntryArchive,
    )
    from structlog.stdlib import (
        BoundLogger,
    )


import numpy as np
from nomad.config import config
from nomad.datamodel.data import Schema,ArchiveSection
from nomad.datamodel.metainfo.annotations import ELNAnnotation, ELNComponentEnum, SectionProperties
from nomad.metainfo import MEnum, Quantity, SchemaPackage, SubSection, Section
from nomad_simulations.schema_packages.general import Simulation
from nomad_simulations.schema_packages.model_system import ModelSystem
from nomad_simulations.schema_packages.model_method import ModelMethod
from nomad_simulations.schema_packages.physical_property import PhysicalProperty
from nomad_simulations.schema_packages.variables import Variables

from nomad.datamodel.metainfo.plot import PlotSection, PlotlyFigure
import plotly.express as px

configuration = config.get_plugin_entry_point(
    'nomad_plugin_qutip.schema_packages:schema_package_entry_point'
)

m_package = SchemaPackage()

class QuantumObject(ArchiveSection):
    """
    A framework-agnostic data container for storing states/operators/superoperators.

    The key fields track:
      - dimensional structure (`dims`, `shape`),
      - type of object (`ket`, `bra`, `oper`, `super`, `dm`, etc.),
      - numerical storage format (e.g., 'Dense', 'csr'),
      - Hermiticity flag,
      - the underlying data (complex matrix or vector).
    """

    # !!! dims might be a list of lists describing the Hilbert space.
    dims = Quantity(
        type=np.int32,
        shape=['*', '*'],
        description=(
            """Tensor-product dimensions.
            For example, a single qubit ket might be [[2], [1]],
            or a 2-qubit system [[2,2],[1,1]]."""
        ),
    )

    shape = Quantity(
        type=np.int32,
        shape=['*'],
        description=(
            """Shape of the object's matrix or vector representation.
            For a 2-level ket, this might be [2,1].
            For a 2x2 operator, [2,2]."""
        ),
    )

    type = Quantity(
        type=MEnum('ket', 'bra', 'oper', 'super', 'dm'),
        description=(
            """Logical type of the quantum object:
               - 'ket' for state vectors,
               - 'bra' for dual vectors,
               - 'oper' for operators,
               - 'super' for superoperators,
               - 'dm' for density matrices."""
        ),
    )

    storage_format = Quantity(
        type=str,
        description=(
            """String label for the numerical storage format
            (e.g., 'Dense', 'csr', 'ELL')."""
        ),
    )

    is_hermitian = Quantity(
        type=bool,
        description="""True if this object is Hermitian (self-adjoint).
        (only relevant for 'oper'/'dm').""",
    )

    data = Quantity(
        type=np.complex128,
        shape=['*', '*'],
        description=(
            """The underlying array or matrix representing this quantum object.
            TODO: sparse matrix representations."""
        ),
    )

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)

        # If data is present, do shape checks
        if self.data is not None:
            # Convert data.shape to a np.array of type int
            data_shape = np.array(self.data.shape, dtype=np.int32)  # e.g., [2, 2]

            # If shape is None or has zero elements, fill from data
            if self.shape is None or self.shape.size == 0:
                self.shape = data_shape
                logger.info(f"Filling 'shape' from data: {self.shape}")
            # Compare with data_shape using np.array_equal
            elif not np.array_equal(self.shape, data_shape):
                logger.warning(
                    f'Inconsistent shape {self.shape.tolist()} \
                    vs data {data_shape.tolist()}; '
                    'using data shape.'
                )
                self.shape = data_shape

        # If shape is now set but dims is not, try to infer
        # (Check shape.size and shape[0], shape[1], etc. carefully)
        EXPECTED_SHAPE_SIZE = 2

        if (
            self.shape is not None
            and self.shape.size == EXPECTED_SHAPE_SIZE  # e.g. shape=[m, n]
            and (self.dims is None or self.dims.size == 0)
        ):
            m, n = self.shape
            if self.type == 'ket':
                self.dims = np.array([[m], [1]], dtype=np.int32)
                logger.info(f"Inferred 'dims' for ket: {self.dims}")
            elif self.type == 'bra':
                self.dims = np.array([[1], [n]], dtype=np.int32)
            elif self.type in ('oper', 'dm'):
                self.dims = np.array([[m], [n]], dtype=np.int32)
            elif self.type == 'super':
                # Could guess dims differently, but here's a simple approach
                self.dims = np.array([[m], [n]], dtype=np.int32)

        # If type is 'oper' or 'dm' and is_hermitian not set, check from data
        if (
            self.type in ('oper', 'dm')
            and self.data is not None
            and self.is_hermitian is None
        ):
            # Compare data and data.conjugate().T
            if np.allclose(self.data, self.data.conjugate().T):
                self.is_hermitian = True
            else:
                self.is_hermitian = False
            logger.info(f"Set 'is_hermitian' to {self.is_hermitian} based on data.")


class QuantumSystem(ModelSystem):
    """
    A specialized 'model system' for quantum information simulations,
    e.g., representing a set of qubits, spins, or multi-level systems.
    Inherits from ModelSystem to remain compatible with existing archiving.
    """

    name = Quantity(type=str, description='Optional label for this quantum system.')


class QuantumOperator(ArchiveSection):
    """
    A container for quantum operators, referencing a Qobj or storing
    additional operator info. Not strictly required to inherit from ModelMethod.
    """

    name = Quantity(
        type=str, description="Label for the operator (e.g. 'Hamiltonian', 'sigmaz')."
    )

    quantum_object = SubSection(
        sub_section=QuantumObject.m_def,
        repeats=True,
        description='The underlying quantum object (operator form).',
    )


class QuantumState(ArchiveSection):
    """
    A container for wavefunction or density-matrix states.
    Maybe add matrix-states
    """

    label = Quantity(type=str, description='Optional label for this quantum state.')

    quantum_object = SubSection(
        sub_section=QuantumObject.m_def,
        repeats=False,
        description='The underlying quantum object representing this state.',
    )


class HamiltonianParameter(ArchiveSection):
    """Stores a single named parameter used in the Hamiltonian formula."""
    m_def = Section(
        a_eln={'properties': SectionProperties(
                label='Hamiltonian Parameter',
                order=['name', 'value', 'unit']
            )}
    )
    name = Quantity(
        type=str,
        description="Name of the parameter.",
        a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity, label='Parameter Name')
    )
    value = Quantity(
        type=np.float64,
        description="Numerical value of the parameter.",
        a_eln=ELNAnnotation(component=ELNComponentEnum.NumberEditQuantity)
    )
    unit = Quantity(
        type=str,
        description="Unit of the parameter (e.g.'eV'). Optional.",
        a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity)
    )


class ModelHamiltonian(ModelMethod):
    """
    Describes the model Hamiltonian using a formula and parameters.
    """
    m_def = Section(
        a_eln={'properties': SectionProperties(
                label='Model Hamiltonian Description'
            )}
    )

    formula = Quantity(
        type=str,
        description="""
        The mathematical formula representing the Hamiltonian, preferably using
        LaTeX syntax.
        """,
        a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity, props={'render_value': 'markdown'})
    )

    parameters = SubSection(
        section_def=HamiltonianParameter,
        repeats=True,
        description="List of parameters used in the Hamiltonian formula with their values.",
    )


##############################
# New PhysicalProperty Sections
##############################


class SolverStats(ArchiveSection):
    m_def = Section(
        a_eln=dict(lane_width='400px'),
        description="Statistics related to the numerical solver execution."
    )
    solver_name = Quantity(
        type=str,
        description="Name of the solver used (e.g., 'sesolve', 'mesolve')."
    )
    method = Quantity(
        type=str,
        description="Specific algorithm or method used by the solver (e.g., 'scipy zvode adams')."
    )
    solver_description = Quantity(
        type=str,
        description="Additional description provided by the solver (e.g., 'Schrodinger Evolution')."
    )
    initialization_time = Quantity(
        type=np.float64,
        unit='second',
        description="Time taken for solver initialization."
    )
    preparation_time = Quantity(
        type=np.float64,
        unit='second',
        description="Time taken for solver preparation steps."
    )
    run_time = Quantity(
        type=np.float64,
        unit='second',
        description="Time taken for the main solver execution run."
    )
    time_interval_start = Quantity(
        type=np.float64,
        description="Start of the simulation time interval."
    )
    time_interval_end = Quantity(
        type=np.float64,
        description="End of the simulation time interval."
    )
    n_steps = Quantity(
        type=np.int32,
        description="Number of time steps in the simulation interval."
    )
    n_expectation_operators = Quantity(
        type=np.int32,
        description="Number of expectation value operators (e_ops) tracked."
    )
    final_state_saved = Quantity(
        type=bool,
        description="Flag indicating if the final state of the evolution was saved in the result object."
    )

class EigenvaluesInVariable(PhysicalProperty, PlotSection):
    """
    A physical property representing energy eigenvalues as a function of an external variable.
    Expected `value` shape: [n_points, n_eigenvalues]
    """
    m_def = Section(
        a_eln=dict(properties=SectionProperties(label='Eigenvalues vs Variable')),)
    value = Quantity(
        type=np.float64,
        shape=['*', '*'],
        description="Matrix of eigenvalues. Rows correspond to variable points and columns to eigenvalues."
    )

    def plot_eigenvalues(self):
        # Ensure value is 2D
        if self.value is None or self.value.size == 0:
             self.logger.warning("No eigenvalue data available to plot.")
             return None
        if self.value.ndim != 2:
            raise ValueError("Value must be a 2D array (n_points, n_eigenvalues) for eigenvalue plotting.")
        # Retrieve the x-axis from the first variable if available
        x = None
        x_label = 'Index'
        if self.variables:
            var1 = self.variables[0]
            if hasattr(var1, 'get_values') and var1.value is not None:
                 x = var1.get_values()
                 x_label = f"{var1.name or 'Variable'}"
                 if var1.unit:
                     x_label += f" ({var1.unit})"
            elif isinstance(var1.value, (np.ndarray, list)) and len(var1.value) == self.value.shape[0]:
                 x = var1.value # Fallback
                 x_label = f"{var1.name or 'Variable'}"
                 if var1.unit:
                     x_label += f" ({var1.unit})"

        # Generate a line plot where each eigenvalue is a separate curve
        plot_title = f"{self.name or 'Eigenvalues'} vs. {x_label.split('(')[0].strip()}" # Titolo più pulito
        # fig = px.line(x=x, y=self.value, labels={'x': x_label, 'y': 'Eigenvalue'},
        #               title=plot_title)
        fig = px.line(
            x=x,
            y=self.value.T,  # ← transpose so each column becomes its own trace
            labels={'x': x_label, 'y': 'Eigenvalue'},
            title=plot_title
        )
        # More eigenvalues labels
        if self.value.shape[1] > 1:
             fig.update_layout(showlegend=True)
             for i, trace in enumerate(fig.data):
                 trace.name = f'E_{i}'


        # Append the figure to the PlotSection's figures list
        self.figures.append(PlotlyFigure(label='Eigenvalues Plot', figure=fig.to_plotly_json()))
        return fig

    def normalize(self, archive, logger):
        super().normalize(archive, logger)
        try:
            self.plot_eigenvalues()
        except Exception as e:
            logger.error(f"Failed to plot eigenvalues for '{self.name}': {e}")



class TimeEvolutionProperty(PhysicalProperty, PlotSection):
    """
    A physical property representing the time evolution of a quantum state.
    Expected `value` shape: [n_time_steps, state_dimension]
    """
    m_def = Section(a_eln=dict(properties=SectionProperties(label='Time Evolution Results')),
    )

    value = Quantity(
        type=np.float64,
        shape=['*', '*'],
        description=(
            "Expectation values vs time. Rows correspond to time steps, "
            "columns correspond to different expectation operators (e_ops)."
        )
    )

    e_ops_names = Quantity(
        type=str,
        shape=['*'], # Lista of strings
        description="Names/labels of the expectation value operators (e_ops)."
    )

    solver_statistics = SubSection(
        section_def=SolverStats,
        description="Statistics related to the solver execution for this time evolution."
    )

    def plot_expectation_values(self):
        if self.value is None or self.value.size == 0:
             self.logger.warning("No expectation value data available to plot.")
             return None
        if self.value.ndim != 2:
            #Could be one expectation value
             if self.value.ndim == 1:
                 self.value = self.value.reshape(-1, 1)
             else:
                 raise ValueError("Value must be a 1D or 2D array (n_time_steps, [n_e_ops]) for plotting.")

        time = None
        t_label = 'Index'
        if self.variables:
            var_time = self.variables[0] # Assume time is first variable
            if hasattr(var_time, 'get_values') and var_time.value is not None:
                 time = var_time.get_values()
                 t_label = f"{var_time.name or 'Time'}"
                 if var_time.unit:
                      t_label += f" ({var_time.unit})"
            elif isinstance(var_time.value, (np.ndarray, list)) and len(var_time.value) == self.value.shape[0]:
                 time = var_time.value
                 t_label = f"{var_time.name or 'Time'}"
                 if var_time.unit:
                      t_label += f" ({var_time.unit})"

        if time is None:
            time = np.arange(self.value.shape[0])

        y_label = 'Expectation Value'
        plot_title = f"{self.name or 'Expectation Values'} vs. {t_label.split('(')[0].strip()}"

        # fig = px.line(x=time, y=self.value, labels={'x': t_label, 'y': y_label},
        #               title=plot_title)

        fig = px.line(
            x=time,
            y=self.value.T,  # ← transpose so each expectation operator is a separate trace
            labels={'x': t_label, 'y': y_label},
            title=plot_title
        )

        num_operators = self.value.shape[1]
        operator_names_available = (
            self.e_ops_names is not None and
            len(self.e_ops_names) == num_operators
            )

        if num_operators > 1:
            fig.update_layout(showlegend=True)
            for i, trace in enumerate(fig.data):
                if operator_names_available:
                    # Use name when provided
                    trace.name = self.e_ops_names[i]
                else:
                    # Fallback
                    trace.name = f'&lt;Op_{i}&gt;'
        else:
            fig.update_layout(showlegend=False)

        self.figures.append(PlotlyFigure(label='Expectation Value Plot', figure=fig.to_plotly_json()))
        return fig

    def normalize(self, archive, logger):
        super().normalize(archive, logger)
        try:
            self.plot_expectation_values()
        except Exception as e:
            logger.error(f"Failed to plot expectation values for '{self.name}': {e}")


class QuantumSimulation(Simulation):
    """
    A specialized 'Simulation' to represent quantum calculations
    (time evolution, state preparation, gate-based sim, etc.).
    Inherits from Simulation so it has 'program', 'model_system', 'model_method',
    'outputs', etc. from nomad-simulations by default.
    """

    # Reference a specialized QuantumSystem (instead of the usual ModelSystem)
    quantum_system = SubSection(
        sub_section=QuantumSystem.m_def,
        repeats=False,
        description="""System definition for quantum simulations.
        E.g. qubits, spins etc.""",
    )

    quantum_operators = SubSection(
        sub_section=QuantumOperator.m_def,
        repeats=True,
        description='List of quantum operators (Hamiltonian, jump ops, etc.).',
    )

    quantum_states = SubSection(
        sub_section=QuantumState.m_def,
        repeats=True,
        description='List of quantum states (initial states, final states, etc.).',
    )

    hamiltonian_description = SubSection(
        section_def=ModelHamiltonian,
        description="""
        Describes the model Hamiltonian using a formula and parameters.
        """
    )

    outputs = SubSection(
        section_def=PhysicalProperty,
        repeats=True,
        description="List of calculated physical properties (results) from the simulation."
    )


m_package.__init_metainfo__()
