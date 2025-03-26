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
from nomad.datamodel.metainfo.annotations import ELNAnnotation, ELNComponentEnum
from nomad.metainfo import MEnum, Quantity, SchemaPackage, SubSection
from nomad_simulations.schema_packages.general import Simulation
from nomad_simulations.schema_packages.model_system import ModelSystem

configuration = config.get_plugin_entry_point(
    'nomad_plugin_qutip.schema_packages:schema_package_entry_point'
)

m_package = SchemaPackage()


class NewSchemaPackage(Schema):
    name = Quantity(
        type=str, a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity)
    )
    message = Quantity(type=str)

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        logger.info('NewSchema.normalize', parameter=configuration.parameter)
        self.message = f'Hello {self.name}!'


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

    # def normalize(self, archive, logger) -> None:
    #     """
    #     check consistency between 'shape' and data.shape,
    #     and optionally perform further consistency checks (Hermiticity, etc.).
    #     """
    #     super().normalize(archive, logger)
    #     if self.data is not None and len(self.shape) == 2:
    #         actual_data_shape = list(self.data.shape)
    #         declared_shape = list(self.shape)
    #         if declared_shape != actual_data_shape:
    #             logger.warning(
    #                 'Inconsistent shape!'
    #             )


class QuantumSystem(ModelSystem):
    """
    A specialized 'model system' for quantum information simulations,
    e.g., representing a set of qubits, spins, or multi-level systems.
    Inherits from ModelSystem to remain compatible with existing archiving.
    """

    name = Quantity(type=str, description='Optional label for this quantum system.')

    num_qubits = Quantity(
        type=int, description='Number of qubits/spin-1/2 sites in this quantum system.'
    )


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


class QuantumCircuit(ArchiveSection):
    """
    A container for gate-based quantum circuits, e.g., from OpenQASM or Cirq.
    """

    circuit_representation = Quantity(
        type=str, description='Circuit definition (OpenQASM, Cirq JSON, etc.).'
    )

class Parameter(ArchiveSection):
    """
    Single parameter name-value in Hamiltonian.
    """
    m_def = ArchiveSection(label_quantity='name') 

    name = Quantity(
        type=str,
        description='Name of the parameter (eg. "a", "coupling_strength").'
    )
    value = Quantity(
        type=np.float64, # Usiamo float per i valori numerici
        description='Numerical value of the parameter.'
    )
    unit = Quantity(type=str, description='Unit of the parameter')


class HamiltonianModel(ArchiveSection):
    """
    Text string for the Hamiltonian.
    """
    formula = Quantity(
        type=str,
        description='Hamiltonian formula string.'
    )
    parameters = SubSection(
        sub_section=Parameter.m_def,
        repeats=True, # More than one parameter
        description='Parameters list for the Hamiltonian.'
    )

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

    quantum_circuit = SubSection(
        sub_section=QuantumCircuit.m_def,
        repeats=False,
        description='Gate-based circuit if relevant to this calculation.',
    )

    hamiltonian_model = SubSection(
        sub_section=HamiltonianModel.m_def,
        repeats=False, # Just One Hamiltonian
        description='Description of Hamiltonian an ist parameters.'
    )
    #It lacks outputs? 
    #It should have like https://qutip.org/docs/4.0.2/modules/qutip/mesolve.html ?
    #To understand better


m_package.__init_metainfo__()
