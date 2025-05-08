from nomad.datamodel.metainfo.annotations import Mapper
from nomad.metainfo import SchemaPackage
from nomad.parsing.file_parser.mapping_parser import MAPPING_ANNOTATION_KEY
from nomad_simulations.schema_packages.general import Program, Simulation
from nomad_simulations.schema_packages.numerical_settings import NumericalSettings
from nomad_simulations.schema_packages.physical_property import PhysicalProperty
from nomad_simulations.schema_packages.variables import Variables

from .schema_package import (
    QuantumObject,
    QuantumOperator,
    QuantumSimulation,
    QuantumSystem,
    QuantumState,
    SpinHamiltonian,
    SpinHamiltonianParameter,
    EigenvaluesInVariable,
    TimeEvolutionProperty,
    SolverStats,
    )

m_package = SchemaPackage()


Simulation.m_def.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='@'))
)

Simulation.program.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper=('get_program', ['.@']), sub_section=Program.m_def))
)

Program.name.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.name'))
)
Program.version.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.version'))
)


# --- Extended Annotations for QuantumSimulation ---
QuantumSimulation.m_def.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='@'))
)


QuantumSimulation.model_system.m_annotations.setdefault(
    MAPPING_ANNOTATION_KEY, {}
).update(
    dict(info=Mapper(mapper=('get_system', ['.@']), sub_section=QuantumSystem.m_def))
)


QuantumSystem.name.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.name'))
)

QuantumSystem.num_qubits.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(dict(info=Mapper(mapper='.num_qubits')))

QuantumSimulation.spin_hamiltonian.m_annotations.setdefault(
    MAPPING_ANNOTATION_KEY, {}
).update(
    dict(
        info=Mapper(
            # This key should match the key used in json_writer.py
            mapper='spin_hamiltonian',
            sub_section=SpinHamiltonian.m_def
        )
    )
)


SpinHamiltonian.name.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.name'))
)
SpinHamiltonian.formula.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.formula'))
)
SpinHamiltonian.parameters.m_def.m_annotations.setdefault(
    MAPPING_ANNOTATION_KEY, {}
).update(
    dict(
        info=Mapper(
            mapper='.parameters',
            sub_section=SpinHamiltonianParameter.m_def,
            repeats=True,
        )
    )
)

# Mappings for SpinHamiltonianParameter fields
SpinHamiltonianParameter.value.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.value'))
)
SpinHamiltonianParameter.unit.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.unit'))
)

######  Quantum operators

QuantumSimulation.quantum_operators.m_def.m_annotations.setdefault(
    MAPPING_ANNOTATION_KEY, {}
).update(
    dict(
        info=Mapper(
            mapper=('get_operators', ['.quantum_operators']),
            sub_section=QuantumOperator.m_def,
            repeats=True,
        )
    )
)

QuantumOperator.name.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.name'))
)
QuantumOperator.quantum_object.m_annotations.setdefault(
    MAPPING_ANNOTATION_KEY, {}
).update(dict(info=Mapper(mapper='.quantum_object')))

###### Quantum states

QuantumSimulation.quantum_states.m_def.m_annotations.setdefault(
    MAPPING_ANNOTATION_KEY, {}
).update(
    dict(
        info=Mapper(
            mapper=('get_states', ['.quantum_states']),
            sub_section=QuantumState.m_def,
            repeats=True,
        )
    )
)

QuantumState.label.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.label'))
)
# Mapping for quantum_object within QuantumState relies on the global QuantumObject mappers below
QuantumState.quantum_object.m_annotations.setdefault(
    MAPPING_ANNOTATION_KEY, {}
).update(dict(info=Mapper(mapper='.quantum_object', sub_section=QuantumObject.m_def)))

#Mapping for results
QuantumSimulation.outputs.m_def.m_annotations.setdefault(
    MAPPING_ANNOTATION_KEY, {}
).update(
    dict(
        info=Mapper(
            
            mapper=('get_outputs', ['.results']), 
            repeats=True
            
        )
    )
)

# Maps EigenvaluesInVariable
EigenvaluesInVariable.name.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.property_name'))
)
EigenvaluesInVariable.value.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.eigenvalues'))
)
EigenvaluesInVariable.variables.m_def.m_annotations.setdefault(
    MAPPING_ANNOTATION_KEY, {}
).update(
    dict(
        info=Mapper(
            mapper=lambda data: [data.get('variable')] if data.get('variable') else [],
            sub_section=Variables.m_def,
            repeats=True
        )
    )
)

# --- Maps TimeEvolutionProperty
TimeEvolutionProperty.name.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.property_name'))
)
TimeEvolutionProperty.value.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.expectation_values'))
)
TimeEvolutionProperty.variables.m_def.m_annotations.setdefault(
    MAPPING_ANNOTATION_KEY, {}
).update(
     dict(
        info=Mapper(
            mapper=lambda data: [data.get('variable')] if data.get('variable') else [],
            sub_section=Variables.m_def,
            repeats=True
        )
    )
)
TimeEvolutionProperty.solver_statistics.m_annotations.setdefault(
    MAPPING_ANNOTATION_KEY, {}
).update(
    dict(
        info=Mapper(
            mapper='.solver_stats',
            sub_section=SolverStats.m_def
        )
    )
)

TimeEvolutionProperty.e_ops_names.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.e_ops_names'))
)

# --- Map for Variable 
Variables.name.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.name'))
)
Variables.unit.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.unit'))
)
Variables.value.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.values'))
)

# --- Map SolverStats ---

SolverStats.solver_name.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.solver_name'))
)
SolverStats.method.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.method'))
)
SolverStats.solver_description.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.description'))
)
SolverStats.initialization_time.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.init_time_s')) # Dal JSON: 'init_time_s'
)
SolverStats.preparation_time.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.prep_time_s')) # Dal JSON: 'prep_time_s'
)
SolverStats.run_time.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.run_time_s')) # Dal JSON: 'run_time_s'
)
SolverStats.time_interval_start.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.t_start')) # Dal JSON: 't_start'
)
SolverStats.time_interval_end.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.t_end')) # Dal JSON: 't_end'
)
SolverStats.n_steps.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.n_steps')) # Dal JSON: 'n_steps'
)
SolverStats.n_expectation_operators.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.num_e_ops')) # Dal JSON: 'num_e_ops'
)
SolverStats.final_state_saved.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.final_state_saved')) # Dal JSON: 'final_state_saved'
)
# --- Mapping for QuantumObject internal fields ---
QuantumObject.dims.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.dims'))
)
QuantumObject.shape.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.shape'))
)
QuantumObject.type.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.type'))
)
QuantumObject.storage_format.m_annotations.setdefault(
    MAPPING_ANNOTATION_KEY, {}
).update(dict(info=Mapper(mapper='.storage_format')))
QuantumObject.is_hermitian.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.is_hermitian'))
)
QuantumObject.data.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.data'))
)

m_package.__init_metainfo__()