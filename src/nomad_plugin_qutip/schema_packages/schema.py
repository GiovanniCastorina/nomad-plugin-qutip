from nomad.datamodel.metainfo.annotations import Mapper
from nomad.metainfo import SchemaPackage
from nomad.parsing.file_parser.mapping_parser import MAPPING_ANNOTATION_KEY
from nomad_simulations.schema_packages.general import Program, Simulation

from .schema_package import (
    QuantumObject,
    QuantumOperator,
    QuantumSimulation,
    QuantumSystem,
    QuantumCircuit, #Needed?
    QuantumState, #Needed?
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


QuantumSimulation.quantum_system.m_annotations.setdefault(
    MAPPING_ANNOTATION_KEY, {}
).update(
    dict(info=Mapper(mapper=('get_system', ['.@']), sub_section=QuantumSystem.m_def))
)


QuantumSystem.name.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.name'))
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

###### Quantum states  (needed?)

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


###### Quantum circuit (needed?))
'''
QuantumSimulation.quantum_circuit.m_annotations.setdefault(
    MAPPING_ANNOTATION_KEY, {}
).update(
    dict(
        info=Mapper(
            mapper=('get_circuit', ['.@']),  
            sub_section=QuantumCircuit.m_def,
            repeats=False, 
        )
    )
)
'''
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
