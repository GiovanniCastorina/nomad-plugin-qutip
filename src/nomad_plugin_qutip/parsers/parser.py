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
import json

import numpy as np
from nomad.config import config
from nomad.parsing.file_parser.mapping_parser import MappingParser, MetainfoParser
from nomad.datamodel.metainfo.workflow import Workflow
from nomad.parsing.parser import MatchingParser
from typing import Any

from nomad_plugin_qutip.schema_packages.schema_package import QuantumSimulation,EigenvaluesInVariable,TimeEvolutionProperty

configuration = config.get_plugin_entry_point(
    'nomad_plugin_qutip.parsers:parser_entry_point'
)

class JSONParser(MappingParser):
    """
    A minimal JSON-based MappingParser, analogous to XMLParser but for JSON.
    """

    value_key = '__value'
    attribute_prefix = '@'

    def load_file(self):
        with open(self.filepath) as f:
            self.data_object = json.load(f)
        return self.data_object

    def to_dict(self, **kwargs) -> dict:
        """
        Return a dictionary representation of data_object. If the file is
        already JSON, data_object may simply be a dict, so just return it.
        """
        if isinstance(self.data_object, dict):
            return self.data_object
        return {}

    def from_dict(self, data: dict, **kwargs):
        """
        Set the parser’s internal data_object from a given dictionary.
        """
        self.data_object = data

    def get_program(self, source: dict[str, Any], **kwargs) -> dict[str, Any]:
        return source.get('program', {'name': '', 'version': ''})

    def get_system(self, source: dict[str, Any], **kwargs) -> dict[str, Any]:
        return {'name': source.get('simulation_name', 'default')}

    def get_operators(self, source: dict, **kwargs) -> dict:
        """
        Process operator data from the JSON.

        For each operator, it returns a dictionary with two keys:
        - "name": the operator name (string)
        - "quantum_object": a dictionary with fields required by QuantumObject:
            "dims", "shape", "type", "storage_format", "is_hermitian", and "data".

        In particular, if the operator data contains a "matrix" key
        (with "re" and "im"),
        these are combined into a single complex matrix stored under "data".

        The function returns a dictionary with a single key "quantum_operators"
        whose value is the list.
        """
        ops = source.get('operators', {})
        processed = []
        for op_name, op in ops.items():
            qobj = {}
            if 'dims' in op:
                qobj['dims'] = op['dims']
                try:
                    # Assume dims is of the form [[n],[m]] and derive shape as [n, m]
                    dimension = 2
                    if len(op['dims']) == dimension:
                        qobj['shape'] = [int(op['dims'][0][0]), int(op['dims'][1][0])]
                    else:
                        qobj['shape'] = []
                except Exception:
                    qobj['shape'] = []
            qobj['type'] = op.get('type', 'oper')
            qobj['storage_format'] = op.get('storage_format', 'Dense')
            qobj['is_hermitian'] = op.get('is_hermitian', True)
            if 'matrix' in op:
                matrix_dict = op['matrix']
                re = np.array(matrix_dict.get('re', []))
                im = np.array(matrix_dict.get('im', []))
                qobj['data'] = (re + 1j * im).tolist()
            else:
                qobj['data'] = op.get('data', None)
            processed.append({'name': op_name, 'quantum_object': qobj})
        return {'quantum_operators': processed}


    def get_states(self, source: dict, **kwargs) -> dict:
        """
        Process state data from the JSON source.
        Looks for a 'states' key containing a dictionary of states.
        Filters states explicitly marked with type 'ket' or 'bra'.
        If they are marked as 'oper' then counts it as a density matrix ('dm').
        Returns a dictionary {'quantum_states': [list of filtered state dicts]}.
        """
        # Get the dictionary of states from JSON
        states_source = source.get('states', {})
        processed_states = []

        # Iterate through each state found JSON
        for state_label, state_data in states_source.items():
            # Extract the quantum object data

            qobj_source = state_data.get('quantum_object', state_data)

            # Process the quantum object data
            qobj_target = self._process_quantum_object_data(qobj_source)


            # Check the 'type' and go if is ket or bra or oper
            state_type = qobj_target.get('type')
            # the throws away all data populated by _process_quantum_objec_data method.
            # hence commented out
            #qobj_target = {}
            if state_type == 'ket' or state_type == 'bra':
                if 'dims' in qobj_source:
                    qobj_target['dims'] = qobj_source['dims']
                    try:
                        dim0 = np.prod(qobj_source['dims'][0]) if qobj_source['dims'][0] else 1
                        dim1 = np.prod(qobj_source['dims'][1]) if qobj_source['dims'][1] else 1
                        qobj_target['shape'] = [int(dim0), int(dim1)]
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f"State '{state_label}' (ket/bra): Could not determine shape from dims {qobj_source['dims']}: {e}")
                        qobj_target['shape'] = []
                else:
                    qobj_target['shape'] = qobj_source.get('shape', [])

                #qobj_target['type'] = source_type # Keep original type for ket and bra
                qobj_target['type'] = state_type

                qobj_target['storage_format'] = qobj_source.get('storage_format', 'Dense')

                qobj_target['is_hermitian'] = qobj_source.get('is_hermitian', None)
                processed_states.append({
                    'label': state_label,
                    'quantum_object': qobj_target
                })

            elif state_type == 'oper':
                if 'dims' in qobj_source:
                    qobj_target['dims'] = qobj_source['dims']
                    try:
                        dim0 = np.prod(qobj_source['dims'][0]) if qobj_source['dims'][0] else 1
                        dim1 = np.prod(qobj_source['dims'][1]) if qobj_source['dims'][1] else 1
                        qobj_target['shape'] = [int(dim0), int(dim1)]
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f"State '{state_label}' (oper): Could not determine shape from dims {qobj_source['dims']}: {e}")
                        qobj_target['shape'] = []
                else:
                    qobj_target['shape'] = qobj_source.get('shape', [])
                #Changing to 'dm'
                qobj_target['type'] = 'dm'

                qobj_target['storage_format'] = qobj_source.get('storage_format', 'Dense')
                qobj_target['is_hermitian'] = qobj_source.get('is_hermitian', None)
                if 'matrix' in state_data:
                    matrix_dict = state_data['matrix']
                    re = np.array(matrix_dict.get('re', []))
                    im = np.array(matrix_dict.get('im', []))
                    qobj_target['data'] = (re + 1j * im).tolist()
                else:
                    qobj_target['data'] = state_data.get('data', None)
                processed_states.append({
                    'label': state_label,
                    'quantum_object': qobj_target
                })


            else:
                # If the type is not 'ket','bra' or 'oper' we skip this entry
                if self.logger:
                    self.logger.debug(
                        f"Skipping state '{state_label}' during parsing because its type "
                        f"is '{state_type}' (expected 'ket','bra' or 'oper')."
                    )

        return {'quantum_states': processed_states}
    def get_outputs(self, source: dict, **kwargs) -> list[dict]:
        processed_outputs = []
        if not isinstance(source, dict):
            return processed_outputs

        for result_key, result_data in source.items():
            if not isinstance(result_data, dict):
                continue

            calc_type = result_data.get('calculation_type')

            # Dict for EigenvaluesInVariable
            if calc_type == 'eigenvalues':
                output_dict = {}
                output_dict['__section_def'] = EigenvaluesInVariable.m_def
                output_dict.update(result_data)
                processed_outputs.append(output_dict)

            # Dict for TimeEvolutionProperty
            elif calc_type == 'time_evolution':
                output_dict = {}
                output_dict['__section_def'] = TimeEvolutionProperty.m_def
                output_dict.update(result_data)
                processed_outputs.append(output_dict)

            else:
                if self.logger:
                    self.logger.warning(f"Skipping result '{result_key}' with unknown "
                                        f"calculation_type: '{calc_type}'")

        return processed_outputs




class QutipParser(MatchingParser):
    def parse(
        self,
        mainfile: str,
        archive: 'EntryArchive',
        logger,
        child_archives: dict[str, 'EntryArchive'] = None,
    ) -> None:
        # Create your JSON parser and assign the file path
        json_parser = JSONParser()
        json_parser.filepath = mainfile

        data_object = QuantumSimulation()

        # Create a MetainfoParser using the QuantumSimulation instance
        data_parser = MetainfoParser(data_object=data_object)
        data_parser.annotation_key = 'info'

        # Convert from JSON parser to MetainfoParser
        json_parser.convert(data_parser)

        # Store the resulting data object in the archive
        archive.data = data_parser.data_object

        # Optionally close the parsers
        data_parser.close()
        json_parser.close()