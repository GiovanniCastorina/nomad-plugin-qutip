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

from nomad_plugin_qutip.schema_packages.schema_package import QuantumSimulation

configuration = config.get_plugin_entry_point(
    'nomad_plugin_qutip.parsers:parser_entry_point'
)


class NewParser(MatchingParser):
    def parse(
        self,
        mainfile: str,
        archive: 'EntryArchive',
        logger: 'BoundLogger',
        child_archives: dict[str, 'EntryArchive'] = None,
    ) -> None:
        logger.info('NewParser.parse', parameter=configuration.parameter)

        archive.workflow2 = Workflow(name='test')

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
        return {'name': source.get('simulation_name', 'HASSIKTR')}

    def get_operators(source: dict, **kwargs) -> dict:
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
        Filters and includes ONLY states explicitly marked with type 'ket' or 'bra'.
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


            # Check the 'type' and go if is ket or bra
            state_type = qobj_target.get('type')

            if state_type == 'ket' or state_type == 'bra':

                processed_states.append({
                    'label': state_label,       
                    'quantum_object': qobj_target 
                })

            else:
                # If the type is not 'ket' or 'bra' 
                # we skip this entry 
                if self.logger:
                    self.logger.debug(
                        f"Skipping state '{state_label}' during parsing because its type "
                        f"is '{state_type}' (expected 'ket' or 'bra')."
                    )

        return {'quantum_states': processed_states}
    '''
     def get_circuit(self, source: dict, **kwargs) -> Optional[dict]:
        """
        Process circuit data from the JSON source (expected to be root).
        Looks for a 'circuit' key containing circuit details.
        Returns a dictionary {'circuit_representation': circuit_string} or None.
        """
        circuit_source = source.get('circuit')
        if circuit_source and isinstance(circuit_source, dict):
            circuit_def = circuit_source.get('definition')
            if circuit_def:
                # Return dict structure expected by the mapper for 'quantum_circuit' subsection
                # The key 'circuit_representation' matches the quantity name in QuantumCircuit schema
                return {'circuit_representation': str(circuit_def)}
            else:
                 if self.logger:
                     self.logger.warning("Found 'circuit' section in JSON, but no 'definition' inside.")
        return None # Return None if no circuit data is found'
    '''
    def get_hamiltonian_parameters(self, source: Optional[Dict[str, Any]], **kwargs) -> Dict[str, List[Dict[str, Any]]]:
        """
        Converts the parameter dictionary from JSON (from 'parameter_values')
        into a list of dictionaries suitable for populating the 'Parameter' subsections.

        Args:
        source: The JSON dictionary from ['hamiltonian_model']['parameter_values'].

        Returns:
        A dictionary {'parameters': [list_of_parameters]}, where each element
        in the list is {'name': param_name, 'value': param_value}.
        """
        processed_params: List[Dict[str, Any]] = []
         # Check if the input is a valid dictionary
        if source is None:
             if self.logger:
                  self.logger.info("Source for Hamiltonian parameters is None (key 'parameter_values' likely missing or null). No parameters parsed.")
             return {'parameters': processed_params} # Returns the expected structure with an empty list

        if not isinstance(source, dict):
            if self.logger:
                self.logger.error(f"Expected a dictionary for 'parameter_values', but got {type(source)}. Skipping parameters.")
            return {'parameters': processed_params}

        # Iterate over the key-value pairs of the parameter dictionary
        for param_name, param_value in source.items():
            try:
                # Attempt to convert the value to a float
                numeric_value = float(param_value)
                # Create the dictionary in the format expected by Metainfo (keys 'name', 'value')
                processed_params.append({
                    'name': str(param_name), # Name is a string
                    'value': numeric_value
                })
            except (ValueError, TypeError) as e:
                # Warnings and such
                if self.logger:
                    self.logger.warning(
                        f"Could not convert value for parameter '{param_name}' to float. Value was: '{param_value}'. Skipping parameter."
                        # exc_info=e 
                    )
                continue # Skips this one

        # Return the final dictionary in the required format for the mapper with repeats=True
        return {'parameters': processed_params}





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