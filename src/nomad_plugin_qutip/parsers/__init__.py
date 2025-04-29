from nomad.config.models.plugins import ParserEntryPoint
from pydantic import Field


class NomadParserEntryPoint(ParserEntryPoint):
    def load(self):
        from nomad_plugin_qutip.parsers.parser import QutipParser

        return QutipParser(**self.dict())


parser = NomadParserEntryPoint(
    name='QutipParser',
    description='Parses JSON output files from QuTiP simulations.',
    mainfile_name_re=r'.*\.json$',
    mainfile_mime_re=r'application/json',
)
