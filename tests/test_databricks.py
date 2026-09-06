import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from shardserve.databricks import main
from shardserve import evidence


class DatabricksCLI(unittest.TestCase):
    def test_required_arguments_fail_before_importing_spark(self):
        commands = (
            ['prepare', '--directory', '/tmp/run'],
            ['run', '--directory', '/tmp/run'],
            ['run', '--directory', '/tmp/run', '--results-table', 'catalog.schema.results'],
            ['validate', '--directory', '/tmp/run'],
        )
        for command in commands:
            with self.subTest(command=command), patch.object(sys, 'argv', ['shardserve.databricks', *command]), patch.object(sys, 'stderr', io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main()
                self.assertEqual(raised.exception.code, 2)

    def test_source_digest_works_from_installed_package(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / 'shardserve'
            package.mkdir()
            module = package / 'module.py'
            module.write_text('value = 1\n')
            with patch.object(evidence, '__file__', str(package / 'evidence.py')):
                first = evidence.source()
                module.write_text('value = 2\n')
                self.assertNotEqual(first, evidence.source())
