import io
import sys
import unittest
from unittest.mock import patch

from shardserve.databricks import main


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
