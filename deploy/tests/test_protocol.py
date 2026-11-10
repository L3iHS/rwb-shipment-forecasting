"""Reject arbitrary SSH commands and refs before any privileged deployment operation."""
import os
from pathlib import Path
import subprocess
import unittest

DEPLOY = Path(__file__).resolve().parents[1]


class DeploymentProtocolTests(unittest.TestCase):
    def test_ssh_key_cannot_execute_a_shell_or_arbitrary_command(self):
        for command in ["", "id", "bash", "deploy main", "deploy " + "a" * 40 + "; id", "deploy " + "a" * 40 + " extra"]:
            with self.subTest(command=command):
                result = subprocess.run(
                    ["bash", str(DEPLOY / "ssh-entry.sh")],
                    env={**os.environ, "SSH_ORIGINAL_COMMAND": command},
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 64)
                self.assertIn("Only deploy", result.stderr)

    def test_privileged_entry_rejects_refs_and_extra_arguments(self):
        for args in [[], ["main"], ["a" * 39], ["a" * 40, "extra"], ["a" * 40 + "\nid"]]:
            with self.subTest(args=args):
                result = subprocess.run(
                    ["bash", str(DEPLOY / "main-release.sh"), *args],
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 64)


if __name__ == "__main__":
    unittest.main()
