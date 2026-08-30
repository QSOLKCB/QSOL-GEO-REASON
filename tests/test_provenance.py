import unittest
from pathlib import Path
from unittest.mock import patch

from qsol_geo_reason.provenance import SourceIdentityError, git_source_revision, resolve_implementation_revision


class _Completed:
    def __init__(self, stdout):
        self.stdout = stdout


class ProvenanceTests(unittest.TestCase):
    @patch("qsol_geo_reason.provenance.source_repo_root")
    @patch("qsol_geo_reason.provenance.subprocess.run")
    def test_clean_checkout_binds_head(self, run, repo_root):
        repo_root.return_value = Path("/repo")
        run.side_effect = [_Completed("/repo\n"), _Completed(""), _Completed("abc123\n")]
        self.assertEqual(git_source_revision(), "abc123")

    @patch("qsol_geo_reason.provenance.source_repo_root")
    @patch("qsol_geo_reason.provenance.subprocess.run")
    def test_dirty_checkout_rejected(self, run, repo_root):
        repo_root.return_value = Path("/repo")
        run.side_effect = [_Completed("/repo\n"), _Completed(" M src/qsol_geo_reason/geometry.py\n")]
        with self.assertRaises(SourceIdentityError):
            git_source_revision()

    @patch("qsol_geo_reason.provenance.git_source_revision", return_value="abc123")
    def test_explicit_revision_must_match_checkout(self, _revision):
        with self.assertRaises(SourceIdentityError):
            resolve_implementation_revision("different")

    @patch("qsol_geo_reason.provenance.git_source_revision", return_value=None)
    def test_explicit_revision_allowed_outside_git(self, _revision):
        self.assertEqual(resolve_implementation_revision("release-sha"), "release-sha")
