#!/usr/bin/env python3
"""Local OCR fallback for scanned PDFs (``file_tools.ocr_command``).

A scanned page has no text layer, so extraction finds nothing to return. The deployment may name
its own OCR command; these tests pin the three contracts that make that safe:

* unset — the historic "needs OCR" warning, exactly the old behaviour;
* set and working — the command's stdout is what the caller receives, labelled as recovered text;
* set and broken (non-zero exit, missing binary, unparseable value) — the warning again, never a
  raised exception, because a bad OCR command must not break reading every other document.

The engine itself is not exercised: the command is a stub script, which is the point of the seam.
"""

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import read_extract


class _NeedsOcr(Exception):
    """Stands in for anydoc's NeedsOcrError."""

    def __init__(self, pages):
        super().__init__("needs ocr")
        self.pages = pages


class _FakeModule:
    NeedsOcrError = _NeedsOcr


class LocalOcrFallbackTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.home = Path(self._tmp.name)
        self.pdf = self.home / "scan.pdf"
        self.pdf.write_bytes(b"%PDF-1.4\n% a scanned page\n")
        env = mock.patch.dict(os.environ, {"HERMES_HOME": str(self.home)})
        env.start()
        self.addCleanup(env.stop)

    def _config(self, command):
        """Pin ``file_tools.ocr_command`` as the extraction path reads it."""
        return mock.patch.object(read_extract, "_local_ocr_command", lambda: command or None)

    def _stub(self, body):
        path = self.home / "ocr-stub.sh"
        path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        return str(path)

    def test_unset_command_keeps_the_needs_ocr_warning(self):
        with self._config(None):
            text = read_extract._ocr_scanned_pdf(_FakeModule, str(self.pdf), _NeedsOcr([1]))

        self.assertIn("NEEDS OCR", text)
        self.assertNotIn("local OCR", text)

    def test_a_configured_command_supplies_the_text(self):
        with self._config(self._stub('echo "INVOICE 417"; echo "Total Due: 93.31"')):
            text = read_extract._ocr_scanned_pdf(_FakeModule, str(self.pdf), _NeedsOcr([1, 2]))

        self.assertIn("Total Due: 93.31", text)
        self.assertIn("local OCR", text, "recovered text must be labelled, not passed off as a text layer")
        self.assertIn("1, 2", text, "the pages that needed OCR are named")

    def test_the_command_receives_the_document_path(self):
        seen = self.home / "seen.txt"
        with self._config(self._stub(f'echo "$1" > {seen}')):
            read_extract._run_local_ocr(str(self.pdf))

        self.assertEqual(seen.read_text().strip(), str(self.pdf))

    def test_a_failing_command_falls_back_to_the_warning(self):
        with self._config(self._stub("echo broken >&2; exit 1")):
            self.assertEqual(read_extract._run_local_ocr(str(self.pdf)), "")
            text = read_extract._ocr_scanned_pdf(_FakeModule, str(self.pdf), _NeedsOcr([1]))

        self.assertIn("NEEDS OCR", text, "a failing OCR command must not swallow the warning")

    def test_a_missing_binary_falls_back_to_the_warning(self):
        with self._config("/nonexistent/ocr-engine"):
            self.assertEqual(read_extract._run_local_ocr(str(self.pdf)), "")
            text = read_extract._ocr_scanned_pdf(_FakeModule, str(self.pdf), _NeedsOcr([1]))

        self.assertIn("NEEDS OCR", text)

    def test_an_unparseable_command_value_is_not_a_crash(self):
        with self._config("an 'unbalanced quote"):
            self.assertEqual(read_extract._run_local_ocr(str(self.pdf)), "")

    def test_the_byte_transport_path_recovers_scanned_text(self):
        """read_file passes documents as BYTES, so that entry point needs the fallback too."""
        class _FakeAnyDoc:
            NeedsOcrError = _NeedsOcr

            @staticmethod
            def to_markdown_bytes(data):
                raise _NeedsOcr([1])

        stub = self._stub('echo "RECOVERED VIA BYTES"')
        with self._config(stub), mock.patch.object(read_extract, "_anydoc", lambda: _FakeAnyDoc):
            text = read_extract._extract_anydoc_bytes(self.pdf.read_bytes(), str(self.pdf))

        self.assertIn("RECOVERED VIA BYTES", text)

    def test_a_non_pdf_needs_ocr_failure_is_not_handed_to_ocr(self):
        """Only PDFs go to the OCR command: a scanned .docx is a different failure entirely."""
        class _FakeAnyDoc:
            NeedsOcrError = _NeedsOcr

            @staticmethod
            def to_markdown_bytes(data):
                raise _NeedsOcr([1])

        docx = self.home / "scan.docx"
        docx.write_bytes(b"PK\x03\x04 broken")
        with self._config(self._stub('echo "SHOULD NOT RUN"')), mock.patch.object(read_extract, "_anydoc", lambda: _FakeAnyDoc):
            with self.assertRaises(read_extract.ExtractionError):
                read_extract._extract_anydoc_bytes(docx.read_bytes(), str(docx))


if __name__ == "__main__":
    unittest.main()
