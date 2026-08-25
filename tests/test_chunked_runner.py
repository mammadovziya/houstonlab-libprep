from __future__ import annotations

import csv
import io
import math
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rdkit import Chem
from rdkit.Chem import AllChem

import run_conformers_chunked as runner


class FakeHardwareOptions:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def fake_nvmolkit_modules(embed, optimize):
    package = types.ModuleType("nvmolkit")
    package.__path__ = []
    embed_module = types.ModuleType("nvmolkit.embedMolecules")
    embed_module.EmbedMolecules = embed
    optimize_module = types.ModuleType("nvmolkit.mmffOptimization")
    optimize_module.MMFFOptimizeMoleculesConfs = optimize
    types_module = types.ModuleType("nvmolkit.types")
    types_module.HardwareOptions = FakeHardwareOptions
    package.embedMolecules = embed_module
    package.mmffOptimization = optimize_module
    package.types = types_module
    return {
        "nvmolkit": package,
        "nvmolkit.embedMolecules": embed_module,
        "nvmolkit.mmffOptimization": optimize_module,
        "nvmolkit.types": types_module,
    }


def add_conformers(mol, count):
    for _ in range(count):
        mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()), assignId=True)


def complete_embed(mols, params, confsPerMolecule, hardwareOptions):
    for mol in mols:
        add_conformers(mol, confsPerMolecule)


def finite_energies(mols, **kwargs):
    return [
        [-float(index + 1) for index in range(mol.GetNumConformers())]
        for mol in mols
    ]


def recording_writer_type(instances):
    class RecordingWriter:
        def __init__(self, path):
            self.path = Path(path)
            self.handle = self.path.open("wb")
            self.closed = False
            instances.append(self)

        def write(self, mol, confId):
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else "unnamed"
            self.handle.write(f"{name}:{confId}\n".encode())

        def close(self):
            if not self.closed:
                self.handle.close()
                self.closed = True

    return RecordingWriter


class ChunkedRunnerRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.temp_dir.name)
        self.input_path = self.work_dir / "input.smi"
        self.input_path.write_text("CCO\tmol-1\n", encoding="utf-8")
        self.output_path = self.work_dir / "library.sdf"
        self.writer_instances = []
        self.writer_class = recording_writer_type(self.writer_instances)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def run_once(self, **overrides):
        options = {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "chunk_size": 10,
            "n_conformers": 1,
            "mmff_max_iters": 50,
            "batch_size": 8,
            "batches_per_gpu": 2,
            "preprocessing_threads": 3,
            "random_seed": 123,
            "allow_partial_conformers": False,
            "check_free_space": False,
        }
        options.update(overrides)
        return runner.run_chunked(**options)

    def test_success_atomically_promotes_staging_and_releases_lock(self) -> None:
        prior_final = b"previous completed library"
        self.output_path.write_bytes(prior_final)
        replacements = []
        real_replace = os.replace

        def recording_replace(source, destination):
            source = Path(source)
            destination = Path(destination)
            self.assertEqual(destination, self.output_path)
            self.assertNotEqual(source, self.output_path)
            self.assertEqual(self.output_path.read_bytes(), prior_final)
            self.assertTrue(self.writer_instances[-1].closed)
            replacements.append((source, destination))
            real_replace(source, destination)

        modules = fake_nvmolkit_modules(complete_embed, finite_energies)
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(Chem, "SDWriter", self.writer_class),
            mock.patch.object(runner.os, "replace", side_effect=recording_replace),
        ):
            self.run_once()

        self.assertEqual(len(replacements), 1)
        staged_path, final_path = replacements[0]
        self.assertEqual(final_path, self.output_path)
        self.assertEqual(staged_path.parent, self.output_path.parent)
        self.assertTrue(staged_path.name.endswith(".partial.sdf"))
        self.assertFalse(staged_path.exists())
        self.assertEqual(self.output_path.read_bytes(), b"mol-1:0\n")

        lock = runner._OutputLock(self.output_path)
        self.assertIsNotNone(lock.handle)
        lock.close()

    def test_strict_persistent_shortfall_preserves_final_and_reports_failure(self) -> None:
        prior_final = b"known-good-final"
        self.output_path.write_bytes(prior_final)

        def always_short(mols, params, confsPerMolecule, hardwareOptions):
            for mol in mols:
                add_conformers(mol, 1)

        optimize = mock.Mock(side_effect=AssertionError("shortfall must not be optimized"))
        modules = fake_nvmolkit_modules(always_short, optimize)
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(Chem, "SDWriter", self.writer_class),
        ):
            with self.assertRaisesRegex(RuntimeError, "did not produce all 2 requested"):
                self.run_once(n_conformers=2)

        self.assertEqual(self.output_path.read_bytes(), prior_final)
        optimize.assert_not_called()
        failure_path = runner._failure_csv_path(self.output_path)
        with failure_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["molecule_id"], "mol-1")
        self.assertEqual(rows[0]["stage"], "embedding")
        self.assertEqual(rows[0]["reason"], "fewer_than_requested")
        self.assertEqual(rows[0]["requested"], "2")
        self.assertEqual(rows[0]["generated"], "1")
        self.assertEqual(rows[0]["primary_conformers"], "1")
        self.assertEqual(rows[0]["retry_conformers"], "1")

        self.assertEqual(len(self.writer_instances), 1)
        self.assertNotEqual(self.writer_instances[0].path, self.output_path)
        self.assertTrue(self.writer_instances[0].closed)
        lock = runner._OutputLock(self.output_path)
        lock.close()

    def test_hardlinked_failure_report_cannot_truncate_existing_final(self) -> None:
        prior_final = b"known-good-final"
        self.output_path.write_bytes(prior_final)
        failure_path = runner._failure_csv_path(self.output_path)
        os.link(self.output_path, failure_path)

        with self.assertRaisesRegex(ValueError, "aliases planned output SDF"):
            self.run_once()

        self.assertEqual(self.output_path.read_bytes(), prior_final)

    def test_hardlinked_mmff_report_cannot_truncate_existing_final(self) -> None:
        prior_final = b"known-good-final"
        self.output_path.write_bytes(prior_final)
        mmff_path = runner._mmff_csv_path(self.output_path)
        os.link(self.output_path, mmff_path)

        with self.assertRaisesRegex(ValueError, "MMFF94s report aliases planned output SDF"):
            self.run_once()

        self.assertEqual(self.output_path.read_bytes(), prior_final)

    def test_in_place_input_edit_during_exact_read_prevents_promotion(self) -> None:
        prior_final = b"known-good-final"
        self.output_path.write_bytes(prior_final)
        mutated = False

        def mutate_after_embedding(mols, params, confsPerMolecule, hardwareOptions):
            nonlocal mutated
            complete_embed(mols, params, confsPerMolecule, hardwareOptions)
            if not mutated:
                mutated = True
                self.input_path.write_text("CCN\tmol-1\n", encoding="utf-8")

        modules = fake_nvmolkit_modules(mutate_after_embedding, finite_energies)
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(Chem, "SDWriter", self.writer_class),
        ):
            with self.assertRaisesRegex(RuntimeError, "Input changed"):
                self.run_once()

        self.assertTrue(mutated)
        self.assertEqual(self.output_path.read_bytes(), prior_final)

    def test_symlink_retarget_before_commit_prevents_promotion(self) -> None:
        first = self.work_dir / "first.smi"
        second = self.work_dir / "second.smi"
        first.write_text("CCO\tfirst\n", encoding="utf-8")
        second.write_text("CCN\tsecond\n", encoding="utf-8")
        self.input_path.unlink()
        self.input_path.symlink_to(first)
        prior_final = b"known-good-final"
        self.output_path.write_bytes(prior_final)
        base_writer = self.writer_class
        input_link = self.input_path

        class RetargetingWriter(base_writer):
            def close(inner_self):
                already_closed = inner_self.closed
                super().close()
                if not already_closed:
                    input_link.unlink()
                    input_link.symlink_to(second)

        modules = fake_nvmolkit_modules(complete_embed, finite_energies)
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(Chem, "SDWriter", RetargetingWriter),
        ):
            with self.assertRaisesRegex(RuntimeError, "Input changed"):
                self.run_once()

        self.assertEqual(self.input_path.resolve(), second.resolve())
        self.assertEqual(self.output_path.read_bytes(), prior_final)

    def test_read_only_final_mode_is_applied_only_after_writer_closes(self) -> None:
        self.output_path.write_bytes(b"known-good-final")
        self.output_path.chmod(0o444)
        modules = fake_nvmolkit_modules(complete_embed, finite_energies)

        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(Chem, "SDWriter", self.writer_class),
        ):
            self.run_once()

        self.assertEqual(self.output_path.read_bytes(), b"mol-1:0\n")
        self.assertEqual(self.output_path.stat().st_mode & 0o777, 0o444)

    def test_mmff_unparametrizable_molecules_are_streamed_and_counted(self) -> None:
        modules = fake_nvmolkit_modules(complete_embed, finite_energies)
        stdout = io.StringIO()
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(Chem, "SDWriter", self.writer_class),
            mock.patch.object(AllChem, "MMFFGetMoleculeProperties", return_value=None),
            redirect_stdout(stdout),
        ):
            self.run_once()

        mmff_path = runner._mmff_csv_path(self.output_path)
        with mmff_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(
            rows,
            [{"ID": "mol-1", "reason": "mmff94s_unparametrizable"}],
        )
        self.assertIn("MMFF94s-unparametrizable: 1", stdout.getvalue())
        self.assertIn(f"MMFF94s CSV: {mmff_path} (1 records)", stdout.getvalue())

    def test_empty_or_header_only_input_cannot_replace_existing_final(self) -> None:
        modules = fake_nvmolkit_modules(complete_embed, finite_energies)
        for contents in ("", "SMILES\tID\n"):
            with self.subTest(contents=contents):
                self.input_path.write_text(contents, encoding="utf-8")
                self.output_path.write_bytes(b"known-good-final")
                with (
                    mock.patch.dict(sys.modules, modules),
                    mock.patch.object(Chem, "SDWriter", self.writer_class),
                ):
                    with self.assertRaisesRegex(RuntimeError, "no molecule rows"):
                        self.run_once()
                self.assertEqual(
                    self.output_path.read_bytes(), b"known-good-final"
                )

    def test_accounting_mismatch_cannot_replace_existing_final(self) -> None:
        prior_final = b"known-good-final"
        self.output_path.write_bytes(prior_final)
        modules = fake_nvmolkit_modules(complete_embed, finite_energies)
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(Chem, "SDWriter", self.writer_class),
            mock.patch.object(runner, "_embed_and_write", return_value=(2, 0)),
        ):
            with self.assertRaisesRegex(RuntimeError, "accounting error"):
                self.run_once()

        self.assertEqual(self.output_path.read_bytes(), prior_final)

    def test_each_run_uses_a_unique_staging_path(self) -> None:
        modules = fake_nvmolkit_modules(complete_embed, finite_energies)
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(Chem, "SDWriter", self.writer_class),
        ):
            self.run_once()
            self.run_once()

        staged_paths = [writer.path for writer in self.writer_instances]
        self.assertEqual(len(staged_paths), 2)
        self.assertEqual(len(set(staged_paths)), 2)
        for path in staged_paths:
            self.assertEqual(path.parent, self.output_path.parent)
            self.assertNotEqual(path, self.output_path)
            self.assertTrue(path.name.startswith("library."))
            self.assertTrue(path.name.endswith(".partial.sdf"))
            self.assertFalse(path.exists())

    def test_retry_uses_next_seed_and_passes_explicit_mmff94s_properties(self) -> None:
        mols, parse_failures = runner._load_chunk([("CCO", "mol-1")])
        self.assertEqual(parse_failures, 0)
        embed_calls = []

        def recover_on_retry(molecules, params, confsPerMolecule, hardwareOptions):
            embed_calls.append((params.randomSeed, hardwareOptions))
            count = 1 if len(embed_calls) == 1 else confsPerMolecule
            for mol in molecules:
                add_conformers(mol, count)

        optimizer_calls = []

        def optimize(molecules, **kwargs):
            optimizer_calls.append((molecules, kwargs))
            return [[-1.0, -2.0] for _ in molecules]

        modules = fake_nvmolkit_modules(recover_on_retry, optimize)
        properties = object()
        memory_writer = mock.Mock()
        failures = []
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(
                AllChem, "MMFFGetMoleculeProperties", return_value=properties
            ) as get_properties,
        ):
            written, incomplete = runner._embed_and_write(
                mols,
                memory_writer,
                n_conformers=2,
                mmff_max_iters=77,
                batch_size=8,
                batches_per_gpu=4,
                preprocessing_threads=6,
                random_seed=500,
                failure_records=failures,
                chunk_idx=3,
            )

        self.assertEqual([seed for seed, _ in embed_calls], [500, 501])
        primary_hardware = embed_calls[0][1]
        self.assertEqual(primary_hardware.batchSize, 8)
        self.assertEqual(primary_hardware.batchesPerGpu, 4)
        self.assertEqual(primary_hardware.preprocessingThreads, 6)
        retry_hardware = embed_calls[1][1]
        self.assertEqual(retry_hardware.batchSize, 1)
        self.assertEqual(retry_hardware.batchesPerGpu, 1)
        self.assertEqual(retry_hardware.preprocessingThreads, 1)

        get_properties.assert_called_once_with(mols[0], mmffVariant="MMFF94s")
        self.assertEqual(len(optimizer_calls), 1)
        optimized_mols, optimizer_kwargs = optimizer_calls[0]
        self.assertEqual(optimized_mols, mols)
        self.assertEqual(optimizer_kwargs["properties"], [properties])
        self.assertEqual(optimizer_kwargs["maxIters"], 77)
        self.assertIs(optimizer_kwargs["hardwareOptions"], primary_hardware)
        self.assertEqual(written, 2)
        self.assertEqual(incomplete, 0)
        self.assertEqual(failures, [])
        self.assertEqual(memory_writer.write.call_count, 2)

    def test_sampled_disk_guard_rejects_estimate_over_free_margin(self) -> None:
        self.input_path.write_text(
            "C\tmol-1\nCC\tmol-2\nCCC\tmol-3\n", encoding="utf-8"
        )
        record_sizes = mock.Mock(side_effect=[100, 200, 300])
        with (
            mock.patch.object(runner, "_estimated_sdf_record_bytes", record_sizes),
            mock.patch.object(runner, "MIN_BYTES_PER_CONFORMER", 0),
            mock.patch.object(
                runner.shutil, "disk_usage", return_value=SimpleNamespace(free=1_600)
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "Insufficient disk space"):
                runner.check_disk_space(
                    self.output_path, self.input_path, n_conformers=2
                )

        self.assertEqual(record_sizes.call_count, 3)
        sampled_mean = (100 + 200 + 300) / 3
        bytes_per_conformer = math.ceil(sampled_mean * runner.DISK_SAFETY_FACTOR)
        self.assertEqual(3 * 2 * bytes_per_conformer, 1_500)
        self.assertGreater(1_500, 1_600 * runner.DISK_FREE_MARGIN)

    def test_disk_check_bypass_skips_preflight_and_per_chunk_guards(self) -> None:
        modules = fake_nvmolkit_modules(complete_embed, finite_energies)
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(Chem, "SDWriter", self.writer_class),
            mock.patch.object(runner, "check_disk_space") as preflight,
            mock.patch.object(runner, "_check_chunk_disk_space") as per_chunk,
        ):
            self.run_once(check_free_space=False)

        preflight.assert_not_called()
        per_chunk.assert_not_called()
        self.assertTrue(self.output_path.exists())


if __name__ == "__main__":
    unittest.main()
