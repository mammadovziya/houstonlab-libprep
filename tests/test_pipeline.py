from __future__ import annotations

import csv
import json
import math
import os
import stat
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

try:
    import pandas as pd

    import library_pipeline as pipeline
except ImportError:
    pd = None
    pipeline = None


@unittest.skipIf(pipeline is None, "chemistry dependencies are not installed")
class PipelineRegressionTests(unittest.TestCase):
    @staticmethod
    def _frame(smiles: str, mol_id: str = "mol_1"):
        return pd.DataFrame(
            [
                {
                    "ID": mol_id,
                    "SMILES": smiles,
                    "original_supplier_smiles": smiles,
                    "supplier": "test_supplier",
                }
            ]
        )

    def test_post_ionisation_dedup_preserves_all_provenance(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "ID": "supplier_a_1",
                    "SMILES": "CCN",
                    "original_supplier_smiles": "CCN",
                    "supplier": "supplier_a",
                },
                {
                    "ID": "supplier_b_9",
                    "SMILES": "NCC",
                    "original_supplier_smiles": "NCC",
                    "supplier": "supplier_b",
                },
            ]
        )
        result = pipeline.canonical_redup(frame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["ID"], "supplier_a_1;supplier_b_9")
        self.assertEqual(result.iloc[0]["supplier"], "supplier_a;supplier_b")

    def test_current_dimorphite_api_is_detected(self) -> None:
        self.assertIsNotNone(pipeline.dimorphite_protonate)
        variants = pipeline.dimorphite_protonate(
            "CCN", min_ph=7.4, max_ph=7.4, precision=0.0
        )
        self.assertTrue(variants)

    def test_path_validation_rejects_input_output_and_derived_output_collisions(self) -> None:
        parser = pipeline.build_parser()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shared_path = root / "shared.sdf"
            shared_path.write_text("CCO\tshared\n")
            derived_input = root / "library_final.smi"
            derived_input.write_text("CCO\tderived\n")

            cases = [
                (
                    "input equals output",
                    [
                        "--input",
                        str(shared_path),
                        "--output",
                        str(shared_path),
                    ],
                    "Planned output SDF path aliases input",
                ),
                (
                    "input equals derived final SMILES",
                    [
                        "--input",
                        str(derived_input),
                        "--output",
                        str(root / "library.sdf"),
                    ],
                    "Planned final SMILES path aliases input",
                ),
            ]

            for label, argv, message in cases:
                with self.subTest(label=label):
                    args = parser.parse_args(argv)
                    params = pipeline.resolve_params(args)
                    with self.assertRaisesRegex(ValueError, message):
                        pipeline._validate_pipeline_paths(args, params)

    def test_direct_conformer_api_rejects_hardlinked_planned_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.smi"
            output_path = root / "library.sdf"
            failure_path = root / "library_conformer_failures.csv"
            input_path.write_text("CCO\tmol_1\n")
            output_path.write_bytes(b"known-good-final")
            os.link(output_path, failure_path)

            with self.assertRaisesRegex(
                ValueError, "aliases planned output SDF"
            ):
                pipeline.generate_conformers(input_path, output_path)

            self.assertEqual(output_path.read_bytes(), b"known-good-final")

    def test_output_lock_serializes_all_artifacts_in_one_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = pipeline._OutputLock(root / "library.sdf")
            try:
                with self.assertRaisesRegex(RuntimeError, "already writing"):
                    pipeline._OutputLock(root / "library.mol")
            finally:
                first.close()

            second = pipeline._OutputLock(root / "library.mol")
            second.close()

    def test_staged_sdf_uses_normal_permissions_and_rejects_symlink_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_path = root / "library.sdf"
            previous_umask = os.umask(0o022)
            try:
                staged = pipeline._create_staged_sdf(output_path)
            finally:
                os.umask(previous_umask)
            try:
                self.assertEqual(stat.S_IMODE(staged.stat().st_mode), 0o644)
            finally:
                staged.unlink()

            restrictive_output = root / "restrictive.sdf"
            previous_umask = os.umask(0o333)
            try:
                restrictive_stage = pipeline._create_staged_sdf(
                    restrictive_output
                )
            finally:
                os.umask(previous_umask)
            try:
                self.assertTrue(
                    restrictive_stage.stat().st_mode & stat.S_IWUSR
                )
            finally:
                restrictive_stage.unlink()

            target = root / "target.sdf"
            target.touch()
            output_path.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                pipeline._validate_regular_output_path(output_path, "output SDF")

    def test_resolve_params_accepts_hardware_auto_sentinels_but_rejects_zero(self) -> None:
        parser = pipeline.build_parser()
        automatic = parser.parse_args(
            [
                "--input",
                "input.smi",
                "--batch-size",
                "-1",
                "--batches-per-gpu",
                "-1",
                "--preprocessing-threads",
                "-1",
            ]
        )

        pipeline.resolve_params(automatic)
        self.assertEqual(automatic.batch_size, -1)
        self.assertEqual(automatic.batches_per_gpu, -1)
        self.assertEqual(automatic.preprocessing_threads, -1)

        for flag in (
            "--batch-size",
            "--batches-per-gpu",
            "--preprocessing-threads",
        ):
            with self.subTest(flag=flag):
                args = parser.parse_args(["--input", "input.smi", flag, "0"])
                with self.assertRaisesRegex(ValueError, flag):
                    pipeline.resolve_params(args)

    def test_e_z_only_stereo_is_enumerated(self) -> None:
        result = pipeline.filter_and_enumerate_stereo(
            self._frame("CC=CC", "alkene"), max_unspecified=2
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(set(result["ID"]), {"alkene_iso1", "alkene_iso2"})
        self.assertEqual(set(result["SMILES"]), {"C/C=C/C", "C/C=C\\C"})

    def test_mixed_tetrahedral_and_e_z_stereo_is_fully_enumerated(self) -> None:
        mol = pipeline.Chem.MolFromSmiles("CC=CC(C)F")
        self.assertEqual(pipeline.count_unspecified_stereocentres(mol), 2)

        result = pipeline.filter_and_enumerate_stereo(
            self._frame("CC=CC(C)F", "mixed"), max_unspecified=2
        )

        self.assertEqual(len(result), 4)
        self.assertEqual(len(set(result["SMILES"])), 4)
        self.assertTrue(all("@" in smiles for smiles in result["SMILES"]))
        self.assertTrue(all("/" in smiles or "\\" in smiles for smiles in result["SMILES"]))

    def test_embed_propagates_the_requested_random_seed(self) -> None:
        captured = {}

        def fake_embed(mols, params, confsPerMolecule, hardwareOptions):
            captured["random_seed"] = params.randomSeed
            for mol in mols:
                for _ in range(confsPerMolecule):
                    mol.AddConformer(pipeline.Chem.Conformer(mol.GetNumAtoms()), assignId=True)

        def fake_optimize(mols, *args, **kwargs):
            return [[0.0] * mol.GetNumConformers() for mol in mols]

        class RecordingWriter:
            def write(self, mol, confId):
                return None

        mol = pipeline.Chem.AddHs(pipeline.Chem.MolFromSmiles("CCO"))
        with mock.patch.object(
            pipeline,
            "_import_nvmolkit",
            return_value=(fake_embed, fake_optimize, object),
        ):
            pipeline._embed_and_write(
                [mol],
                RecordingWriter(),
                n_conformers=2,
                mmff_max_iters=20,
                hw=object(),
                random_seed=123456,
            )

        self.assertEqual(captured["random_seed"], 123456)

    def test_retry_seed_wraps_to_zero_at_the_maximum_allowed_base_seed(self) -> None:
        seeds = []

        def fake_embed(mols, params, confsPerMolecule, hardwareOptions):
            seeds.append(params.randomSeed)
            if len(seeds) == 2:
                for mol in mols:
                    for _ in range(confsPerMolecule):
                        mol.AddConformer(
                            pipeline.Chem.Conformer(mol.GetNumAtoms()), assignId=True
                        )

        def fake_optimize(mols, *args, **kwargs):
            return [[0.0] * mol.GetNumConformers() for mol in mols]

        class RecordingWriter:
            def write(self, mol, confId):
                return None

        mol = pipeline.Chem.AddHs(pipeline.Chem.MolFromSmiles("CCO"))
        mol.SetProp("_Name", "seed_wrap")
        with mock.patch.object(
            pipeline,
            "_import_nvmolkit",
            return_value=(fake_embed, fake_optimize, object),
        ):
            result = pipeline._embed_and_write(
                [mol],
                RecordingWriter(),
                n_conformers=1,
                mmff_max_iters=20,
                hw=object(),
                retry_hw=object(),
                random_seed=2**31 - 2,
            )

        self.assertEqual(seeds, [2**31 - 2, 0])
        self.assertEqual(result["retried_mols"], 1)
        self.assertEqual(result["successful_mols"], 1)

    def test_primary_embed_exception_is_fatal_even_after_creating_all_conformers(self) -> None:
        failure_records = []

        def fake_embed(mols, params, confsPerMolecule, hardwareOptions):
            for mol in mols:
                for _ in range(confsPerMolecule):
                    mol.AddConformer(
                        pipeline.Chem.Conformer(mol.GetNumAtoms()), assignId=True
                    )
            raise ValueError("primary exploded after mutation")

        class RecordingWriter:
            def __init__(self):
                self.writes = []

            def write(self, mol, confId):
                self.writes.append((mol.GetProp("_Name"), confId))

        mol = pipeline.Chem.AddHs(pipeline.Chem.MolFromSmiles("CCO"))
        mol.SetProp("_Name", "mutated_then_failed")
        writer = RecordingWriter()
        with mock.patch.object(
            pipeline,
            "_import_nvmolkit",
            return_value=(fake_embed, object(), object),
        ):
            with self.assertRaisesRegex(RuntimeError, "primary embedding failed"):
                pipeline._embed_and_write(
                    [mol],
                    writer,
                    n_conformers=2,
                    mmff_max_iters=20,
                    hw=object(),
                    conformer_failure_records=failure_records,
                )

        self.assertEqual(mol.GetNumConformers(), 2)
        self.assertEqual(writer.writes, [])
        self.assertEqual(
            failure_records,
            [
                {
                    "ID": "mutated_then_failed",
                    "reason": (
                        "primary_embedding_error:ValueError:"
                        "primary exploded after mutation"
                    ),
                    "requested": 2,
                    "generated": 2,
                    "retry_attempted": False,
                }
            ],
        )

    def test_mmff94s_properties_are_passed_to_the_optimizer(self) -> None:
        sentinel_properties = object()
        captured = {}

        def fake_embed(mols, params, confsPerMolecule, hardwareOptions):
            for mol in mols:
                mol.AddConformer(pipeline.Chem.Conformer(mol.GetNumAtoms()), assignId=True)

        def fake_optimize(mols, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return [[1.25] * mol.GetNumConformers() for mol in mols]

        class RecordingWriter:
            def write(self, mol, confId):
                return None

        mol = pipeline.Chem.AddHs(pipeline.Chem.MolFromSmiles("CCO"))
        with (
            mock.patch.object(
                pipeline,
                "_import_nvmolkit",
                return_value=(fake_embed, fake_optimize, object),
            ),
            mock.patch.object(
                pipeline.AllChem,
                "MMFFGetMoleculeProperties",
                return_value=sentinel_properties,
            ) as get_properties,
        ):
            pipeline._embed_and_write(
                [mol],
                RecordingWriter(),
                n_conformers=1,
                mmff_max_iters=20,
                hw=object(),
                random_seed=7,
            )

        get_properties.assert_called_once_with(mol, mmffVariant="MMFF94s")
        optimizer_values = [*captured["args"], *captured["kwargs"].values()]
        self.assertTrue(
            any(
                value is sentinel_properties
                or (
                    isinstance(value, (list, tuple))
                    and any(item is sentinel_properties for item in value)
                )
                for value in optimizer_values
            ),
            "MMFF94s properties were computed but not passed to the optimizer",
        )

    def test_mmff_energy_count_mismatch_and_nonfinite_energy_are_fatal(self) -> None:
        cases = [
            ("count mismatch", [[0.0]], "different number of energies"),
            ("nonfinite", [[0.0, float("nan")]], "non-finite energy"),
        ]

        for label, energies, message in cases:
            with self.subTest(label=label):
                def fake_embed(mols, params, confsPerMolecule, hardwareOptions):
                    for mol in mols:
                        for _ in range(confsPerMolecule):
                            mol.AddConformer(
                                pipeline.Chem.Conformer(mol.GetNumAtoms()),
                                assignId=True,
                            )

                def fake_optimize(mols, *args, **kwargs):
                    return energies

                class RecordingWriter:
                    def __init__(self):
                        self.writes = []

                    def write(self, mol, confId):
                        self.writes.append((mol.GetProp("_Name"), confId))

                mol = pipeline.Chem.AddHs(pipeline.Chem.MolFromSmiles("CCO"))
                mol.SetProp("_Name", label)
                writer = RecordingWriter()
                with mock.patch.object(
                    pipeline,
                    "_import_nvmolkit",
                    return_value=(fake_embed, fake_optimize, object),
                ):
                    with self.assertRaisesRegex(RuntimeError, message):
                        pipeline._embed_and_write(
                            [mol],
                            writer,
                            n_conformers=2,
                            mmff_max_iters=20,
                            hw=object(),
                        )

                self.assertEqual(writer.writes, [])

    def test_partial_conformer_batch_is_retried_and_recovers(self) -> None:
        seeds = []

        def fake_embed(mols, params, confsPerMolecule, hardwareOptions):
            seeds.append(params.randomSeed)
            generated = 1 if len(seeds) == 1 else confsPerMolecule
            for mol in mols:
                for _ in range(generated):
                    mol.AddConformer(pipeline.Chem.Conformer(mol.GetNumAtoms()), assignId=True)

        def fake_optimize(mols, *args, **kwargs):
            return [[0.0] * mol.GetNumConformers() for mol in mols]

        class RecordingWriter:
            def __init__(self):
                self.writes = []

            def write(self, mol, confId):
                self.writes.append((mol.GetProp("_Name"), confId))

        mol = pipeline.Chem.AddHs(pipeline.Chem.MolFromSmiles("CCO"))
        mol.SetProp("_Name", "retry_me")
        writer = RecordingWriter()
        with mock.patch.object(
            pipeline,
            "_import_nvmolkit",
            return_value=(fake_embed, fake_optimize, object),
        ):
            result = pipeline._embed_and_write(
                [mol],
                writer,
                n_conformers=2,
                mmff_max_iters=20,
                hw=object(),
                retry_hw=object(),
                random_seed=100,
            )

        self.assertEqual(seeds, [100, 101])
        self.assertEqual(result["retried_mols"], 1)
        self.assertEqual(result["successful_mols"], 1)
        self.assertEqual(result["failed_mols"], 0)
        self.assertEqual(result["conformer_shortfall"], 0)
        self.assertEqual(result["unwritten_conformers"], 0)
        self.assertEqual(result["policy_dropped_conformers"], 0)
        self.assertEqual(result["failures"], [])
        self.assertEqual(result["confs"], 2)
        self.assertEqual(writer.writes, [("retry_me", 0), ("retry_me", 1)])

    def test_persistent_partial_conformer_is_detected_and_only_written_when_allowed(self) -> None:
        def run(allow_partial):
            attempts = []

            def fake_embed(mols, params, confsPerMolecule, hardwareOptions):
                attempts.append(params.randomSeed)
                if len(attempts) == 1:
                    for mol in mols:
                        mol.AddConformer(
                            pipeline.Chem.Conformer(mol.GetNumAtoms()), assignId=True
                        )

            def fake_optimize(mols, *args, **kwargs):
                return [[0.0] * mol.GetNumConformers() for mol in mols]

            class RecordingWriter:
                def __init__(self):
                    self.writes = []

                def write(self, mol, confId):
                    self.writes.append((mol.GetProp("_Name"), confId))

            mol = pipeline.Chem.AddHs(pipeline.Chem.MolFromSmiles("CCO"))
            mol.SetProp("_Name", "still_partial")
            writer = RecordingWriter()
            with mock.patch.object(
                pipeline,
                "_import_nvmolkit",
                return_value=(fake_embed, fake_optimize, object),
            ):
                result = pipeline._embed_and_write(
                    [mol],
                    writer,
                    n_conformers=2,
                    mmff_max_iters=20,
                    hw=object(),
                    retry_hw=object(),
                    random_seed=200,
                    allow_partial_conformers=allow_partial,
                )
            return result, writer.writes, attempts

        strict_result, strict_writes, strict_attempts = run(False)
        allowed_result, allowed_writes, allowed_attempts = run(True)

        expected_failure = {
            "ID": "still_partial",
            "reason": "conformer_shortfall",
            "requested": 2,
            "generated": 1,
            "retry_attempted": True,
        }
        self.assertEqual(strict_result["failures"], [expected_failure])
        self.assertEqual(strict_result["failed_mols"], 1)
        self.assertEqual(strict_result["conformer_shortfall"], 1)
        self.assertEqual(strict_result["unwritten_conformers"], 2)
        self.assertEqual(strict_result["policy_dropped_conformers"], 1)
        self.assertEqual(strict_result["confs"], 0)
        self.assertEqual(strict_writes, [])
        self.assertEqual(strict_attempts, [200, 201])
        self.assertEqual(allowed_result["failures"], [expected_failure])
        self.assertEqual(allowed_result["unwritten_conformers"], 1)
        self.assertEqual(allowed_result["policy_dropped_conformers"], 0)
        self.assertEqual(allowed_result["confs"], 1)
        self.assertEqual(allowed_writes, [("still_partial", 0)])
        self.assertEqual(allowed_attempts, [200, 201])

    def test_generate_conformers_strict_raises_but_allow_returns_partial_totals(self) -> None:
        partial_result = {
            "confs": 1,
            "successful_mols": 0,
            "failed_mols": 1,
            "conformer_shortfall": 1,
            "unwritten_conformers": 1,
            "policy_dropped_conformers": 0,
            "retried_mols": 1,
            "failures": [
                {
                    "ID": "partial",
                    "reason": "conformer_shortfall",
                    "requested": 2,
                    "generated": 1,
                    "retry_attempted": True,
                }
            ],
            "mmff_skipped": 0,
            "mmff_skipped_ids": [],
        }

        class FakeHardwareOptions:
            def __init__(self, **kwargs):
                self.options = kwargs

        class FakeWriter:
            def __init__(self, path):
                self.path = path
                self.closed = False
                Path(path).touch()

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.smi"
            input_path.write_text("CCO\tpartial\n")

            def run(allow_partial, output_name):
                with (
                    mock.patch.object(
                        pipeline,
                        "_import_nvmolkit",
                        return_value=(object(), object(), FakeHardwareOptions),
                    ),
                    mock.patch.object(pipeline.Chem, "SDWriter", side_effect=FakeWriter),
                    mock.patch.object(
                        pipeline, "_embed_and_write", return_value=partial_result
                    ) as embed_and_write,
                ):
                    result = pipeline.generate_conformers(
                        input_path,
                        root / output_name,
                        n_conformers=2,
                        chunk_size=10,
                        random_seed=4242,
                        allow_partial_conformers=allow_partial,
                        check_free_space=False,
                    )
                return result, embed_and_write

            with self.assertRaisesRegex(RuntimeError, "without all 2 requested conformers"):
                run(False, "strict.sdf")

            totals, embed_and_write = run(True, "allowed.sdf")

        self.assertTrue(embed_and_write.call_args.kwargs["allow_partial_conformers"])
        self.assertEqual(embed_and_write.call_args.kwargs["random_seed"], 4242)
        self.assertEqual(totals["failed_mols"], 1)
        self.assertEqual(totals["conformer_shortfall"], 1)
        self.assertEqual(totals["unwritten_conformers"], 1)
        self.assertEqual(totals["policy_dropped_conformers"], 0)
        self.assertEqual(totals["confs"], 1)
        self.assertEqual(totals["failures"], partial_result["failures"])

    def test_allow_partial_all_invalid_input_reaches_parse_report_with_disk_check(self) -> None:
        class FakeHardwareOptions:
            def __init__(self, **kwargs):
                self.options = kwargs

        class FakeWriter:
            def __init__(self, path):
                Path(path).touch()

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "invalid.smi"
            input_path.write_text("not_a_smiles\tbad_1\n")
            output_path = root / "library.sdf"
            with (
                mock.patch.object(
                    pipeline,
                    "_import_nvmolkit",
                    return_value=(object(), object(), FakeHardwareOptions),
                ),
                mock.patch.object(pipeline.Chem, "SDWriter", side_effect=FakeWriter),
            ):
                totals = pipeline.generate_conformers(
                    input_path,
                    output_path,
                    allow_partial_conformers=True,
                )

            with open(totals["failure_report"], newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(totals["failure_count"], 1)
            self.assertEqual(totals["parse_fail"], 1)
            self.assertEqual(totals["conformer_shortfall"], 1)
            self.assertEqual(rows[0]["ID"], "bad_1")

    def test_direct_conformer_api_rejects_invalid_runtime_values_early(self) -> None:
        with self.assertRaisesRegex(ValueError, "n_conformers"):
            pipeline.generate_conformers("input.smi", "output.sdf", n_conformers=0)
        with self.assertRaisesRegex(ValueError, "estimated_bytes_per_conformer"):
            pipeline.generate_conformers(
                "input.smi", "output.sdf", estimated_bytes_per_conformer=0
            )

    def test_empty_conformer_input_cannot_replace_existing_final(self) -> None:
        class FakeHardwareOptions:
            def __init__(self, **kwargs):
                self.options = kwargs

        class FakeWriter:
            def __init__(self, path):
                self.path = Path(path)

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "empty.smi"
            input_path.touch()
            output_path = root / "library.sdf"
            output_path.write_bytes(b"known-good-final")
            with (
                mock.patch.object(
                    pipeline,
                    "_import_nvmolkit",
                    return_value=(object(), object(), FakeHardwareOptions),
                ),
                mock.patch.object(pipeline.Chem, "SDWriter", side_effect=FakeWriter),
            ):
                with self.assertRaisesRegex(RuntimeError, "no molecule records"):
                    pipeline.generate_conformers(
                        input_path,
                        output_path,
                        allow_partial_conformers=True,
                        check_free_space=False,
                    )

            self.assertEqual(output_path.read_bytes(), b"known-good-final")

    def test_strict_conformer_failure_preserves_existing_final_and_uses_partial_sdf(self) -> None:
        partial_result = {
            "confs": 0,
            "successful_mols": 0,
            "failed_mols": 1,
            "conformer_shortfall": 1,
            "unwritten_conformers": 2,
            "policy_dropped_conformers": 1,
            "retried_mols": 1,
            "failures": [
                {
                    "ID": "partial",
                    "reason": "conformer_shortfall",
                    "requested": 2,
                    "generated": 1,
                    "retry_attempted": True,
                }
            ],
            "mmff_skipped": 0,
            "mmff_skipped_ids": [],
        }

        class FakeHardwareOptions:
            def __init__(self, **kwargs):
                self.options = kwargs

        class FakeWriter:
            paths = []

            def __init__(self, path):
                self.path = Path(path)
                self.paths.append(self.path)
                self.path.touch()

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.smi"
            input_path.write_text("CCO\tpartial\n")
            output_path = root / "library.sdf"
            output_path.write_bytes(b"previous-complete-sdf")

            with (
                mock.patch.object(
                    pipeline,
                    "_import_nvmolkit",
                    return_value=(object(), object(), FakeHardwareOptions),
                ),
                mock.patch.object(pipeline.Chem, "SDWriter", side_effect=FakeWriter),
                mock.patch.object(
                    pipeline, "_embed_and_write", return_value=partial_result
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "without all 2 requested conformers"
                ):
                    pipeline.generate_conformers(
                        input_path,
                        output_path,
                        n_conformers=2,
                        allow_partial_conformers=False,
                        check_free_space=False,
                    )

            self.assertEqual(output_path.read_bytes(), b"previous-complete-sdf")
            self.assertEqual(len(FakeWriter.paths), 1)
            partial_path = FakeWriter.paths[0]
            self.assertEqual(partial_path.parent, root)
            self.assertTrue(partial_path.name.startswith("library."))
            self.assertTrue(partial_path.name.endswith(".partial.sdf"))
            self.assertTrue(partial_path.is_file())

    def test_precommit_validation_failure_preserves_existing_final_sdf(self) -> None:
        complete_result = {
            "confs": 1,
            "successful_mols": 1,
            "failed_mols": 0,
            "conformer_shortfall": 0,
            "unwritten_conformers": 0,
            "policy_dropped_conformers": 0,
            "retried_mols": 0,
            "failures": [],
            "mmff_skipped": 0,
            "mmff_skipped_ids": [],
        }

        class FakeHardwareOptions:
            def __init__(self, **kwargs):
                self.options = kwargs

        class FakeWriter:
            paths = []

            def __init__(self, path):
                self.path = Path(path)
                self.paths.append(self.path)
                self.path.write_bytes(b"new-staged-sdf")

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.smi"
            input_path.write_text("CCO\tmol_1\n")
            output_path = root / "library.sdf"
            output_path.write_bytes(b"previous-complete-sdf")

            with (
                mock.patch.object(
                    pipeline,
                    "_import_nvmolkit",
                    return_value=(object(), object(), FakeHardwareOptions),
                ),
                mock.patch.object(pipeline.Chem, "SDWriter", side_effect=FakeWriter),
                mock.patch.object(
                    pipeline, "_embed_and_write", return_value=complete_result
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "input changed"):
                    pipeline.generate_conformers(
                        input_path,
                        output_path,
                        check_free_space=False,
                        before_commit=lambda: (_ for _ in ()).throw(
                            RuntimeError("input changed")
                        ),
                    )

            self.assertEqual(output_path.read_bytes(), b"previous-complete-sdf")
            self.assertEqual(len(FakeWriter.paths), 1)
            self.assertTrue(FakeWriter.paths[0].is_file())

    def test_read_only_existing_final_is_replaced_and_keeps_its_mode(self) -> None:
        complete_result = {
            "confs": 1,
            "successful_mols": 1,
            "failed_mols": 0,
            "conformer_shortfall": 0,
            "unwritten_conformers": 0,
            "policy_dropped_conformers": 0,
            "retried_mols": 0,
            "failures": [],
            "mmff_skipped": 0,
            "mmff_skipped_ids": [],
        }

        class FakeHardwareOptions:
            def __init__(self, **kwargs):
                self.options = kwargs

        class FakeWriter:
            def __init__(self, path):
                self.path = Path(path)
                self.path.write_bytes(b"replacement-sdf")

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.smi"
            input_path.write_text("CCO\tmol_1\n")
            output_path = root / "library.sdf"
            output_path.write_bytes(b"old-sdf")
            output_path.chmod(0o444)

            with (
                mock.patch.object(
                    pipeline,
                    "_import_nvmolkit",
                    return_value=(object(), object(), FakeHardwareOptions),
                ),
                mock.patch.object(pipeline.Chem, "SDWriter", side_effect=FakeWriter),
                mock.patch.object(
                    pipeline, "_embed_and_write", return_value=complete_result
                ),
            ):
                pipeline.generate_conformers(
                    input_path,
                    output_path,
                    check_free_space=False,
                )

            self.assertEqual(output_path.read_bytes(), b"replacement-sdf")
            self.assertEqual(stat.S_IMODE(output_path.stat().st_mode), 0o444)

    def test_streamed_failure_sink_bounds_examples_but_reports_every_failure(self) -> None:
        total_failures = pipeline.FAILURE_EXAMPLE_LIMIT + 7

        class FakeHardwareOptions:
            def __init__(self, **kwargs):
                self.options = kwargs

        class FakeWriter:
            def __init__(self, path):
                self.path = Path(path)
                self.path.touch()

            def close(self):
                return None

        def reject_every_row(rows):
            return [], [
                {
                    "ID": mol_id,
                    "reason": "parse_failed",
                    "requested": 0,
                    "generated": 0,
                    "retry_attempted": False,
                }
                for _, mol_id in rows
            ]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.smi"
            input_path.write_text(
                "".join(f"CCO\tfailure_{index}\n" for index in range(total_failures))
            )
            output_path = root / "library.sdf"
            with (
                mock.patch.object(
                    pipeline,
                    "_import_nvmolkit",
                    return_value=(object(), object(), FakeHardwareOptions),
                ),
                mock.patch.object(pipeline.Chem, "SDWriter", side_effect=FakeWriter),
                mock.patch.object(pipeline, "_load_chunk", side_effect=reject_every_row),
            ):
                totals = pipeline.generate_conformers(
                    input_path,
                    output_path,
                    n_conformers=1,
                    chunk_size=13,
                    allow_partial_conformers=True,
                    check_free_space=False,
                )

            with open(totals["failure_report"], newline="") as handle:
                report_rows = list(csv.DictReader(handle))

            self.assertEqual(totals["failure_count"], total_failures)
            self.assertEqual(len(totals["failures"]), pipeline.FAILURE_EXAMPLE_LIMIT)
            self.assertTrue(totals["failures_truncated"])
            self.assertEqual(len(report_rows), total_failures)
            self.assertEqual(report_rows[0]["ID"], "failure_0")
            self.assertEqual(
                report_rows[-1]["ID"], f"failure_{total_failures - 1}"
            )

    def test_dynamic_disk_estimate_uses_sampled_accepted_molecule_size(self) -> None:
        accepted_smiles = "CCOC(=O)N1CCC(c2ccccc2)CC1"
        record_bytes = pipeline._estimated_sdf_record_bytes(
            accepted_smiles, "accepted"
        )
        self.assertIsNotNone(record_bytes)

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "accepted.smi"
            input_path.write_text(f"{accepted_smiles}\taccepted\n")
            with mock.patch.object(pipeline, "MIN_BYTES_PER_CONFORMER", 1):
                estimate = pipeline.estimate_sdf_bytes(
                    input_path,
                    n_conformers=3,
                    sample_size=1,
                    safety_factor=1.25,
                )

        expected_per_conformer = math.ceil(record_bytes * 1.25)
        self.assertEqual(estimate["molecule_count"], 1)
        self.assertEqual(estimate["sample_count"], 1)
        self.assertEqual(estimate["sampled_mean_bytes"], record_bytes)
        self.assertEqual(estimate["bytes_per_conformer"], expected_per_conformer)
        self.assertEqual(
            estimate["estimated_total_bytes"], 3 * expected_per_conformer
        )

    def test_disk_guard_rejects_estimate_above_ninety_percent_of_free_space(self) -> None:
        estimate = {
            "molecule_count": 1,
            "sample_count": 1,
            "sampled_mean_bytes": 721,
            "bytes_per_conformer": 901,
            "estimated_total_bytes": 901,
            "safety_factor": 1.25,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with (
                mock.patch.object(
                    pipeline, "estimate_sdf_bytes", return_value=estimate
                ) as estimate_sdf_bytes,
                mock.patch.object(
                    pipeline.shutil,
                    "disk_usage",
                    return_value=Namespace(free=1_000),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "Insufficient disk space"):
                    pipeline.check_disk_space(
                        root / "library.sdf",
                        root / "library.smi",
                        n_conformers=1,
                        margin=0.90,
                        sample_size=1,
                    )

        estimate_sdf_bytes.assert_called_once_with(
            root / "library.smi", 1, sample_size=1
        )

    def test_skip_conformers_manifest_does_not_claim_an_sdf(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.smi"
            input_path.write_text("CCOC(=O)N1CCC(c2ccccc2)CC1\tmol_1\n")
            output_path = root / "library.sdf"
            manifest_path = root / "library_manifest.json"
            args = Namespace(
                input=[str(input_path)],
                output=str(output_path),
                preset="docking",
                skip_conformers=True,
            )

            pipeline.write_manifest(
                manifest_path,
                args,
                params={"skip_conformers": True},
                counts={"final_2d": 1},
                timings={"total": 0.0},
                hash_inputs=False,
                artifacts={"final_smiles": input_path, "output_sdf": None},
            )

            manifest = json.loads(manifest_path.read_text())
            self.assertIsNone(manifest.get("output_sdf"))
            self.assertNotIn("output_sdf", manifest["artifacts"])
            self.assertNotIn(
                str(output_path.resolve()), json.dumps(manifest["artifacts"])
            )

    def test_run_marker_is_in_progress_and_success_manifest_is_succeeded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.smi"
            input_path.write_text("CCO\tmol_1\n")
            output_path = root / "library.sdf"
            output_path.touch()
            manifest_path = root / "library_manifest.json"
            args = Namespace(
                input=[str(input_path)],
                output=str(output_path),
                preset="docking",
                skip_conformers=False,
                random_seed=123,
                allow_partial_conformers=False,
            )
            params = dict(pipeline.PRESETS["docking"])

            pipeline.write_run_marker(manifest_path, args, params)
            marker = json.loads(manifest_path.read_text())
            self.assertEqual(marker["status"], "in_progress")
            self.assertIsNone(marker["output_sdf"])
            self.assertEqual(
                marker["planned_output_sdf"], str(output_path.resolve())
            )

            pipeline.write_manifest(
                manifest_path,
                args,
                params=params,
                counts={"final_2d": 1, "conformers_written": 1},
                timings={"total": 0.0},
                hash_inputs=False,
                artifacts={"output_sdf": output_path},
            )
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["status"], "succeeded")
            self.assertEqual(manifest["output_sdf"], str(output_path.resolve()))

    def test_success_manifest_refuses_changed_start_of_run_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.smi"
            input_path.write_text("CCO\tmol_1\n")
            provenance = pipeline.capture_input_provenance(
                [input_path], hash_inputs=True
            )
            input_path.write_text("CCCC\tchanged\n")

            output_path = root / "library.sdf"
            output_path.touch()
            manifest_path = root / "library_manifest.json"
            args = Namespace(
                input=[str(input_path)],
                output=str(output_path),
                preset="docking",
                skip_conformers=False,
            )
            with self.assertRaisesRegex(RuntimeError, "Input changed"):
                pipeline.write_manifest(
                    manifest_path,
                    args,
                    params={},
                    counts={},
                    timings={},
                    artifacts={"output_sdf": output_path},
                    input_provenance=provenance,
                )

            self.assertFalse(manifest_path.exists())

    def test_supplier_read_must_match_captured_input_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.smi"
            input_path.write_text("CCO\tmol_1\n")
            provenance = pipeline.capture_input_provenance(
                [input_path], hash_inputs=True
            )
            input_path.write_text("CCCC\tchanged\n")

            with self.assertRaisesRegex(RuntimeError, "Input changed"):
                pipeline.merge_suppliers([input_path], provenance)

    def test_derived_smiles_read_must_match_captured_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "final.smi"
            input_path.write_text("CCO\tmol_1\n")
            provenance = pipeline.capture_input_provenance(
                [input_path], hash_inputs=True
            )[0]
            input_path.write_text("CCCC\tchanged\n")

            with self.assertRaisesRegex(RuntimeError, "Input changed"):
                list(
                    pipeline._iter_smiles_file(
                        input_path, expected_provenance=provenance
                    )
                )

    def test_failed_final_validation_preserves_existing_2d_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.smi"
            input_path.write_text(
                "CCN(CC)CCOC1=CC=C(C=C1)C2=CC=CC=C2\tmol_1\n"
            )
            output_path = root / "library.sdf"
            final_smi = root / "library_final.smi"
            final_metadata = root / "library_final_metadata.csv"
            final_smi.write_bytes(b"previous-final-smiles")
            final_metadata.write_bytes(b"previous-final-metadata")
            argv = [
                "library_pipeline.py",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--skip-conformers",
                "--skip-ionise",
                "--pains-backend",
                "cpu",
            ]

            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    pipeline,
                    "verify_input_provenance",
                    side_effect=RuntimeError("forced final validation failure"),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "forced final validation failure"
                ):
                    pipeline.main()

            self.assertEqual(final_smi.read_bytes(), b"previous-final-smiles")
            self.assertEqual(
                final_metadata.read_bytes(), b"previous-final-metadata"
            )

    def test_repointed_supplier_symlink_cannot_bypass_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.smi"
            second = root / "second.smi"
            link = root / "input.smi"
            first.write_text("CCO\tfrom_first\n")
            second.write_text("CCCC\tfrom_second\n")
            link.symlink_to(first)
            provenance = pipeline.capture_input_provenance(
                [link], hash_inputs=True
            )

            link.unlink()
            link.symlink_to(second)

            with self.assertRaisesRegex(RuntimeError, "Input changed"):
                pipeline.merge_suppliers([link], provenance)


if __name__ == "__main__":
    unittest.main()
