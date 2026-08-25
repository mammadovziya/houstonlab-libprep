"""Memory-bounded GPU conformer generation for large libraries.

Reads a tab-delimited SMILES file produced by library_pipeline.py
(--skip-conformers --save-intermediates), processes molecules in fixed-size
chunks, and streams conformers to the output SDF. Each chunk's RDKit mol
objects are freed before the next chunk is loaded, keeping peak RAM at
~chunk-size molecules regardless of total input size.

Validated at 2.6M molecules (≈1M input after stereo/tautomer/ionisation
expansion) producing 24.7M conformers in 3h 39m on 1× NVIDIA RTX 5090.
"""

import argparse
import csv
import fcntl
import gc
import hashlib
import math
import os
import random
import re
import secrets
import shutil
import stat
import sys
import time
from pathlib import Path

_CXSMILES_EXT_RE = re.compile(r'\s*\|[^|]*\|\s*$')
DEFAULT_RANDOM_SEED = 0xF00D
MIN_BYTES_PER_CONFORMER = 4_500
DISK_SAMPLE_SIZE = 1_000
DISK_SAFETY_FACTOR = 1.25
DISK_FREE_MARGIN = 0.90
FAILURE_COLUMNS = (
    "chunk",
    "molecule_id",
    "smiles",
    "stage",
    "reason",
    "requested",
    "generated",
    "primary_conformers",
    "retry_conformers",
)
MMFF_FAILURE_COLUMNS = (
    "ID",
    "reason",
)


def _load_chunk(rows, failure_records=None, chunk_idx=None, n_conformers=None):
    """Convert a list of (smiles, mol_id) pairs into AddHs'd RDKit mols."""
    from rdkit import Chem

    mols = []
    parse_fail = 0
    for smiles, mol_id in rows:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            parse_fail += 1
            if failure_records is not None:
                failure_records.append({
                    "chunk": chunk_idx if chunk_idx is not None else "",
                    "molecule_id": str(mol_id),
                    "smiles": smiles,
                    "stage": "parse",
                    "reason": "invalid_smiles",
                    "requested": n_conformers if n_conformers is not None else "",
                    "generated": 0,
                    "primary_conformers": 0,
                    "retry_conformers": 0,
                })
            continue
        m = Chem.AddHs(m)
        m.SetProp("_Name", str(mol_id))
        m.SetProp("_InputSMILES", smiles, computed=True)
        mols.append(m)
    return mols, parse_fail


def _embed_params(random_seed):
    from rdkit.Chem.rdDistGeom import ETKDGv3

    params = ETKDGv3()
    params.useRandomCoords = True
    params.randomSeed = random_seed
    return params


def _failure_reason(primary_error, retry_error):
    reasons = ["fewer_than_requested"]
    if primary_error is not None:
        reasons.append(f"primary_error={type(primary_error).__name__}:{primary_error}")
    if retry_error is not None:
        reasons.append(f"retry_error={type(retry_error).__name__}:{retry_error}")
    return ";".join(reasons)


def _embed_and_write(mols, writer, n_conformers, mmff_max_iters,
                     batch_size, batches_per_gpu, preprocessing_threads,
                     random_seed=DEFAULT_RANDOM_SEED, failure_records=None,
                     chunk_idx=None, allow_partial_conformers=False,
                     mmff_failure_records=None):
    """Embed, retry shortfalls once, MMFF94s-minimise, and write conformers."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from nvmolkit.embedMolecules import EmbedMolecules
    from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs
    from nvmolkit.types import HardwareOptions

    hw = HardwareOptions(
        preprocessingThreads=preprocessing_threads,
        batchSize=batch_size,
        batchesPerGpu=batches_per_gpu,
        gpuIds=[],
    )

    primary_error = None
    try:
        EmbedMolecules(
            mols,
            _embed_params(random_seed),
            confsPerMolecule=n_conformers,
            hardwareOptions=hw,
        )
    except Exception as exc:
        if failure_records is not None:
            reason = f"primary_error={type(exc).__name__}:{exc}"
            for m in mols:
                generated = m.GetNumConformers()
                failure_records.append({
                    "chunk": chunk_idx if chunk_idx is not None else "",
                    "molecule_id": m.GetProp("_Name") if m.HasProp("_Name") else "",
                    "smiles": m.GetProp("_InputSMILES") if m.HasProp("_InputSMILES") else "",
                    "stage": "embedding",
                    "reason": reason,
                    "requested": n_conformers,
                    "generated": generated,
                    "primary_conformers": generated,
                    "retry_conformers": "",
                })
        raise RuntimeError(
            "nvMolKit primary embedding failed; no conformers from this batch "
            "were accepted"
        ) from exc

    primary_counts = [m.GetNumConformers() for m in mols]
    retry_indices = [
        idx for idx, count in enumerate(primary_counts) if count < n_conformers
    ]
    retry_counts = [None] * len(mols)
    retry_error = None
    retry_seed = (random_seed + 1) % (2**31 - 1)

    if retry_indices:
        retry_mols = []
        for idx in retry_indices:
            retry_mol = Chem.Mol(mols[idx])
            retry_mol.RemoveAllConformers()
            retry_mols.append(retry_mol)

        retry_hw = HardwareOptions(
            preprocessingThreads=1,
            batchSize=1,
            batchesPerGpu=1,
            gpuIds=[],
        )
        try:
            EmbedMolecules(
                retry_mols,
                _embed_params(retry_seed),
                confsPerMolecule=n_conformers,
                hardwareOptions=retry_hw,
            )
        except Exception as exc:
            if failure_records is not None:
                reason = f"retry_error={type(exc).__name__}:{exc}"
                for idx, retry_mol in zip(retry_indices, retry_mols):
                    retry_count = retry_mol.GetNumConformers()
                    failure_records.append({
                        "chunk": chunk_idx if chunk_idx is not None else "",
                        "molecule_id": (
                            mols[idx].GetProp("_Name")
                            if mols[idx].HasProp("_Name") else ""
                        ),
                        "smiles": (
                            mols[idx].GetProp("_InputSMILES")
                            if mols[idx].HasProp("_InputSMILES") else ""
                        ),
                        "stage": "embedding",
                        "reason": reason,
                        "requested": n_conformers,
                        "generated": max(primary_counts[idx], retry_count),
                        "primary_conformers": primary_counts[idx],
                        "retry_conformers": retry_count,
                    })
            raise RuntimeError(
                "nvMolKit conservative embedding retry failed; no conformers "
                "from this batch were accepted"
            ) from exc

        for idx, retry_mol in zip(retry_indices, retry_mols):
            retry_count = retry_mol.GetNumConformers()
            retry_counts[idx] = retry_count
            if retry_count > primary_counts[idx]:
                mols[idx] = retry_mol

        recovered = sum(
            mols[idx].GetNumConformers() >= n_conformers
            for idx in retry_indices
        )
        print(f"  Retried {len(retry_indices):,} incomplete molecules "
              f"with batch-size=1; recovered {recovered:,}")

    incomplete = 0
    for idx, m in enumerate(mols):
        final_count = m.GetNumConformers()
        if final_count >= n_conformers:
            continue
        incomplete += 1
        if failure_records is not None:
            failure_records.append({
                "chunk": chunk_idx if chunk_idx is not None else "",
                "molecule_id": m.GetProp("_Name") if m.HasProp("_Name") else "",
                "smiles": m.GetProp("_InputSMILES") if m.HasProp("_InputSMILES") else "",
                "stage": "embedding",
                "reason": _failure_reason(primary_error, retry_error),
                "requested": n_conformers,
                "generated": final_count,
                "primary_conformers": primary_counts[idx],
                "retry_conformers": (
                    retry_counts[idx] if retry_counts[idx] is not None else ""
                ),
            })

    write_mols = [
        m
        for m in mols
        if m.GetNumConformers() > 0
        and (
            allow_partial_conformers
            or m.GetNumConformers() >= n_conformers
        )
    ]

    mmff_ok, mmff_properties, mmff_bad = [], [], []
    for m in write_mols:
        properties = AllChem.MMFFGetMoleculeProperties(
            m, mmffVariant="MMFF94s"
        )
        if properties is None:
            mmff_bad.append(m)
            if mmff_failure_records is not None:
                mmff_failure_records.append({
                    "ID": (
                        m.GetProp("_Name") if m.HasProp("_Name") else ""
                    ),
                    "reason": "mmff94s_unparametrizable",
                })
        else:
            mmff_ok.append(m)
            mmff_properties.append(properties)

    energies = MMFFOptimizeMoleculesConfs(
        mmff_ok,
        maxIters=mmff_max_iters,
        properties=mmff_properties,
        hardwareOptions=hw,
    ) if mmff_ok else []
    if len(energies) != len(mmff_ok):
        raise RuntimeError(
            "nvMolKit returned MMFF energies for "
            f"{len(energies)} of {len(mmff_ok)} molecules"
        )

    n_written = 0
    for m, mol_energies in zip(mmff_ok, energies):
        if len(mol_energies) != m.GetNumConformers():
            raise RuntimeError(
                "nvMolKit MMFF94s returned a different number of energies than "
                "conformers for "
                f"{m.GetProp('_Name') if m.HasProp('_Name') else '<unnamed>'}"
            )
        if any(not math.isfinite(float(energy)) for energy in mol_energies):
            raise RuntimeError(
                "nvMolKit MMFF94s returned a non-finite energy for "
                f"{m.GetProp('_Name') if m.HasProp('_Name') else '<unnamed>'}"
            )
        m.SetProp("MMFF_Minimised", "True")
        for energy_idx, conf in enumerate(list(m.GetConformers())[:n_conformers]):
            m.SetProp("MMFF_Energy", f"{mol_energies[energy_idx]:.3f}")
            writer.write(m, confId=conf.GetId())
            n_written += 1
    for m in mmff_bad:
        m.SetProp("MMFF_Minimised", "False")
        for conf in list(m.GetConformers())[:n_conformers]:
            writer.write(m, confId=conf.GetId())
            n_written += 1

    return n_written, incomplete


def _absolute_path(path):
    """Return an absolute access path without resolving symbolic links."""
    return Path(os.path.abspath(Path(path).expanduser()))


def _stat_fingerprint(stat_result):
    return {
        "device": stat_result.st_dev,
        "inode": stat_result.st_ino,
        "bytes": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
        "ctime_ns": stat_result.st_ctime_ns,
    }


def _raise_input_changed(path):
    raise RuntimeError(
        f"Input changed while the conformer runner was reading it: "
        f"{_absolute_path(path)}. The staged SDF was not promoted; rerun with "
        "an unchanged input file."
    )


def _verify_observed_input(
    path,
    expected_provenance,
    stat_before,
    stat_after,
    observed_sha256,
):
    """Prove that the bytes consumed match the captured input snapshot."""
    before = _stat_fingerprint(stat_before)
    after = _stat_fingerprint(stat_after)
    access_path = _absolute_path(path)
    try:
        current = _stat_fingerprint(access_path.stat())
        resolved_path = str(access_path.resolve(strict=True))
    except OSError:
        _raise_input_changed(access_path)

    if before != after or after != current:
        _raise_input_changed(access_path)
    if resolved_path != expected_provenance["resolved_path"]:
        _raise_input_changed(access_path)
    for key in ("device", "inode", "bytes", "mtime_ns", "ctime_ns"):
        if after[key] != expected_provenance[key]:
            _raise_input_changed(access_path)
    if observed_sha256 != expected_provenance["sha256"]:
        _raise_input_changed(access_path)


def capture_input_provenance(path):
    """Hash a stable snapshot while retaining the original symlink access path."""
    access_path = _absolute_path(path)
    try:
        resolved_path = access_path.resolve(strict=True)
        digest = hashlib.sha256()
        with access_path.open("rb") as handle:
            stat_before = os.fstat(handle.fileno())
            if not stat.S_ISREG(stat_before.st_mode):
                raise ValueError(f"Input must be a regular file: {access_path}")
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
            stat_after = os.fstat(handle.fileno())
        current = access_path.stat()
        current_resolved = access_path.resolve(strict=True)
    except (FileNotFoundError, OSError):
        _raise_input_changed(access_path)

    fingerprint = _stat_fingerprint(stat_after)
    if (
        _stat_fingerprint(stat_before) != fingerprint
        or _stat_fingerprint(current) != fingerprint
        or current_resolved != resolved_path
    ):
        _raise_input_changed(access_path)
    return {
        "path": str(access_path),
        "resolved_path": str(resolved_path),
        **fingerprint,
        "sha256": digest.hexdigest(),
    }


def verify_input_provenance(expected_provenance):
    """Reject commit if the input path or bytes no longer match the snapshot."""
    current = capture_input_provenance(expected_provenance["path"])
    for key in (
        "resolved_path",
        "device",
        "inode",
        "bytes",
        "mtime_ns",
        "ctime_ns",
        "sha256",
    ):
        if current[key] != expected_provenance[key]:
            _raise_input_changed(expected_provenance["path"])


def _iter_smiles_file(path, expected_provenance=None):
    """Yield rows while hashing the exact bytes consumed from the input."""
    digest = hashlib.sha256() if expected_provenance is not None else None
    with open(path, "rb") as handle:
        stat_before = os.fstat(handle.fileno())
        for line_num, raw_line in enumerate(handle):
            if digest is not None:
                digest.update(raw_line)
            line = raw_line.decode()
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                fields = line.split("\t")
                smiles_raw = fields[0].strip()
                mol_id = fields[1].strip() if len(fields) > 1 else f"mol_{line_num}"
            else:
                fields = line.split()
                smiles_raw = fields[0]
                non_ext = [f for f in fields[1:] if not f.startswith("|")]
                mol_id = non_ext[0] if non_ext else f"mol_{line_num}"

            bare = _CXSMILES_EXT_RE.sub("", smiles_raw).strip()
            if bare.upper() in ("SMILES", "CANONICAL_SMILES", "SMI", "SMILE"):
                continue

            smiles = _CXSMILES_EXT_RE.sub("", smiles_raw).strip()
            yield smiles, mol_id
        stat_after = os.fstat(handle.fileno())

    if expected_provenance is not None:
        _verify_observed_input(
            path,
            expected_provenance,
            stat_before,
            stat_after,
            digest.hexdigest(),
        )


def _estimated_sdf_record_bytes(smiles, mol_id="sample"):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    mol.SetProp("_Name", str(mol_id))
    block = Chem.MolToMolBlock(mol)
    properties = (
        "\n>  <MMFF_Minimised>\nTrue\n\n"
        ">  <MMFF_Energy>\n-123.456\n\n$$$$\n"
    )
    return len((block + properties).encode("utf-8"))


def estimate_sdf_bytes(
    input_path,
    n_conformers,
    sample_size=DISK_SAMPLE_SIZE,
    safety_factor=DISK_SAFETY_FACTOR,
    input_provenance=None,
):
    """Estimate output bytes from a deterministic explicit-H molecule sample."""
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")
    if safety_factor < 1:
        raise ValueError("safety_factor must be at least 1")
    rng = random.Random(DEFAULT_RANDOM_SEED)
    sample = []
    molecule_count = 0
    for smiles, mol_id in _iter_smiles_file(
        input_path, expected_provenance=input_provenance
    ):
        record_bytes = _estimated_sdf_record_bytes(smiles, mol_id)
        if record_bytes is None:
            continue
        molecule_count += 1
        if len(sample) < sample_size:
            sample.append(record_bytes)
        else:
            replacement = rng.randrange(molecule_count)
            if replacement < sample_size:
                sample[replacement] = record_bytes
    if not sample:
        return {
            "molecule_count": 0,
            "sample_count": 0,
            "bytes_per_conformer": MIN_BYTES_PER_CONFORMER,
            "estimated_total_bytes": 0,
        }
    sampled_mean = sum(sample) / len(sample)
    bytes_per_conformer = max(
        MIN_BYTES_PER_CONFORMER, math.ceil(sampled_mean * safety_factor)
    )
    return {
        "molecule_count": molecule_count,
        "sample_count": len(sample),
        "bytes_per_conformer": bytes_per_conformer,
        "estimated_total_bytes": molecule_count * n_conformers * bytes_per_conformer,
    }


def check_disk_space(
    output_path, input_path, n_conformers, input_provenance=None
):
    estimate = estimate_sdf_bytes(
        input_path,
        n_conformers,
        input_provenance=input_provenance,
    )
    target = Path(output_path).resolve().parent
    free = shutil.disk_usage(target).free
    required = estimate["estimated_total_bytes"]
    print(
        f"Estimated output: {required / 1024**3:.1f} GB "
        f"({estimate['molecule_count']:,} mols; "
        f"{estimate['sample_count']:,}-molecule sample)"
    )
    print(f"Free space:      {free / 1024**3:.1f} GB on {target}")
    if required > free * DISK_FREE_MARGIN:
        raise RuntimeError(
            f"Insufficient disk space: estimated {required / 1024**3:.1f} GB, "
            f"{free / 1024**3:.1f} GB free. Reduce --n-conformers, use a larger "
            "volume, or pass --skip-disk-check to override."
        )
    return estimate


def _check_chunk_disk_space(
    output_path, rows_in_chunk, n_conformers, bytes_per_conformer
):
    free = shutil.disk_usage(Path(output_path).resolve().parent).free
    required = rows_in_chunk * n_conformers * bytes_per_conformer
    if required > free * DISK_FREE_MARGIN:
        raise RuntimeError(
            "Insufficient disk space for the next chunk: "
            f"{required / 1024**3:.1f} GB estimated, {free / 1024**3:.1f} GB free"
        )


def _failure_csv_path(output_path):
    return output_path.with_name(f"{output_path.stem}_conformer_failures.csv")


def _mmff_csv_path(output_path):
    return output_path.with_name(f"{output_path.stem}_mmff_unparametrizable.csv")


class _CsvRecordSink:
    """Append report rows directly to disk with constant memory usage."""

    def __init__(self, path, fieldnames):
        self.handle = open(path, "w", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=fieldnames)
        self.writer.writeheader()
        self.count = 0

    def append(self, record):
        self.writer.writerow(record)
        self.count += 1

    def flush(self):
        self.handle.flush()

    def close(self):
        self.handle.close()

    def __len__(self):
        return self.count


def _paths_alias(left, right):
    left = Path(left).expanduser().resolve()
    right = Path(right).expanduser().resolve()
    if left == right:
        return True
    try:
        return os.path.samefile(left, right)
    except (FileNotFoundError, OSError):
        return False


def _validate_regular_output_path(path, label):
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"Planned {label} must not be a symbolic link: {path}")
    if path.exists() and not path.is_file():
        raise ValueError(f"Planned {label} must be a regular file: {path}")


def _create_staged_sdf(output_path):
    """Create a unique, writable, same-directory SDF stage."""
    output = Path(output_path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    for _ in range(100):
        candidate = output.with_name(
            f"{output.stem}.{secrets.token_hex(6)}.partial{output.suffix}"
        )
        try:
            descriptor = os.open(candidate, flags, 0o666)
        except FileExistsError:
            continue
        try:
            created_mode = stat.S_IMODE(os.fstat(descriptor).st_mode)
            os.fchmod(descriptor, created_mode | stat.S_IWUSR)
        except Exception:
            os.close(descriptor)
            candidate.unlink(missing_ok=True)
            raise
        else:
            os.close(descriptor)
        return candidate
    raise RuntimeError(f"Could not allocate a unique staged SDF beside {output}")


def _output_lock_path(output_path):
    output = Path(output_path).expanduser().resolve()
    return output.parent / ".libprep.pipeline.lock"


class _OutputLock:
    def __init__(self, output_path):
        output = Path(output_path).expanduser().resolve()
        self.path = _output_lock_path(output)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, "a+")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError(
                f"Another conformer process is already writing {output}"
            ) from exc

    def close(self):
        if self.handle is None:
            return
        try:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def run_chunked(input_path, output_path, chunk_size, n_conformers,
                mmff_max_iters, batch_size, batches_per_gpu,
                preprocessing_threads, random_seed=DEFAULT_RANDOM_SEED,
                allow_partial_conformers=False, check_free_space=True):
    input_path = Path(input_path)
    output_path = Path(output_path)
    planned_paths = {
        "output SDF": output_path,
        "failure report": _failure_csv_path(output_path),
        "MMFF94s report": _mmff_csv_path(output_path),
        "output lock": _output_lock_path(output_path),
    }
    checked_paths = []
    for label, planned_path in planned_paths.items():
        _validate_regular_output_path(planned_path, label)
        if _paths_alias(input_path, planned_path):
            raise ValueError(
                f"--input aliases planned {label}: {input_path.resolve()}"
            )
        for other_label, other_path in checked_paths:
            if _paths_alias(planned_path, other_path):
                raise ValueError(
                    f"Planned {label} aliases planned {other_label}: "
                    f"{planned_path.resolve()}"
                )
        checked_paths.append((label, planned_path))
    output_lock = _OutputLock(output_path)
    try:
        input_provenance = capture_input_provenance(input_path)
        return _run_chunked_locked(
            input_path,
            output_path,
            chunk_size,
            n_conformers,
            mmff_max_iters,
            batch_size,
            batches_per_gpu,
            preprocessing_threads,
            random_seed=random_seed,
            allow_partial_conformers=allow_partial_conformers,
            check_free_space=check_free_space,
            input_provenance=input_provenance,
        )
    finally:
        output_lock.close()


def _run_chunked_locked(input_path, output_path, chunk_size, n_conformers,
                        mmff_max_iters, batch_size, batches_per_gpu,
                        preprocessing_threads, random_seed=DEFAULT_RANDOM_SEED,
                        allow_partial_conformers=False, check_free_space=True,
                        input_provenance=None):
    try:
        from rdkit.Chem import SDWriter
        import nvmolkit  # noqa: F401 — fail early with a clear message
    except ImportError as e:
        print(f"ERROR: {e}")
        print("Install nvMolKit: conda install -c conda-forge nvmolkit")
        sys.exit(1)

    positive_options = {
        "chunk size": chunk_size,
        "conformer count": n_conformers,
        "MMFF iterations": mmff_max_iters,
    }
    invalid = [name for name, value in positive_options.items() if value < 1]
    if invalid:
        raise ValueError(f"{', '.join(invalid)} must be positive")
    auto_tunable_options = {
        "batch size": batch_size,
        "batches per GPU": batches_per_gpu,
        "preprocessing threads": preprocessing_threads,
    }
    invalid = [
        name
        for name, value in auto_tunable_options.items()
        if value == 0 or value < -1
    ]
    if invalid:
        raise ValueError(f"{', '.join(invalid)} must be -1 (automatic) or positive")
    if not 0 <= random_seed <= 2_147_483_646:
        raise ValueError("random seed must be between 0 and 2147483646")
    failure_path = _failure_csv_path(output_path)
    mmff_path = _mmff_csv_path(output_path)
    if _paths_alias(input_path, output_path):
        raise ValueError(f"--output must not alias --input: {input_path.resolve()}")
    if _paths_alias(input_path, failure_path):
        raise ValueError(
            f"Failure report path must not alias --input: {input_path.resolve()}"
        )
    if _paths_alias(input_path, mmff_path):
        raise ValueError(
            f"MMFF94s report path must not alias --input: {input_path.resolve()}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    desired_output_mode = (
        stat.S_IMODE(output_path.stat().st_mode) if output_path.is_file() else None
    )
    if input_provenance is None:
        input_provenance = capture_input_provenance(input_path)
    disk_estimate = None
    if check_free_space:
        disk_estimate = check_disk_space(
            output_path,
            input_path,
            n_conformers,
            input_provenance=input_provenance,
        )
    partial_output_path = _create_staged_sdf(output_path)

    t_total = time.time()
    total_inputs = 0
    total_mols = 0
    total_confs = 0
    total_parse_fail = 0
    total_incomplete = 0
    total_unwritten = 0
    total_mmff_unparametrizable = 0
    chunk_idx = 0
    failure_records = _CsvRecordSink(failure_path, FAILURE_COLUMNS)
    try:
        mmff_failure_records = _CsvRecordSink(mmff_path, MMFF_FAILURE_COLUMNS)
    except Exception:
        failure_records.close()
        raise

    print(f"Input:      {input_path}")
    print(f"Output:     {output_path}")
    print(f"Chunk size: {chunk_size:,}")
    print(f"Conformers: {n_conformers}/mol")
    retry_seed = (random_seed + 1) % (2**31 - 1)
    print(f"Random seed: {random_seed} (retry: {retry_seed})")
    print(f"Partial output allowed: {allow_partial_conformers}")
    print(f"Staged SDF: {partial_output_path}")
    print()

    try:
        writer = SDWriter(str(partial_output_path))
    except Exception:
        try:
            failure_records.close()
        finally:
            mmff_failure_records.close()
        raise

    def process_chunk(rows, final=False):
        nonlocal chunk_idx, total_inputs, total_mols, total_confs
        nonlocal total_parse_fail, total_incomplete, total_unwritten
        nonlocal total_mmff_unparametrizable

        chunk_idx += 1
        t_chunk = time.time()
        failure_start = len(failure_records)
        mmff_failure_start = len(mmff_failure_records)
        total_inputs += len(rows)
        mols, parse_fail = _load_chunk(
            rows,
            failure_records=failure_records,
            chunk_idx=chunk_idx,
            n_conformers=n_conformers,
        )
        if check_free_space:
            _check_chunk_disk_space(
                output_path,
                len(mols),
                n_conformers,
                disk_estimate["bytes_per_conformer"],
            )
        tag = " (final)" if final else ""
        print(f"Chunk {chunk_idx}{tag}: {len(mols):,} mols "
              f"({parse_fail} parse failures)")

        if mols:
            n_written, incomplete = _embed_and_write(
                mols,
                writer,
                n_conformers,
                mmff_max_iters,
                batch_size,
                batches_per_gpu,
                preprocessing_threads,
                random_seed=random_seed,
                failure_records=failure_records,
                chunk_idx=chunk_idx,
                allow_partial_conformers=allow_partial_conformers,
                mmff_failure_records=mmff_failure_records,
            )
        else:
            n_written, incomplete = 0, 0

        total_mols += len(mols)
        total_confs += n_written
        total_parse_fail += parse_fail
        total_incomplete += incomplete
        total_mmff_unparametrizable += (
            len(mmff_failure_records) - mmff_failure_start
        )
        dt = time.time() - t_chunk
        requested = len(rows) * n_conformers
        embedding_shortfall = sum(
            max(0, n_conformers - mol.GetNumConformers()) for mol in mols
        )
        strict_policy_dropped = (
            sum(
                min(n_conformers, mol.GetNumConformers())
                for mol in mols
                if mol.GetNumConformers() < n_conformers
            )
            if not allow_partial_conformers
            else 0
        )
        chunk_unwritten = (
            parse_fail * n_conformers
            + embedding_shortfall
            + strict_policy_dropped
        )
        if n_written + chunk_unwritten != requested:
            raise RuntimeError(
                "Internal conformer accounting error: written + unwritten "
                "does not equal requested"
            )
        total_unwritten += chunk_unwritten
        print(f"  requested {requested:,}; wrote {n_written:,} conformers in {dt:.0f}s "
              f"({n_written/max(dt, 1e-9):.0f} confs/s)\n")

        del mols
        gc.collect()

        chunk_failures = len(failure_records) - failure_start
        failure_records.flush()
        mmff_failure_records.flush()
        if chunk_failures and not allow_partial_conformers:
            raise RuntimeError(
                f"Chunk {chunk_idx}: {chunk_failures:,} input molecules did not "
                f"produce all {n_conformers} requested conformers. "
                f"Details: {failure_path}. Use --allow-partial-conformers "
                "to keep a partial result."
            )

    try:
        chunk = []
        for smiles, mol_id in _iter_smiles_file(
            input_path, expected_provenance=input_provenance
        ):
            chunk.append((smiles, mol_id))
            if len(chunk) < chunk_size:
                continue
            process_chunk(chunk)
            chunk = []

        if chunk:
            process_chunk(chunk, final=True)
    finally:
        try:
            writer.close()
        finally:
            try:
                failure_records.close()
            finally:
                mmff_failure_records.close()

    if total_inputs == 0:
        raise RuntimeError(
            "Input contains no molecule rows; the existing final SDF was not replaced"
        )
    requested_total = total_inputs * n_conformers
    if total_confs + total_unwritten != requested_total:
        raise RuntimeError(
            "Internal conformer accounting error: written + unwritten does not "
            "equal requested"
        )
    verify_input_provenance(input_provenance)
    if desired_output_mode is not None:
        partial_output_path.chmod(desired_output_mode)
    os.replace(partial_output_path, output_path)
    dt_total = time.time() - t_total
    print("=" * 60)
    print(f"Done.  {total_inputs:,} inputs ({total_mols:,} parsed) "
          f"-> {total_confs:,} conformers")
    print(f"Requested conformers: {total_inputs * n_conformers:,}")
    print(f"Parse failures: {total_parse_fail:,}")
    print(f"Incomplete after retry: {total_incomplete:,}")
    print(f"Conformers unwritten: {total_unwritten:,}")
    print(f"MMFF94s-unparametrizable: {total_mmff_unparametrizable:,}")
    print(f"Total time: {dt_total:.0f}s ({dt_total/3600:.2f}h)")
    print(f"Throughput: {total_confs/max(dt_total, 1e-9):.0f} confs/s")
    print(f"Output: {output_path}")
    print(f"Failure CSV: {failure_path} ({len(failure_records):,} records)")
    print(
        f"MMFF94s CSV: {mmff_path} "
        f"({len(mmff_failure_records):,} records)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Memory-bounded chunked GPU conformer generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True,
                        help="Tab-delimited SMILES file (from library_pipeline.py)")
    parser.add_argument("--output", default="library.sdf",
                        help="Output SDF file (default: library.sdf)")
    parser.add_argument("--chunk-size", type=int, default=200_000,
                        help="Molecules per chunk (default: 200000)")
    parser.add_argument("--n-conformers", type=int, default=10,
                        help="Conformers per molecule (default: 10)")
    parser.add_argument("--mmff-max-iters", type=int, default=200,
                        help="MMFF max iterations (default: 200)")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="nvMolKit GPU batch size (default: 500; -1 automatic)")
    parser.add_argument("--batches-per-gpu", type=int, default=4,
                        help="nvMolKit batches per GPU (default: 4; -1 automatic)")
    parser.add_argument("--preprocessing-threads", type=int, default=8,
                        help="CPU preprocessing threads (default: 8; -1 automatic)")
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"ETKDG random seed (default: {DEFAULT_RANDOM_SEED}); retry uses seed + 1",
    )
    parser.add_argument(
        "--allow-partial-conformers",
        action="store_true",
        help=(
            "Succeed even when parsing or embedding leaves molecules with fewer "
            "than the requested conformer count"
        ),
    )
    parser.add_argument(
        "--skip-disk-check",
        action="store_true",
        help="Bypass sampled preflight and per-chunk free-space checks",
    )
    args = parser.parse_args()

    run_chunked(
        input_path=Path(args.input),
        output_path=Path(args.output),
        chunk_size=args.chunk_size,
        n_conformers=args.n_conformers,
        mmff_max_iters=args.mmff_max_iters,
        batch_size=args.batch_size,
        batches_per_gpu=args.batches_per_gpu,
        preprocessing_threads=args.preprocessing_threads,
        random_seed=args.random_seed,
        allow_partial_conformers=args.allow_partial_conformers,
        check_free_space=not args.skip_disk_check,
    )


if __name__ == "__main__":
    main()
