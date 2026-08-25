import argparse
import csv
import fcntl
import gc
import hashlib
import json
import math
import os
import random
import secrets
import shutil
import stat
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, SaltRemover, FilterCatalog, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalogParams
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)

DEFAULT_RANDOM_SEED = 0xF00D
MIN_BYTES_PER_CONFORMER = 4_500
DISK_SAMPLE_SIZE = 1_000
DISK_SAFETY_FACTOR = 1.25
DISK_FREE_MARGIN = 0.90
FAILURE_EXAMPLE_LIMIT = 100

PRESETS = {
    "docking": {
        "tautomers": False,
        "max_tautomers": 5,
        "ionise": True,
        "ph_min": 7.4,
        "ph_max": 7.4,
        "n_conformers": 1,
        "max_unspecified_stereo": 2,
    },
    "enumerate": {
        "tautomers": True,
        "max_tautomers": 5,
        "ionise": True,
        "ph_min": 6.4,
        "ph_max": 8.4,
        "n_conformers": 1,
        "max_unspecified_stereo": 2,
    },
}



# Optional dependencies


def _build_dimorphite_wrapper():
    def wrap_smiles_api(protonate):
        def fn(smi, min_ph, max_ph, precision):
            try:
                return protonate(
                    smi, ph_min=min_ph, ph_max=max_ph, precision=precision
                )
            except TypeError:
                try:
                    return protonate(
                        smi, min_ph=min_ph, max_ph=max_ph, precision=precision
                    )
                except TypeError:
                    return protonate(smi, min_ph=min_ph, max_ph=max_ph)

        return fn

    try:
        from dimorphite_dl.protonate import protonate_smiles as _p
        return wrap_smiles_api(_p), _dimorphite_version()
    except ImportError:
        pass

    try:
        import dimorphite_dl

        top_level_api = getattr(dimorphite_dl, "protonate_smiles", None)
        if callable(top_level_api):
            return wrap_smiles_api(top_level_api), _dimorphite_version()

        def fn(smi, min_ph, max_ph, precision):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return []
            try:
                return dimorphite_dl.run_with_mol_list(
                    [mol], min_ph=min_ph, max_ph=max_ph, pka_precision=precision
                )
            except TypeError:
                return dimorphite_dl.run_with_mol_list(
                    [mol], min_ph=min_ph, max_ph=max_ph
                )

        return fn, _dimorphite_version()
    except ImportError:
        return None, None


def _dimorphite_version():
    try:
        from importlib.metadata import version
        return version("dimorphite-dl")
    except Exception:
        return "unknown"


dimorphite_protonate, DIMORPHITE_VERSION = _build_dimorphite_wrapper()

try:
    from nvmolkit.substructure import hasSubstructMatch as _nvmk_has_match
    _NVMOLKIT_AVAILABLE = True
except ImportError:
    _NVMOLKIT_AVAILABLE = False
    _nvmk_has_match = None


def _nvmolkit_version():
    if not _NVMOLKIT_AVAILABLE:
        return None
    try:
        from importlib.metadata import version
        return version("nvmolkit")
    except Exception:
        return "unknown"



# PAINS SMARTS loading (for the batched GPU path)

def find_pains_csv():
   
    import rdkit as _rdkit

    env = os.environ.get("PAINS_CSV")
    if env and os.path.exists(env):
        return env
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bundled = os.path.join(script_dir, "pains_smarts.csv")
    if os.path.exists(bundled):
        return bundled
    rdkit_path = os.path.join(
        os.path.dirname(_rdkit.__file__), "Data", "Pains", "wehi_pains.csv"
    )
    if os.path.exists(rdkit_path):
        return rdkit_path
    raise FileNotFoundError(
        "PAINS SMARTS CSV not found. Tried $PAINS_CSV, "
        f"{bundled}, and {rdkit_path}. Use --pains-backend cpu to avoid needing it."
    )


def load_pains_query_mols():
    """Load PAINS SMARTS as RDKit query Mols + their reg-IDs. Cached per process."""
    if hasattr(load_pains_query_mols, "_cache"):
        return load_pains_query_mols._cache
    path = find_pains_csv()
    queries, names = [], []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            q = Chem.MolFromSmarts(row[0])
            if q is None:
                continue
            queries.append(q)
            names.append(row[1].strip("<>").replace("regId=", ""))
    load_pains_query_mols._cache = (queries, names)
    return queries, names


def _get_tautomer_canonicaliser():
    if not hasattr(_get_tautomer_canonicaliser, "_cache"):
        _get_tautomer_canonicaliser._cache = rdMolStandardize.TautomerEnumerator()
    return _get_tautomer_canonicaliser._cache


def canonical_tautomer(mol):
    
    if mol is None:
        return None
    try:
        return _get_tautomer_canonicaliser().Canonicalize(mol)
    except Exception:
        return mol


# Step 1: load + merge


def _strip_cxsmiles_ext(token):
    """Remove a trailing CXSmiles |...| extension block from a SMILES token."""
    if "|" in token:
        return token.split("|", 1)[0].strip()
    return token


def load_supplier_file(filepath, supplier_name=None, expected_provenance=None):
    """Load a supplier SMILES/cxsmiles file into a DataFrame."""

    
    if supplier_name is None:
        supplier_name = Path(filepath).stem

    records = []
    with open(filepath, "rb") as f:
        stat_before = os.fstat(f.fileno())
        digest = hashlib.sha256() if expected_provenance and "sha256" in expected_provenance else None
        for line_num, raw_line in enumerate(f):
            if digest is not None:
                digest.update(raw_line)
            line = raw_line.decode()
            line = line.rstrip("\n").strip()
            if not line:
                continue

            if "\t" in line:
                fields = line.split("\t")
                smiles = _strip_cxsmiles_ext(fields[0].strip())
                mol_id = fields[1].strip() if len(fields) > 1 else f"{supplier_name}_{line_num}"
            else:
                fields = line.split()
                if not fields:
                    continue
                smiles = _strip_cxsmiles_ext(fields[0])
                rest = [t for t in fields[1:] if not t.startswith("|")]
                mol_id = rest[0] if rest else f"{supplier_name}_{line_num}"

            if smiles.upper() in ("SMILES", "CANONICAL_SMILES", "SMI", "SMILE"):
                continue

            records.append({
                "ID": mol_id,
                "SMILES": smiles,
                "original_supplier_smiles": smiles,
                "supplier": supplier_name,
            })
        stat_after = os.fstat(f.fileno())

    if expected_provenance is not None:
        _verify_observed_input(
            filepath,
            expected_provenance,
            stat_before,
            stat_after,
            digest.hexdigest() if digest is not None else None,
        )

    df = pd.DataFrame(records)
    print(f"  Loaded {len(df):,} molecules from {supplier_name}")
    return df


def merge_suppliers(supplier_files, input_provenance=None):
    print("\n" + "=" * 60)
    print("STEP 1: LOAD AND MERGE SUPPLIER CATALOGUES")
    print("=" * 60)
    if input_provenance is not None and len(input_provenance) != len(supplier_files):
        raise ValueError("Input provenance must match supplier files by position")
    frames = [
        load_supplier_file(
            f,
            Path(f).stem,
            input_provenance[index] if input_provenance is not None else None,
        )
        for index, f in enumerate(supplier_files)
    ]
    merged = pd.concat(frames, ignore_index=True)
    print(f"\n  Total molecules after merge: {merged.shape[0]:,}")
    return merged


# Step 2: salts


def strip_salts(df):
    """Strip counter-ions, keeping the largest fragment by heavy-atom count."""
    print("\n" + "=" * 60)
    print("STEP 2: STRIP SALTS")
    print("=" * 60)

    remover = SaltRemover.SaltRemover()
    stripped = []
    failed = 0

    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol is None:
            failed += 1
            continue

        clean_smi = Chem.MolToSmiles(remover.StripMol(mol))

        if "." in clean_smi:
            frags = clean_smi.split(".")
            largest = max(
                frags,
                key=lambda s: (
                    Chem.MolFromSmiles(s).GetNumHeavyAtoms()
                    if Chem.MolFromSmiles(s) else 0
                ),
            )
            clean_smi = largest

        new_row = row.copy()
        new_row["SMILES"] = clean_smi
        stripped.append(new_row)

    result = pd.DataFrame(stripped)
    print(f"  Parse failures removed: {failed:,}")
    print(f"  Molecules after salt stripping: {len(result):,}")
    return result


# Step 3: filters


def resolve_pains_backend(pains_backend):
    if pains_backend == "auto":
        return "gpu" if _NVMOLKIT_AVAILABLE else "cpu"
    if pains_backend == "gpu" and not _NVMOLKIT_AVAILABLE:
        return "cpu"
    return pains_backend


def apply_filters(
    df,
    pains_backend="auto",
    custom_smarts=None,
    custom_smarts_provenance=None,
):
    """Complexity, BRENK, Lipinski, rings, aggregator, PAINS, optional custom SMARTS.

    PAINS is batched on GPU when nvMolKit is available; BRENK stays on CPU
    (RDKit ships no public BRENK SMARTS file). Returns (passed_df, failed_df).
    """
    print("\n" + "=" * 60)
    print("STEP 3: COMPOUND FILTERING")
    print("=" * 60)

    backend = resolve_pains_backend(pains_backend)
    if pains_backend == "gpu" and backend == "cpu":
        print("  WARNING: --pains-backend gpu requested but nvMolKit not importable. "
              "Falling back to CPU.")
    print(f"  PAINS backend: {backend}")

    brenk_params = FilterCatalogParams()
    brenk_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    brenk_cat = FilterCatalog.FilterCatalog(brenk_params)

    if backend == "gpu":
        pains_queries, pains_names = load_pains_query_mols()
        print(f"  PAINS patterns loaded: {len(pains_queries)}")
        pains_cat = None
    else:
        pains_params = FilterCatalogParams()
        pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        pains_cat = FilterCatalog.FilterCatalog(pains_params)
        pains_queries = pains_names = None

    custom_queries = []
    if custom_smarts:
        custom_queries = load_custom_smarts(
            custom_smarts, expected_provenance=custom_smarts_provenance
        )
        print(f"  Custom SMARTS patterns: {len(custom_queries)}")

    
    t1 = time.time()
    survivors = []
    failed_records = []

    for orig_idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol is None:
            failed_records.append({**row, "fail_reason": "parse_failed"})
            continue


        n_heavy = mol.GetNumHeavyAtoms()
        if n_heavy < 15:
            failed_records.append({**row, "fail_reason": f"too_small:heavy_atoms={n_heavy}"})
            continue
        if n_heavy > 70:
            failed_records.append({**row, "fail_reason": f"too_large:heavy_atoms={n_heavy}"})
            continue

        if brenk_cat.HasMatch(mol):
            match = brenk_cat.GetFirstMatch(mol)
            failed_records.append({**row, "fail_reason": f"brenk:{match.GetDescription()}"})
            continue

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        lip = []
        if mw > 500:
            lip.append(f"MW={mw:.0f}")
        if logp > 5:
            lip.append(f"logP={logp:.1f}")
        if hbd > 5:
            lip.append(f"HBD={hbd}")
        if hba > 10:
            lip.append(f"HBA={hba}")
        if lip:
            failed_records.append({**row, "fail_reason": f"lipinski:{';'.join(lip)}"})
            continue

        ring_info = mol.GetRingInfo()
        n_rings = ring_info.NumRings()
        if n_rings > 6:
            failed_records.append({**row, "fail_reason": f"too_many_rings:{n_rings}"})
            continue
        if n_rings > 0:
            largest_ring = max(len(r) for r in ring_info.AtomRings())
            if largest_ring > 8:
                failed_records.append({**row, "fail_reason": f"large_ring:size={largest_ring}"})
                continue

        if logp > 4.0 and mw > 400:
            fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
            if fsp3 < 0.1 and logp > 4.5:
                failed_records.append({
                    **row,
                    "fail_reason": f"aggregator:logP={logp:.1f};MW={mw:.0f};Fsp3={fsp3:.2f}",
                })
                continue

        if custom_queries:
            hit = next((n for n, q in custom_queries if mol.HasSubstructMatch(q)), None)
            if hit is not None:
                failed_records.append({**row, "fail_reason": f"custom_smarts:{hit}"})
                continue

        if backend == "cpu":
            if pains_cat.HasMatch(mol):
                match = pains_cat.GetFirstMatch(mol)
                failed_records.append({**row, "fail_reason": f"pains:{match.GetDescription()}"})
                continue

        survivors.append((orig_idx, row, mol))

    print(f"  Pass 1 (CPU checks): {time.time() - t1:.1f}s")
    print(f"    rejected:  {len(failed_records):,}")
    print(f"    survivors: {len(survivors):,}")

    # ---- Pass 2: batched GPU PAINS ----
    if backend == "gpu" and survivors:
        import numpy as np

        t2 = time.time()
        survivor_mols = [m for _, _, m in survivors]
        print(f"  Pass 2 (GPU PAINS): {len(survivor_mols):,} mols x "
              f"{len(pains_queries)} patterns")

        match_matrix = np.asarray(_nvmk_has_match(survivor_mols, pains_queries))

        passed = []
        for i, (_, row, _) in enumerate(survivors):
            hits = np.where(match_matrix[i] == 1)[0]
            if hits.size > 0:
                failed_records.append({**row, "fail_reason": f"pains:{pains_names[int(hits[0])]}"})
            else:
                passed.append(row)
        print(f"    GPU call + reduction: {time.time() - t2:.2f}s")
        print(f"    PAINS rejects: {len(survivors) - len(passed):,}")
    else:
        passed = [row for _, row, _ in survivors]

    pass_df = pd.DataFrame(passed)
    fail_df = pd.DataFrame(failed_records)

    print(f"\n  Passed: {len(pass_df):,}")
    print(f"  Failed: {len(fail_df):,}")
    if len(fail_df) > 0 and "fail_reason" in fail_df.columns:
        reasons = fail_df["fail_reason"].apply(lambda x: x.split(":")[0])
        for reason, count in reasons.value_counts().items():
            print(f"    {reason:20s} {count:>8,}")

    return pass_df, fail_df


def load_custom_smarts(path, expected_provenance=None):
    """Load user-supplied SMARTS rejection patterns."""

    
    queries = []
    with open(path, "rb") as f:
        stat_before = os.fstat(f.fileno())
        digest = hashlib.sha256() if expected_provenance and "sha256" in expected_provenance else None
        for lineno, raw_line in enumerate(f, 1):
            if digest is not None:
                digest.update(raw_line)
            line = raw_line.decode()
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            smarts = parts[0]
            name = parts[1].strip() if len(parts) > 1 else f"line{lineno}"
            q = Chem.MolFromSmarts(smarts)
            if q is None:
                raise ValueError(f"{path}:{lineno}: unparseable SMARTS: {smarts!r}")
            queries.append((name, q))
        stat_after = os.fstat(f.fileno())

    if expected_provenance is not None:
        _verify_observed_input(
            path,
            expected_provenance,
            stat_before,
            stat_after,
            digest.hexdigest() if digest is not None else None,
        )
    return queries



# Step 4: stereo


def unspecified_stereo_elements(mol):
    """Return every unassigned stereo element RDKit can enumerate.

    ``FindMolChiralCenters`` only sees tetrahedral atoms, so using it as the
    enumeration gate silently leaves E/Z-only molecules unspecified.
    ``FindPotentialStereo`` covers both atom and bond stereochemistry.
    """
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return [
        info
        for info in Chem.FindPotentialStereo(mol)
        if info.specified == Chem.StereoSpecified.Unspecified
    ]


def count_unspecified_stereocentres(mol):
    """Backward-compatible name for the total unassigned stereo-element count."""
    return len(unspecified_stereo_elements(mol))


def filter_and_enumerate_stereo(df, max_unspecified=2):
    print("\n" + "=" * 60)
    print("STEP 4: STEREO FILTERING + ENUMERATION")
    print("=" * 60)

    filtered_out = 0
    records = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol is None:
            continue

        n_unspec = len(unspecified_stereo_elements(mol))
        if n_unspec > max_unspecified:
            filtered_out += 1
            continue

        if n_unspec == 0:
            records.append(row.to_dict())
        else:
            opts = StereoEnumerationOptions(
                tryEmbedding=False,
                onlyUnassigned=True,
                maxIsomers=1 << n_unspec,
                unique=True,
            )
            for iso_idx, iso_mol in enumerate(EnumerateStereoisomers(mol, options=opts)):
                rec = row.to_dict()
                rec["SMILES"] = Chem.MolToSmiles(iso_mol, isomericSmiles=True)
                rec["ID"] = f"{row['ID']}_iso{iso_idx + 1}"
                records.append(rec)

    result = pd.DataFrame(records)
    print(f"  Removed (>{max_unspecified} unspecified stereo elements): {filtered_out:,}")
    print(f"  Molecules after enumeration: {len(result):,}")
    return result



# Step 4b: tautomers (off by default — see PRESETS)


def enumerate_tautomers(df, max_tautomers=5):
    print("\n" + "=" * 60)
    print("STEP 4b: TAUTOMER ENUMERATION")
    print("=" * 60)

    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTautomers(max_tautomers * 5)
    enumerator.SetMaxTransforms(1000)

    records = []
    expanded = 0
    failed = 0

    for i, (_, row) in enumerate(df.iterrows()):
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol is None:
            records.append(row.to_dict())
            continue

        try:
            tauts = list(enumerator.Enumerate(mol))
            if len(tauts) <= 1:
                records.append(row.to_dict())
            else:
                expanded += 1
                for t_idx, t_mol in enumerate(tauts[:max_tautomers]):
                    t_smi = Chem.MolToSmiles(t_mol, isomericSmiles=True)
                    if Chem.MolFromSmiles(t_smi) is None:
                        continue
                    rec = row.to_dict()
                    rec["SMILES"] = t_smi
                    if t_idx > 0:
                        rec["ID"] = f"{row['ID']}_tau{t_idx + 1}"
                    records.append(rec)
        except Exception:
            records.append(row.to_dict())
            failed += 1

        if (i + 1) % 50_000 == 0:
            print(f"  Processed {i + 1:,} / {len(df):,}...")

    result = pd.DataFrame(records)
    print(f"  Molecules with tautomers: {expanded:,}")
    print(f"  Enumeration failures (kept original): {failed:,}")
    print(f"  Molecules after tautomer enumeration: {len(result):,}")
    return result



# Step 5: dedup


def deduplicate(df):
    """Deduplicate by canonical SMILES, merging IDs / suppliers rather than dropping.

    NOTE: this is SMILES-level dedup on 2D structures, which is correct here.
    It must NEVER be applied to a conformer SDF — different conformers of the
    same molecule share a canonical SMILES and would be collapsed.
    """
    print("\n" + "=" * 60)
    print("STEP 5: DEDUPLICATE")
    print("=" * 60)

    before = len(df)
    df = df.copy()
    df["canonical"] = df["SMILES"].apply(
        lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=True)
        if Chem.MolFromSmiles(s) is not None else None
    )
    df = df.dropna(subset=["canonical"])

    grouped = df.groupby("canonical", as_index=False).agg({
        "ID": lambda x: ";".join(sorted(set(x))),
        "original_supplier_smiles": lambda x: ";".join(sorted(set(x))),
        "supplier": lambda x: ";".join(sorted(set(x))),
    })
    grouped = grouped.rename(columns={"canonical": "SMILES"})

    print(f"  Before: {before:,}")
    print(f"  After:  {len(grouped):,}")
    print(f"  Duplicates removed: {before - len(grouped):,}")
    return grouped



# Step 6: ionisation


def ionise_molecules(df, ph_min=7.4, ph_max=7.4, precision=None):
    """Assign protonation state(s) with Dimorphite-DL.

    When ph_min == ph_max and precision == 0.0 this yields a SINGLE state per
    molecule (equivalent to Open Babel -p 7.4) — the docking default.
    A pH *range* enumerates multiple states and multiplies library size.
    """
    if precision is None:
        precision = 0.0 if ph_min == ph_max else 1.0

    print("\n" + "=" * 60)
    if ph_min == ph_max:
        print(f"STEP 6: IONISE (Dimorphite-DL, single state @ pH {ph_min:.1f})")
    else:
        print(f"STEP 6: IONISE (Dimorphite-DL, pH {ph_min:.1f}\u2013{ph_max:.1f}, "
              f"multi-state)")
    print("=" * 60)

    if dimorphite_protonate is None:
        print("  WARNING: dimorphite_dl not installed. Skipping ionisation.")
        return df

    records = []
    failed = 0

    for i, (_, row) in enumerate(df.iterrows()):
        try:
            variants = dimorphite_protonate(
                row["SMILES"], min_ph=ph_min, max_ph=ph_max, precision=precision
            )
            if not variants:
                records.append(row.to_dict())
                continue

            for v_idx, variant in enumerate(variants):
                v_smi = variant.strip() if isinstance(variant, str) else Chem.MolToSmiles(variant)
                if not v_smi or Chem.MolFromSmiles(v_smi) is None:
                    continue
                rec = row.to_dict()
                rec["SMILES"] = v_smi
                if len(variants) > 1:
                    rec["ID"] = ";".join(
                        f"{source_id}_pH{v_idx + 1}"
                        for source_id in str(row["ID"]).split(";")
                        if source_id
                    )
                records.append(rec)
        except Exception:
            records.append(row.to_dict())
            failed += 1

        if (i + 1) % 10_000 == 0:
            print(f"  Ionised {i + 1:,} / {len(df):,}...")

    result = pd.DataFrame(records)
    print(f"  Dimorphite failures (kept original): {failed:,}")
    print(f"  Molecules after ionisation: {len(result):,}")
    return result


def canonical_redup(df):
    """Canonicalise and merge exact duplicates after ionisation.

    Keep every supplier and source ID. Dropping the later duplicate here loses
    provenance when two ionisation paths converge on the same structure.
    """
    before = len(df)
    df = df.copy()
    df["canonical"] = df["SMILES"].apply(
        lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=True)
        if Chem.MolFromSmiles(s) is not None else None
    )
    df = df.dropna(subset=["canonical"])

    def merge_values(values):
        parts = set()
        for value in values.dropna().astype(str):
            parts.update(part for part in value.split(";") if part)
        return ";".join(sorted(parts))

    grouped = df.groupby("canonical", as_index=False).agg({
        "ID": merge_values,
        "original_supplier_smiles": merge_values,
        "supplier": merge_values,
    })
    grouped = grouped.rename(columns={"canonical": "SMILES"})
    print(f"  Re-dedup: {before:,} -> {len(grouped):,}")
    return grouped



# Step 7: conformers — GPU only, always chunked, streamed from disk


def _import_nvmolkit():
    """Import nvMolKit conformer entry points."""
    try:
        from rdkit.Chem import SDWriter  # noqa: F401
        from rdkit.Chem.rdDistGeom import ETKDGv3  # noqa: F401
        from nvmolkit.embedMolecules import EmbedMolecules
        from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs
        from nvmolkit.types import HardwareOptions
    except ImportError as e:
        raise RuntimeError(
            f"nvMolKit is required for conformer generation but is not importable: {e}\n"
            "  Install:  conda install -c conda-forge nvmolkit\n"
            "  This pipeline is GPU-only. There is no CPU conformer backend.\n"
            "  Use --skip-conformers to stop after the 2D stages."
        ) from e
    return EmbedMolecules, MMFFOptimizeMoleculesConfs, HardwareOptions


def gpu_smoke_test(random_seed=DEFAULT_RANDOM_SEED):
    """Embed one trivial molecule and verify real 3D coordinates come back."""
    EmbedMolecules, MMFFOptimizeMoleculesConfs, HardwareOptions = _import_nvmolkit()
    from rdkit.Chem.rdDistGeom import ETKDGv3

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    params = ETKDGv3()
    params.useRandomCoords = True
    params.randomSeed = random_seed
    hw = HardwareOptions(preprocessingThreads=1, batchSize=1, batchesPerGpu=1, gpuIds=[])

    EmbedMolecules([mol], params, confsPerMolecule=1, hardwareOptions=hw)

    if mol.GetNumConformers() == 0:
        raise RuntimeError(
            "GPU smoke test FAILED: nvMolKit imported but produced no conformer "
            "for ethanol.\n"
            "  Almost always a CUDA driver / kernel mismatch. Check `nvidia-smi` "
            "driver version against the nvMolKit build requirements."
        )

    pos = mol.GetConformer().GetPositions()
    if not pos.any():
        raise RuntimeError("GPU smoke test FAILED: conformer returned all-zero coordinates.")

    properties = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    if properties is None:
        raise RuntimeError("GPU smoke test FAILED: ethanol has no MMFF94s parameters.")
    energies = MMFFOptimizeMoleculesConfs(
        [mol], maxIters=50, properties=[properties], hardwareOptions=hw
    )
    if (
        len(energies) != 1
        or len(energies[0]) != mol.GetNumConformers()
        or any(not math.isfinite(float(energy)) for energy in energies[0])
    ):
        raise RuntimeError("GPU smoke test FAILED: invalid MMFF94s energy output.")
    print("  GPU smoke test passed (ethanol embedded + minimised).")


def _iter_smiles_file(path, expected_provenance=None):
    """Yield final SMILES while verifying the exact captured input stream."""
    with open(path, "rb") as f:
        stat_before = os.fstat(f.fileno())
        digest = (
            hashlib.sha256()
            if expected_provenance and "sha256" in expected_provenance
            else None
        )
        for line_num, raw_line in enumerate(f):
            if digest is not None:
                digest.update(raw_line)
            line = raw_line.decode()
            line = line.strip()
            if not line:
                continue
            fields = line.split("\t")
            smiles = fields[0].strip()
            mol_id = fields[1].strip() if len(fields) > 1 else f"mol_{line_num}"
            if smiles.upper() in ("SMILES", "CANONICAL_SMILES", "SMI", "SMILE"):
                continue
            yield smiles, mol_id
        stat_after = os.fstat(f.fileno())

    if expected_provenance is not None:
        _verify_observed_input(
            path,
            expected_provenance,
            stat_before,
            stat_after,
            digest.hexdigest() if digest is not None else None,
        )


def _load_chunk(rows):
    mols = []
    failures = []
    for smiles, mol_id in rows:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            failures.append({
                "ID": str(mol_id),
                "reason": "parse_failed",
                "requested": 0,
                "generated": 0,
                "retry_attempted": False,
            })
            continue
        m = Chem.AddHs(m)
        m.SetProp("_Name", str(mol_id))
        mols.append(m)
    return mols, failures


def _embed_once(mols, n_conformers, hw, random_seed):
    """Embed a molecule batch in-place with a deterministic ETKDG seed."""
    if not mols:
        return
    EmbedMolecules, _, _ = _import_nvmolkit()
    from rdkit.Chem.rdDistGeom import ETKDGv3

    params = ETKDGv3()
    params.useRandomCoords = True  # required by nvMolKit
    params.randomSeed = random_seed
    EmbedMolecules(mols, params, confsPerMolecule=n_conformers, hardwareOptions=hw)


def _conformer_shortfalls(mols, n_conformers, retry_attempted):
    failures = []
    for mol in mols:
        generated = mol.GetNumConformers()
        if generated < n_conformers:
            failures.append({
                "ID": mol.GetProp("_Name") if mol.HasProp("_Name") else "",
                "reason": "conformer_shortfall",
                "requested": n_conformers,
                "generated": generated,
                "retry_attempted": retry_attempted,
            })
    return failures


def _embedding_exception_failures(
    mols, n_conformers, exc, *, stage, retry_attempted
):
    reason = f"{stage}_error:{type(exc).__name__}:{exc}"
    return [
        {
            "ID": mol.GetProp("_Name") if mol.HasProp("_Name") else "",
            "reason": reason,
            "requested": n_conformers,
            "generated": mol.GetNumConformers(),
            "retry_attempted": retry_attempted,
        }
        for mol in mols
    ]


def _embed_and_write(
    mols,
    writer,
    n_conformers,
    mmff_max_iters,
    hw,
    retry_hw=None,
    random_seed=DEFAULT_RANDOM_SEED,
    allow_partial_conformers=False,
    conformer_failure_records=None,
    mmff_failure_records=None,
):
    """Embed, retry shortfalls once, MMFF94s-minimise, and write a batch."""
    _, MMFFOptimizeMoleculesConfs, _ = _import_nvmolkit()
    try:
        _embed_once(mols, n_conformers, hw, random_seed)
    except Exception as exc:
        error_failures = _embedding_exception_failures(
            mols,
            n_conformers,
            exc,
            stage="primary_embedding",
            retry_attempted=False,
        )
        if conformer_failure_records is not None:
            conformer_failure_records.extend(error_failures)
        raise RuntimeError(
            "nvMolKit primary embedding failed; no conformers from this batch were accepted"
        ) from exc

    primary_counts = [mol.GetNumConformers() for mol in mols]
    retry_indices = [
        index for index, count in enumerate(primary_counts) if count < n_conformers
    ]
    if retry_indices and retry_hw is not None:
        retry_mols = []
        for index in retry_indices:
            retry_mol = Chem.Mol(mols[index])
            retry_mol.RemoveAllConformers()
            retry_mols.append(retry_mol)
        try:
            _embed_once(
                retry_mols,
                n_conformers,
                retry_hw,
                (random_seed + 1) % (2**31 - 1),
            )
        except Exception as exc:
            error_failures = _embedding_exception_failures(
                retry_mols,
                n_conformers,
                exc,
                stage="retry_embedding",
                retry_attempted=True,
            )
            for failure, index in zip(error_failures, retry_indices):
                failure["generated"] = max(
                    failure["generated"], primary_counts[index]
                )
            if conformer_failure_records is not None:
                conformer_failure_records.extend(error_failures)
            raise RuntimeError(
                "nvMolKit conservative embedding retry failed; no conformers from "
                "this batch were accepted"
            ) from exc
        for index, retry_mol in zip(retry_indices, retry_mols):
            if retry_mol.GetNumConformers() > primary_counts[index]:
                mols[index] = retry_mol

    failures = _conformer_shortfalls(
        mols, n_conformers, retry_attempted=bool(retry_indices and retry_hw is not None)
    )
    if conformer_failure_records is not None:
        conformer_failure_records.extend(failures)
    incomplete_molecules = {
        id(mol) for mol in mols if mol.GetNumConformers() < n_conformers
    }
    if allow_partial_conformers:
        write_mols = [mol for mol in mols if mol.GetNumConformers() > 0]
    else:
        write_mols = [
            mol
            for mol in mols
            if id(mol) not in incomplete_molecules
        ]

    mmff_ok, mmff_properties, mmff_bad = [], [], []
    for mol in write_mols:
        properties = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        if properties is None:
            mmff_bad.append(mol)
        else:
            mmff_ok.append(mol)
            mmff_properties.append(properties)

    mmff_skipped_ids = [
        mol.GetProp("_Name") if mol.HasProp("_Name") else "" for mol in mmff_bad
    ]
    if mmff_failure_records is not None:
        mmff_failure_records.extend(
            {"ID": mol_id, "reason": "mmff94s_unparametrizable"}
            for mol_id in mmff_skipped_ids
        )

    energies = (
        MMFFOptimizeMoleculesConfs(
            mmff_ok,
            maxIters=mmff_max_iters,
            properties=mmff_properties,
            hardwareOptions=hw,
        )
        if mmff_ok
        else []
    )
    if len(energies) != len(mmff_ok):
        raise RuntimeError(
            "nvMolKit MMFF94s returned energy results for "
            f"{len(energies)} of {len(mmff_ok)} molecules"
        )

    n_written = 0
    for mol, mol_energies in zip(mmff_ok, energies):
        conformers_to_write = min(mol.GetNumConformers(), n_conformers)
        if len(mol_energies) != mol.GetNumConformers():
            raise RuntimeError(
                "nvMolKit MMFF94s returned a different number of energies than conformers for "
                f"{mol.GetProp('_Name') if mol.HasProp('_Name') else '<unnamed>'}"
            )
        if any(not math.isfinite(float(energy)) for energy in mol_energies):
            raise RuntimeError(
                "nvMolKit MMFF94s returned a non-finite energy for "
                f"{mol.GetProp('_Name') if mol.HasProp('_Name') else '<unnamed>'}"
            )
        mol.SetProp("MMFF_Minimised", "True")
        for energy_index, conformer in enumerate(
            list(mol.GetConformers())[:conformers_to_write]
        ):
            mol.SetProp("MMFF_Energy", f"{mol_energies[energy_index]:.3f}")
            writer.write(mol, confId=conformer.GetId())
            n_written += 1

    for mol in mmff_bad:
        mol.SetProp("MMFF_Minimised", "False")
        for conformer in list(mol.GetConformers())[:n_conformers]:
            writer.write(mol, confId=conformer.GetId())
            n_written += 1

    conformer_shortfall = sum(
        failure["requested"] - failure["generated"] for failure in failures
    )
    unwritten_conformers = len(mols) * n_conformers - n_written
    return {
        "confs": n_written,
        "successful_mols": len(mols) - len(failures),
        "failed_mols": len(failures),
        "conformer_shortfall": conformer_shortfall,
        "unwritten_conformers": unwritten_conformers,
        "policy_dropped_conformers": max(
            0, unwritten_conformers - conformer_shortfall
        ),
        "retried_mols": len(retry_indices),
        "failures": failures,
        "mmff_skipped": len(mmff_bad),
        "mmff_skipped_ids": mmff_skipped_ids,
    }


class _CsvRecordSink:
    """Stream report rows to disk while retaining only bounded examples."""

    def __init__(self, path, fieldnames, example_limit=0):
        self.path = str(path)
        self.handle = open(path, "w", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=fieldnames)
        self.writer.writeheader()
        self.count = 0
        self.example_limit = example_limit
        self.examples = []

    def append(self, record):
        self.writer.writerow(record)
        self.count += 1
        if len(self.examples) < self.example_limit:
            self.examples.append(dict(record))

    def extend(self, records):
        for record in records:
            self.append(record)
        self.handle.flush()

    def close(self):
        self.handle.close()


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
    """Process-level lock preventing concurrent writers for one output path."""

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
                f"Another pipeline process is already writing {output}"
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


def _check_chunk_disk_space(output_sdf, rows_in_chunk, n_conformers, bytes_per_conformer):
    if not bytes_per_conformer:
        return
    target = Path(output_sdf).resolve().parent
    free = shutil.disk_usage(target).free
    required = rows_in_chunk * n_conformers * bytes_per_conformer
    if required > free * DISK_FREE_MARGIN:
        raise RuntimeError(
            f"Insufficient disk space for the next chunk. Estimated {required / 1024**3:.1f} GB "
            f"needed, {free / 1024**3:.1f} GB free on {target}."
        )


def generate_conformers(smi_path, output_sdf, n_conformers=1, chunk_size=100_000,
                        mmff_max_iters=200, batch_size=500, batches_per_gpu=4,
                        preprocessing_threads=8, gpu_ids=None,
                        random_seed=DEFAULT_RANDOM_SEED,
                        allow_partial_conformers=False,
                        check_free_space=True,
                        estimated_bytes_per_conformer=None,
                        output_lock_held=False,
                        before_commit=None,
                        input_provenance=None):
    """Lock an output target, then run bounded-memory GPU generation."""
    positive_options = {
        "n_conformers": n_conformers,
        "chunk_size": chunk_size,
        "mmff_max_iters": mmff_max_iters,
    }
    for name, value in positive_options.items():
        if value < 1:
            raise ValueError(f"{name} must be at least 1")
    auto_options = {
        "batch_size": batch_size,
        "batches_per_gpu": batches_per_gpu,
        "preprocessing_threads": preprocessing_threads,
    }
    for name, value in auto_options.items():
        if value == 0 or value < -1:
            raise ValueError(f"{name} must be -1 (automatic) or at least 1")
    if not 0 <= random_seed < 2**31 - 1:
        raise ValueError("random_seed must be between 0 and 2147483646")
    if (
        estimated_bytes_per_conformer is not None
        and estimated_bytes_per_conformer <= 0
    ):
        raise ValueError("estimated_bytes_per_conformer must be positive")

    smi_path = Path(smi_path).expanduser().resolve(strict=True)
    output_path = Path(output_sdf)
    stem = str(output_path.with_suffix(""))
    planned_paths = {
        "output SDF": output_path,
        "conformer failure report": Path(f"{stem}_conformer_failures.csv"),
        "MMFF94s report": Path(f"{stem}_mmff_unparametrizable.csv"),
        "output lock": _output_lock_path(output_path),
    }
    checked_paths = []
    for label, planned_path in planned_paths.items():
        _validate_regular_output_path(planned_path, label)
        if _paths_alias(smi_path, planned_path):
            raise ValueError(
                f"Conformer input aliases planned {label}: {Path(smi_path).resolve()}"
            )
        for other_label, other_path in checked_paths:
            if _paths_alias(planned_path, other_path):
                raise ValueError(
                    f"Planned {label} aliases planned {other_label}: "
                    f"{planned_path.resolve()}"
                )
        checked_paths.append((label, planned_path))
    output_lock = None if output_lock_held else _OutputLock(output_sdf)
    try:
        if input_provenance is None:
            input_provenance = capture_input_provenance(
                [smi_path], hash_inputs=True
            )[0]
        return _generate_conformers_locked(
            smi_path,
            output_sdf,
            n_conformers=n_conformers,
            chunk_size=chunk_size,
            mmff_max_iters=mmff_max_iters,
            batch_size=batch_size,
            batches_per_gpu=batches_per_gpu,
            preprocessing_threads=preprocessing_threads,
            gpu_ids=gpu_ids,
            random_seed=random_seed,
            allow_partial_conformers=allow_partial_conformers,
            check_free_space=check_free_space,
            estimated_bytes_per_conformer=estimated_bytes_per_conformer,
            before_commit=before_commit,
            input_provenance=input_provenance,
        )
    finally:
        if output_lock is not None:
            output_lock.close()


def _generate_conformers_locked(
    smi_path,
    output_sdf,
    n_conformers=1,
    chunk_size=100_000,
    mmff_max_iters=200,
    batch_size=500,
    batches_per_gpu=4,
    preprocessing_threads=8,
    gpu_ids=None,
    random_seed=DEFAULT_RANDOM_SEED,
    allow_partial_conformers=False,
    check_free_space=True,
    estimated_bytes_per_conformer=None,
    before_commit=None,
    input_provenance=None,
):
    """GPU conformer generation, streamed from disk in bounded-memory chunks."""
    print("\n" + "=" * 60)
    print("STEP 7: CONFORMER GENERATION (nvMolKit / GPU, chunked)")
    print("=" * 60)

    from rdkit.Chem import SDWriter
    _, _, HardwareOptions = _import_nvmolkit()

    hw = HardwareOptions(
        preprocessingThreads=preprocessing_threads,
        batchSize=batch_size,
        batchesPerGpu=batches_per_gpu,
        gpuIds=gpu_ids if gpu_ids else [],
    )
    retry_hw = HardwareOptions(
        preprocessingThreads=1,
        batchSize=1,
        batchesPerGpu=1,
        gpuIds=gpu_ids if gpu_ids else [],
    )

    print(f"  chunk-size: {chunk_size:,}  batch-size: {batch_size}  "
          f"batches/gpu: {batches_per_gpu}")
    print(f"  random-seed: {random_seed}  force-field: MMFF94s")

    output_path = Path(output_sdf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    desired_output_mode = (
        stat.S_IMODE(output_path.stat().st_mode) if output_path.is_file() else None
    )
    if check_free_space and estimated_bytes_per_conformer is None:
        disk_estimate = check_disk_space(
            output_path,
            smi_path,
            n_conformers,
            input_provenance=input_provenance,
        )
        estimated_bytes_per_conformer = disk_estimate["bytes_per_conformer"]

    t_total = time.time()
    totals = {
        "input_rows": 0,
        "mols": 0,
        "successful_mols": 0,
        "failed_mols": 0,
        "conformer_shortfall": 0,
        "unwritten_conformers": 0,
        "policy_dropped_conformers": 0,
        "retried_mols": 0,
        "confs": 0,
        "parse_fail": 0,
        "mmff_skipped": 0,
    }
    chunk_idx = 0
    stem = str(output_path.with_suffix(""))
    failure_report = f"{stem}_conformer_failures.csv"
    mmff_report = f"{stem}_mmff_unparametrizable.csv"
    partial_output_path = _create_staged_sdf(output_path)
    print(f"  staged SDF: {partial_output_path}")
    failure_sink = _CsvRecordSink(
        failure_report,
        ["ID", "reason", "requested", "generated", "retry_attempted"],
        example_limit=FAILURE_EXAMPLE_LIMIT,
    )
    try:
        mmff_sink = _CsvRecordSink(mmff_report, ["ID", "reason"])
    except Exception:
        failure_sink.close()
        raise
    try:
        writer = SDWriter(str(partial_output_path))
    except Exception:
        try:
            failure_sink.close()
        finally:
            mmff_sink.close()
        raise

    def _process(rows, final=False):
        nonlocal chunk_idx
        chunk_idx += 1
        t0 = time.time()
        mols, parse_failures = _load_chunk(rows)
        for failure in parse_failures:
            failure["requested"] = n_conformers
        failure_sink.extend(parse_failures)
        if check_free_space:
            _check_chunk_disk_space(
                output_path, len(mols), n_conformers, estimated_bytes_per_conformer
            )
        tag = " (final)" if final else ""
        print(
            f"  Chunk {chunk_idx}{tag}: {len(mols):,} mols "
            f"({len(parse_failures)} parse failures)"
        )

        result = {
            "confs": 0,
            "successful_mols": 0,
            "failed_mols": 0,
            "conformer_shortfall": 0,
            "unwritten_conformers": 0,
            "policy_dropped_conformers": 0,
            "retried_mols": 0,
            "failures": [],
            "mmff_skipped": 0,
            "mmff_skipped_ids": [],
        }
        if mols:
            conformer_failure_count = failure_sink.count
            mmff_failure_count = mmff_sink.count
            result = _embed_and_write(
                mols,
                writer,
                n_conformers,
                mmff_max_iters,
                hw,
                retry_hw=retry_hw,
                random_seed=(random_seed + chunk_idx - 1) % (2**31 - 1),
                allow_partial_conformers=allow_partial_conformers,
                conformer_failure_records=failure_sink,
                mmff_failure_records=mmff_sink,
            )
            # Test doubles and older private callers may not consume the optional
            # report sinks, so retain a compatibility fallback without duplicating
            # records from the real implementation.
            if failure_sink.count == conformer_failure_count:
                failure_sink.extend(result["failures"])
            if mmff_sink.count == mmff_failure_count:
                mmff_sink.extend(
                    {"ID": mol_id, "reason": "mmff94s_unparametrizable"}
                    for mol_id in result["mmff_skipped_ids"]
                )

        chunk_failures = parse_failures + result["failures"]
        totals["input_rows"] += len(rows)
        totals["mols"] += len(mols)
        totals["successful_mols"] += result["successful_mols"]
        totals["failed_mols"] += len(parse_failures) + result["failed_mols"]
        totals["conformer_shortfall"] += (
            len(parse_failures) * n_conformers + result["conformer_shortfall"]
        )
        totals["unwritten_conformers"] += (
            len(parse_failures) * n_conformers
            + result.get("unwritten_conformers", result["conformer_shortfall"])
        )
        totals["policy_dropped_conformers"] += result.get(
            "policy_dropped_conformers", 0
        )
        totals["retried_mols"] += result["retried_mols"]
        totals["confs"] += result["confs"]
        totals["parse_fail"] += len(parse_failures)
        totals["mmff_skipped"] += result["mmff_skipped"]

        dt = time.time() - t0
        print(
            f"    {result['confs']:,} confs in {dt:.0f}s "
            f"({result['confs'] / max(dt, 1e-9):.0f} confs/s)"
        )

        if chunk_failures and not allow_partial_conformers:
            raise RuntimeError(
                f"Chunk {chunk_idx} has {len(chunk_failures):,} molecule(s) without all "
                f"{n_conformers} requested conformers after retry. Details: {failure_report}. "
                "Use --allow-partial-conformers only if an incomplete library is acceptable."
            )

        del mols
        gc.collect()

    try:
        chunk = []
        for smiles, mol_id in _iter_smiles_file(
            smi_path, expected_provenance=input_provenance
        ):
            chunk.append((smiles, mol_id))
            if len(chunk) >= chunk_size:
                _process(chunk)
                chunk = []
        if chunk:
            _process(chunk, final=True)
    finally:
        try:
            writer.close()
        finally:
            try:
                failure_sink.close()
            finally:
                mmff_sink.close()

    dt = time.time() - t_total
    if totals["input_rows"] == 0:
        raise RuntimeError(
            "Conformer input contained no molecule records; refusing to replace "
            "the final SDF with an empty file"
        )
    requested_conformers = totals["input_rows"] * n_conformers
    if totals["confs"] + totals["unwritten_conformers"] != requested_conformers:
        raise RuntimeError(
            "Internal conformer accounting error: written + unwritten does not "
            "equal requested"
        )
    if (
        totals["conformer_shortfall"] + totals["policy_dropped_conformers"]
        != totals["unwritten_conformers"]
    ):
        raise RuntimeError(
            "Internal conformer accounting error: shortfall + policy-dropped does "
            "not equal unwritten"
        )
    if input_provenance is not None:
        verify_input_provenance([input_provenance])
    if before_commit is not None:
        before_commit()
    if desired_output_mode is not None:
        partial_output_path.chmod(desired_output_mode)
    os.replace(partial_output_path, output_path)
    totals["failure_report"] = failure_report
    totals["mmff_report"] = mmff_report
    totals["failure_count"] = failure_sink.count
    totals["failures"] = list(failure_sink.examples)
    totals["failures_truncated"] = failure_sink.count > len(failure_sink.examples)
    print(f"\n  Molecules attempted: {totals['input_rows']:,}")
    print(f"  Molecules parsed: {totals['mols']:,}")
    print(f"  Molecules complete: {totals['successful_mols']:,}")
    print(f"  Molecules incomplete: {totals['failed_mols']:,}")
    print(f"  Conformers written: {totals['confs']:,}")
    print(f"  Conformers unwritten: {totals['unwritten_conformers']:,}")
    print(f"  Embedding shortfall: {totals['conformer_shortfall']:,}")
    print(f"  Withheld by strict policy: {totals['policy_dropped_conformers']:,}")
    print(f"  Parse failures: {totals['parse_fail']:,}")
    print(f"  MMFF-unparametrisable (written unminimised): {totals['mmff_skipped']:,}")
    print(f"  Conformer failure report: {failure_report}")
    print(f"  MMFF94s report: {mmff_report}")
    print(f"  Total time: {dt:.0f}s ({dt / 3600:.2f}h)")
    if dt > 0:
        print(f"  Throughput: {totals['confs'] / dt:.0f} confs/s")
    return totals



# Pre-flight disk check + manifest


def _estimated_sdf_record_bytes(smiles, mol_id="sample"):
    """Estimate one explicit-H conformer record including pipeline properties."""
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
    smi_path,
    n_conformers,
    sample_size=DISK_SAMPLE_SIZE,
    safety_factor=DISK_SAFETY_FACTOR,
    input_provenance=None,
):
    """Estimate total SDF size from a deterministic reservoir sample."""
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")
    if safety_factor < 1:
        raise ValueError("safety_factor must be at least 1")

    rng = random.Random(DEFAULT_RANDOM_SEED)
    sample = []
    molecule_count = 0
    for smiles, mol_id in _iter_smiles_file(
        smi_path, expected_provenance=input_provenance
    ):
        record_bytes = _estimated_sdf_record_bytes(smiles, mol_id)
        if record_bytes is None:
            continue
        molecule_count += 1
        if len(sample) < sample_size:
            sample.append(record_bytes)
            continue
        replacement = rng.randrange(molecule_count)
        if replacement < sample_size:
            sample[replacement] = record_bytes

    if not sample:
        return {
            "molecule_count": 0,
            "sample_count": 0,
            "sampled_mean_bytes": 0,
            "bytes_per_conformer": MIN_BYTES_PER_CONFORMER,
            "estimated_total_bytes": 0,
            "safety_factor": safety_factor,
        }

    sampled_mean = sum(sample) / len(sample)
    bytes_per_conformer = max(
        MIN_BYTES_PER_CONFORMER,
        math.ceil(sampled_mean * safety_factor),
    )
    return {
        "molecule_count": molecule_count,
        "sample_count": len(sample),
        "sampled_mean_bytes": math.ceil(sampled_mean),
        "bytes_per_conformer": bytes_per_conformer,
        "estimated_total_bytes": molecule_count * n_conformers * bytes_per_conformer,
        "safety_factor": safety_factor,
    }


def check_disk_space(
    output_path,
    smi_path,
    n_conformers,
    margin=DISK_FREE_MARGIN,
    sample_size=DISK_SAMPLE_SIZE,
    input_provenance=None,
):
    """Refuse runs whose sampled, padded SDF estimate could fill the volume."""
    estimate_kwargs = {"sample_size": sample_size}
    if input_provenance is not None:
        estimate_kwargs["input_provenance"] = input_provenance
    estimate = estimate_sdf_bytes(smi_path, n_conformers, **estimate_kwargs)
    est_bytes = estimate["estimated_total_bytes"]
    target = Path(output_path).resolve().parent
    free = shutil.disk_usage(target).free

    def gb(x):
        return x / 1024 ** 3

    print(
        f"\n  Estimated output size: {gb(est_bytes):.1f} GB "
        f"({estimate['molecule_count']:,} mols x {n_conformers} confs; "
        f"{estimate['sample_count']:,}-molecule sample, "
        f"{estimate['safety_factor']:.0%} size factor)"
    )
    print(f"  Free space on {target}: {gb(free):.1f} GB")

    if est_bytes > free * margin:
        raise RuntimeError(
            f"Insufficient disk space. Estimated {gb(est_bytes):.1f} GB needed, "
            f"{gb(free):.1f} GB free on {target}.\n"
            "  Reduce --n-conformers, split the input, write to a larger volume, "
            "or pass --skip-disk-check to override."
        )
    return estimate


def _sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(block), b""):
            h.update(buf)
    return h.hexdigest()


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
        f"Input changed while the pipeline was reading it: {Path(path).resolve()}. "
        "The run was not marked successful; rerun with an unchanged input file."
    )


def _verify_observed_input(
    path,
    expected_provenance,
    stat_before,
    stat_after,
    observed_sha256=None,
):
    """Prove that bytes consumed from an input match the captured snapshot."""
    before = _stat_fingerprint(stat_before)
    after = _stat_fingerprint(stat_after)
    try:
        current = _stat_fingerprint(Path(path).stat())
    except OSError:
        _raise_input_changed(path)

    if before != after or after != current:
        _raise_input_changed(path)
    for key in ("device", "inode", "bytes", "mtime_ns", "ctime_ns"):
        if key in expected_provenance and after[key] != expected_provenance[key]:
            _raise_input_changed(path)
    expected_digest = expected_provenance.get("sha256")
    if expected_digest is not None and observed_sha256 != expected_digest:
        _raise_input_changed(path)


def capture_input_provenance(paths, hash_inputs=True):
    """Capture a stable source snapshot before long-running pipeline work."""
    inputs = []
    for path in paths:
        source = Path(path).expanduser().resolve(strict=True)
        digest = hashlib.sha256() if hash_inputs else None
        with open(source, "rb") as handle:
            stat_before = os.fstat(handle.fileno())
            if digest is not None:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            stat_after = os.fstat(handle.fileno())
        fingerprint = _stat_fingerprint(stat_after)
        if (
            _stat_fingerprint(stat_before) != fingerprint
            or _stat_fingerprint(source.stat()) != fingerprint
        ):
            _raise_input_changed(source)
        entry = {
            "path": str(source),
            **fingerprint,
        }
        if hash_inputs:
            entry["sha256"] = digest.hexdigest()
        inputs.append(entry)
    return inputs


def verify_input_provenance(input_provenance):
    """Refuse success if any captured input no longer matches its snapshot."""
    for expected in input_provenance:
        current = capture_input_provenance(
            [expected["path"]], hash_inputs="sha256" in expected
        )[0]
        for key in ("device", "inode", "bytes", "mtime_ns", "ctime_ns", "sha256"):
            if key in expected and current.get(key) != expected[key]:
                _raise_input_changed(expected["path"])


def _write_json_atomic(path, payload):
    """Replace a JSON sidecar atomically so readers never see a torn file."""
    destination = Path(path)
    _validate_regular_output_path(destination, "JSON sidecar")
    destination.parent.mkdir(parents=True, exist_ok=True)
    desired_mode = (
        stat.S_IMODE(destination.stat().st_mode) if destination.is_file() else None
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    for _ in range(100):
        temporary = destination.with_name(
            f".{destination.name}.{secrets.token_hex(6)}.tmp"
        )
        try:
            descriptor = os.open(temporary, flags, 0o666)
        except FileExistsError:
            continue
        try:
            if desired_mode is not None:
                os.fchmod(descriptor, desired_mode)
        except Exception:
            os.close(descriptor)
            temporary.unlink(missing_ok=True)
            raise
        break
    else:
        raise RuntimeError(f"Could not allocate a temporary JSON file beside {destination}")
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_run_marker(path, args, params, input_provenance=None):
    """Invalidate any previous success manifest before this run mutates outputs."""
    inputs = input_provenance
    if inputs is None:
        inputs = []
        for input_path in args.input:
            source = Path(input_path)
            entry = {"path": str(source.resolve()), "exists": source.is_file()}
            if entry["exists"]:
                stat_result = source.stat()
                entry["bytes"] = stat_result.st_size
                entry["mtime_ns"] = stat_result.st_mtime_ns
            inputs.append(entry)
    marker = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "2.1",
        "status": "in_progress",
        "preset": args.preset,
        "inputs": inputs,
        "planned_output_sdf": (
            str(Path(args.output).resolve()) if not args.skip_conformers else None
        ),
        "output_sdf": None,
        "parameters": {
            **params,
            "random_seed": args.random_seed,
            "allow_partial_conformers": args.allow_partial_conformers,
        },
        "note": "Replaced by a status=succeeded manifest only after all outputs finish.",
    }
    _write_json_atomic(path, marker)


def write_manifest(
    path,
    args,
    params,
    counts,
    timings,
    hash_inputs=True,
    artifacts=None,
    input_provenance=None,
):
    """Emit a JSON sidecar describing inputs, resolved parameters, and outputs."""
    import rdkit

    if input_provenance is not None:
        verify_input_provenance(input_provenance)
    inputs = (
        [dict(entry) for entry in input_provenance]
        if input_provenance is not None
        else capture_input_provenance(args.input, hash_inputs=hash_inputs)
    )

    artifact_entries = {}
    for name, artifact_path in (artifacts or {}).items():
        if artifact_path is None:
            continue
        artifact = Path(artifact_path)
        entry = {"path": str(artifact.resolve()), "exists": artifact.is_file()}
        if entry["exists"]:
            entry["bytes"] = artifact.stat().st_size
        artifact_entries[name] = entry

    output_sdf = None
    if not args.skip_conformers:
        output_sdf = str(Path(args.output).resolve())

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "2.1",
        "status": "succeeded",
        "preset": args.preset,
        "inputs": inputs,
        "output_sdf": output_sdf,
        "artifacts": artifact_entries,
        "parameters": params,
        "stage_counts": counts,
        "timings_seconds": timings,
        "versions": {
            "python": sys.version.split()[0],
            "rdkit": rdkit.__version__,
            "nvmolkit": _nvmolkit_version(),
            "dimorphite_dl": DIMORPHITE_VERSION,
            "pandas": pd.__version__,
        },
    }
    _write_json_atomic(path, manifest)
    print(f"  Manifest: {path}")



# CLI


def _validate_pipeline_paths(args, params):
    """Reject any planned output that aliases an input or another output."""
    out_path = Path(args.output)
    stem = str(out_path.with_suffix(""))
    protected = {
        f"input[{index}]": Path(path).expanduser().resolve()
        for index, path in enumerate(args.input)
    }
    if args.custom_smarts:
        protected["custom SMARTS"] = Path(args.custom_smarts).expanduser().resolve()

    planned = {
        "final SMILES": Path(f"{stem}_final.smi"),
        "final metadata": Path(f"{stem}_final_metadata.csv"),
        "manifest": Path(f"{stem}_manifest.json"),
        "output lock": _output_lock_path(out_path),
    }
    if not args.skip_conformers:
        planned.update({
            "output SDF": out_path,
            "conformer failure report": Path(f"{stem}_conformer_failures.csv"),
            "MMFF94s report": Path(f"{stem}_mmff_unparametrizable.csv"),
        })
    if args.save_intermediates:
        planned.update({
            "merged intermediate": Path(f"{stem}_01_merged.csv"),
            "salt intermediate": Path(f"{stem}_02_salts_stripped.csv"),
            "filtered intermediate": Path(f"{stem}_03_filtered.csv"),
            "filter rejects": Path(f"{stem}_03_failed.csv"),
            "stereo intermediate": Path(f"{stem}_04_stereo.csv"),
            "deduplicated intermediate": Path(f"{stem}_05_deduplicated.csv"),
        })
        if params["tautomers"]:
            planned["tautomer intermediate"] = Path(f"{stem}_04b_tautomers.csv")
        if params["ionise"]:
            planned["ionisation intermediate"] = Path(f"{stem}_06_ionised.csv")

    resolved_outputs = {}
    for label, path in planned.items():
        _validate_regular_output_path(path, label)
        resolved = path.expanduser().resolve()
        for protected_label, protected_path in protected.items():
            if _paths_alias(resolved, protected_path):
                raise ValueError(
                    f"Planned {label} path aliases {protected_label}: {resolved}"
                )
        for other_path, other_label in resolved_outputs.items():
            if _paths_alias(resolved, other_path):
                raise ValueError(
                    f"Planned {label} path aliases {other_label}: {resolved}"
                )
        resolved_outputs[resolved] = label


def build_parser():
    p = argparse.ArgumentParser(
        description="GPU-accelerated chemical library preparation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Presets:\n"
            "  docking   (default) single protonation state @ pH 7.4, no tautomer\n"
            "            enumeration, 1 conformer. What you want for virtual screening.\n"
            "  enumerate LigPrep-style expansion: tautomers on, pH 6.4-8.4.\n"
            "            Produces a substantially larger library.\n\n"
            "Any explicit flag overrides the preset."
        ),
    )
    p.add_argument("--input", nargs="+", required=True, help="Input SMILES/cxsmiles files")
    p.add_argument("--output", default="library_3d.sdf", help="Output SDF (default: library_3d.sdf)")
    p.add_argument("--preset", choices=list(PRESETS), default="docking",
                   help="Parameter preset (default: docking)")

    # Sentinels: None means "take it from the preset".
    p.add_argument("--n-conformers", type=int, default=None, help="Conformers per molecule")
    p.add_argument("--max-unspecified-stereo", type=int, default=None,
                   help="Max unspecified atom/bond stereo elements before rejection")
    p.add_argument("--max-tautomers", type=int, default=None, help="Max tautomers per molecule")
    p.add_argument("--ph", type=float, default=None,
                   help="Single ionisation pH (sets ph-min = ph-max, one state per molecule)")
    p.add_argument("--ph-min", type=float, default=None, help="Min pH for ionisation")
    p.add_argument("--ph-max", type=float, default=None, help="Max pH for ionisation")

    p.add_argument("--tautomers", dest="tautomers", action="store_true", default=None,
                   help="Force tautomer enumeration on")
    p.add_argument("--skip-tautomers", dest="tautomers", action="store_false",
                   help="Force tautomer enumeration off")
    p.add_argument("--skip-ionise", dest="ionise", action="store_false", default=None,
                   help="Skip ionisation entirely")
    p.add_argument("--skip-conformers", action="store_true",
                   help="Stop after the 2D stages; emit SMILES only")

    p.add_argument("--custom-smarts", default=None,
                   help="File of SMARTS rejection patterns (one per line, optional name)")
    p.add_argument("--pains-backend", choices=["auto", "cpu", "gpu"], default="auto",
                   help="PAINS backend (default: auto). Use cpu if wehi_pains.csv is missing.")

    # GPU / memory knobs. Defaults are conservative; large-VRAM cards can go higher.
    p.add_argument("--chunk-size", type=int, default=100_000,
                   help="Molecules held in RAM per conformer chunk (default: 100000)")
    p.add_argument("--batch-size", type=int, default=500,
                   help="nvMolKit GPU batch size (default: 500; -1 lets nvMolKit choose)")
    p.add_argument("--batches-per-gpu", type=int, default=4,
                   help="nvMolKit batches per GPU (default: 4; -1 automatic)")
    p.add_argument("--preprocessing-threads", type=int, default=8,
                   help="CPU preprocessing threads (default: 8; -1 automatic)")
    p.add_argument("--mmff-max-iters", type=int, default=200, help="MMFF94s max iterations")
    p.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Deterministic ETKDG seed (default: {DEFAULT_RANDOM_SEED})",
    )
    p.add_argument(
        "--allow-partial-conformers",
        action="store_true",
        help="Succeed with a failure report when some molecules remain incomplete after retry",
    )

    p.add_argument("--skip-disk-check", action="store_true", help="Bypass the pre-flight disk check")
    p.add_argument("--skip-smoke-test", action="store_true", help="Bypass the GPU smoke test")
    p.add_argument("--no-hash-inputs", action="store_true",
                   help="Skip SHA-256 of inputs in the manifest (faster on huge files)")
    p.add_argument("--save-intermediates", action="store_true", help="Save per-stage CSVs")
    return p


def resolve_params(args):
    """Merge preset defaults with any explicitly-supplied flags."""
    params = dict(PRESETS[args.preset])

    if args.n_conformers is not None:
        params["n_conformers"] = args.n_conformers
    if args.max_unspecified_stereo is not None:
        params["max_unspecified_stereo"] = args.max_unspecified_stereo
    if args.max_tautomers is not None:
        params["max_tautomers"] = args.max_tautomers
    if args.tautomers is not None:
        params["tautomers"] = args.tautomers
    if args.ionise is not None:
        params["ionise"] = args.ionise

    if args.ph is not None:
        params["ph_min"] = params["ph_max"] = args.ph
    if args.ph_min is not None:
        params["ph_min"] = args.ph_min
    if args.ph_max is not None:
        params["ph_max"] = args.ph_max

    if params["ph_min"] > params["ph_max"]:
        raise ValueError("--ph-min must be <= --ph-max")
    if not 0 <= params["ph_min"] <= params["ph_max"] <= 14:
        raise ValueError("pH values must be between 0 and 14")
    if params["n_conformers"] < 1:
        raise ValueError("--n-conformers must be at least 1")
    if not 0 <= params["max_unspecified_stereo"] <= 4:
        raise ValueError("--max-unspecified-stereo must be between 0 and 4")
    if params["max_tautomers"] < 1:
        raise ValueError("--max-tautomers must be at least 1")
    positive_runtime_args = {
        "--chunk-size": args.chunk_size,
        "--mmff-max-iters": args.mmff_max_iters,
    }
    for flag, value in positive_runtime_args.items():
        if value < 1:
            raise ValueError(f"{flag} must be at least 1")
    auto_tunable_runtime_args = {
        "--batch-size": args.batch_size,
        "--batches-per-gpu": args.batches_per_gpu,
        "--preprocessing-threads": args.preprocessing_threads,
    }
    for flag, value in auto_tunable_runtime_args.items():
        if value == 0 or value < -1:
            raise ValueError(f"{flag} must be -1 (automatic) or at least 1")
    if not 0 <= args.random_seed < 2**31 - 1:
        raise ValueError("--random-seed must be between 0 and 2147483646")

    return params


def _with_pipeline_output_lock(entrypoint):
    def locked_entrypoint():
        args = build_parser().parse_args()
        params = resolve_params(args)
        _validate_pipeline_paths(args, params)
        out_path = Path(args.output)
        output_lock = _OutputLock(out_path)
        try:
            return entrypoint(args, params, out_path)
        finally:
            output_lock.close()

    return locked_entrypoint


@_with_pipeline_output_lock
def main(args, params, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = str(out_path.with_suffix(""))
    manifest_path = f"{stem}_manifest.json"
    supplier_provenance = capture_input_provenance(
        args.input, hash_inputs=not args.no_hash_inputs
    )
    for entry in supplier_provenance:
        entry["role"] = "molecule_input"
    custom_smarts_provenance = None
    if args.custom_smarts:
        custom_smarts_provenance = capture_input_provenance(
            [args.custom_smarts], hash_inputs=not args.no_hash_inputs
        )[0]
        custom_smarts_provenance["role"] = "custom_smarts"
    input_provenance = list(supplier_provenance)
    if custom_smarts_provenance is not None:
        input_provenance.append(custom_smarts_provenance)
    write_run_marker(
        manifest_path, args, params, input_provenance=input_provenance
    )

    t_total = time.time()
    timings = {}
    counts = {}

    print("=" * 60)
    print("CHEMICAL LIBRARY PREPARATION PIPELINE")
    print("=" * 60)
    print(f"Input:   {', '.join(args.input)}")
    print(f"Output:  {args.output}")
    print(f"Preset:  {args.preset}")
    print(f"Params:  {json.dumps(params)}")

    # Fail fast on a dead GPU rather than after the CPU stages.
    if not args.skip_conformers and not args.skip_smoke_test:
        print("\n  Running GPU smoke test...")
        gpu_smoke_test(random_seed=args.random_seed)

    def stage(name, fn, *a, **kw):
        t0 = time.time()
        result = fn(*a, **kw)
        timings[name] = round(time.time() - t0, 1)
        return result

    df = stage(
        "merge", merge_suppliers, args.input, supplier_provenance
    )
    counts["merged"] = len(df)
    if df.empty:
        print("\nNo molecules were found in the input files. Nothing to do.")
        sys.exit(1)
    if args.save_intermediates:
        df.to_csv(f"{stem}_01_merged.csv", index=False)

    df = stage("salts", strip_salts, df)
    counts["after_salts"] = len(df)
    if args.save_intermediates:
        df.to_csv(f"{stem}_02_salts_stripped.csv", index=False)

    df, failed_df = stage("filters", apply_filters, df,
                          pains_backend=args.pains_backend,
                          custom_smarts=args.custom_smarts,
                          custom_smarts_provenance=custom_smarts_provenance)
    counts["after_filters"] = len(df)
    counts["filter_rejects"] = len(failed_df)
    if args.save_intermediates:
        df.to_csv(f"{stem}_03_filtered.csv", index=False)
        failed_df.to_csv(f"{stem}_03_failed.csv", index=False)

    if df.empty:
        print("\nNo molecules survived filtering. Nothing to do.")
        sys.exit(1)

    df = stage("stereo", filter_and_enumerate_stereo, df,
               max_unspecified=params["max_unspecified_stereo"])
    counts["after_stereo"] = len(df)
    if df.empty:
        print("\nNo molecules survived stereochemistry filtering. Nothing to do.")
        sys.exit(1)
    if args.save_intermediates:
        df.to_csv(f"{stem}_04_stereo.csv", index=False)

    if params["tautomers"]:
        df = stage("tautomers", enumerate_tautomers, df,
                   max_tautomers=params["max_tautomers"])
        counts["after_tautomers"] = len(df)
        if df.empty:
            print("\nTautomer enumeration produced no valid molecules. Nothing to do.")
            sys.exit(1)
        if args.save_intermediates:
            df.to_csv(f"{stem}_04b_tautomers.csv", index=False)

    df = stage("dedup", deduplicate, df)
    counts["after_dedup"] = len(df)
    if df.empty:
        print("\nNo valid molecules remained after deduplication. Nothing to do.")
        sys.exit(1)
    if args.save_intermediates:
        df.to_csv(f"{stem}_05_deduplicated.csv", index=False)

    if params["ionise"]:
        df = stage("ionise", ionise_molecules, df,
                   ph_min=params["ph_min"], ph_max=params["ph_max"])
        df = canonical_redup(df)
        counts["after_ionise"] = len(df)
        if df.empty:
            print("\nIonisation produced no valid molecules. Nothing to do.")
            sys.exit(1)
        if args.save_intermediates:
            df.to_csv(f"{stem}_06_ionised.csv", index=False)

    
    meta_csv = Path(f"{stem}_final_metadata.csv")
    meta_mode = (
        stat.S_IMODE(meta_csv.stat().st_mode) if meta_csv.is_file() else None
    )
    staged_meta_csv = _create_staged_sdf(meta_csv)
    df.to_csv(staged_meta_csv, index=False)

    final_smi = Path(f"{stem}_final.smi")
    final_smi_mode = (
        stat.S_IMODE(final_smi.stat().st_mode) if final_smi.is_file() else None
    )
    staged_final_smi = _create_staged_sdf(final_smi)
    with open(staged_final_smi, "w") as f:
        for smi, mol_id in zip(df["SMILES"], df["ID"]):
            f.write(f"{smi}\t{mol_id}\n")
    final_smi_provenance = capture_input_provenance(
        [staged_final_smi], hash_inputs=not args.no_hash_inputs
    )[0]
    final_smi_provenance["role"] = "derived_conformer_input"

    n_final = len(df)
    counts["final_2d"] = n_final
    print(f"\n  Final SMILES:   {final_smi}  ({n_final:,} molecules)")
    print(f"  Final metadata: {meta_csv}")

    del df
    gc.collect()

    conf_totals = None
    disk_estimate = None
    if not args.skip_conformers:
        if not args.skip_disk_check:
            disk_estimate = check_disk_space(
                args.output,
                staged_final_smi,
                params["n_conformers"],
                input_provenance=final_smi_provenance,
            )

        conf_totals = stage(
            "conformers", generate_conformers,
            staged_final_smi, args.output,
            n_conformers=params["n_conformers"],
            chunk_size=args.chunk_size,
            mmff_max_iters=args.mmff_max_iters,
            batch_size=args.batch_size,
            batches_per_gpu=args.batches_per_gpu,
            preprocessing_threads=args.preprocessing_threads,
            random_seed=args.random_seed,
            allow_partial_conformers=args.allow_partial_conformers,
            check_free_space=not args.skip_disk_check,
            estimated_bytes_per_conformer=(
                disk_estimate["bytes_per_conformer"] if disk_estimate else None
            ),
            output_lock_held=True,
            before_commit=lambda: verify_input_provenance(input_provenance),
            input_provenance=final_smi_provenance,
        )
        counts["conformers_written"] = conf_totals["confs"]
        counts["conformer_molecules_complete"] = conf_totals["successful_mols"]
        counts["conformer_molecules_incomplete"] = conf_totals["failed_mols"]
        counts["conformer_shortfall"] = conf_totals["conformer_shortfall"]
        counts["conformers_unwritten"] = conf_totals["unwritten_conformers"]
        counts["conformers_policy_dropped"] = conf_totals[
            "policy_dropped_conformers"
        ]
        counts["conformer_retried"] = conf_totals["retried_mols"]
        counts["mmff_unparametrisable"] = conf_totals["mmff_skipped"]

    timings["total"] = round(time.time() - t_total, 1)

    runtime_params = dict(params)
    runtime_params.update({
        "chunk_size": args.chunk_size,
        "batch_size": args.batch_size,
        "batches_per_gpu": args.batches_per_gpu,
        "preprocessing_threads": args.preprocessing_threads,
        "mmff_max_iters": args.mmff_max_iters,
        "force_field": "MMFF94s",
        "random_seed": args.random_seed,
        "allow_partial_conformers": args.allow_partial_conformers,
        "pains_backend_requested": args.pains_backend,
        "pains_backend_resolved": resolve_pains_backend(args.pains_backend),
        "custom_smarts": args.custom_smarts,
        "skip_conformers": args.skip_conformers,
        "disk_estimate": disk_estimate,
    })
    verify_input_provenance(input_provenance)
    verify_input_provenance([final_smi_provenance])
    if meta_mode is not None:
        staged_meta_csv.chmod(meta_mode)
    if final_smi_mode is not None:
        staged_final_smi.chmod(final_smi_mode)
    os.replace(staged_meta_csv, meta_csv)
    os.replace(staged_final_smi, final_smi)

    artifacts = {
        "final_smiles": final_smi,
        "final_metadata": meta_csv,
        "output_sdf": args.output if not args.skip_conformers else None,
        "conformer_failures": conf_totals["failure_report"] if conf_totals else None,
        "mmff_unparametrizable": conf_totals["mmff_report"] if conf_totals else None,
    }
    write_manifest(
        manifest_path,
        args,
        runtime_params,
        counts,
        timings,
        hash_inputs=not args.no_hash_inputs,
        artifacts=artifacts,
        input_provenance=input_provenance,
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Runtime: {timings['total']:.0f}s ({timings['total'] / 3600:.2f}h)")
    if args.skip_conformers:
        print(f"2D output: {final_smi}")
    else:
        print(f"Output:  {args.output}")


if __name__ == "__main__":
    main()
