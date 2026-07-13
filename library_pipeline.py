import argparse
import csv
import gc
import hashlib
import json
import os
import shutil
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

BYTES_PER_CONFORMER = 4_500

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
   
    try:
        from dimorphite_dl.protonate import protonate_smiles as _p

        def fn(smi, min_ph, max_ph, precision):
            try:
                return _p(smi, min_ph=min_ph, max_ph=max_ph, precision=precision)
            except TypeError:
                return _p(smi, min_ph=min_ph, max_ph=max_ph)

        return fn, _dimorphite_version()
    except ImportError:
        pass

    try:
        import dimorphite_dl

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


def load_supplier_file(filepath, supplier_name=None):
    """Load a supplier SMILES/cxsmiles file into a DataFrame."""

    
    if supplier_name is None:
        supplier_name = Path(filepath).stem

    records = []
    with open(filepath, "r") as f:
        for line_num, line in enumerate(f):
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

    df = pd.DataFrame(records)
    print(f"  Loaded {len(df):,} molecules from {supplier_name}")
    return df


def merge_suppliers(supplier_files):
    print("\n" + "=" * 60)
    print("STEP 1: LOAD AND MERGE SUPPLIER CATALOGUES")
    print("=" * 60)
    frames = [load_supplier_file(f, Path(f).stem) for f in supplier_files]
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

def apply_filters(df, pains_backend="auto", custom_smarts=None):
    """Complexity, BRENK, Lipinski, rings, aggregator, PAINS, optional custom SMARTS.

    PAINS is batched on GPU when nvMolKit is available; BRENK stays on CPU
    (RDKit ships no public BRENK SMARTS file). Returns (passed_df, failed_df).
    """
    print("\n" + "=" * 60)
    print("STEP 3: COMPOUND FILTERING")
    print("=" * 60)

    if pains_backend == "auto":
        backend = "gpu" if _NVMOLKIT_AVAILABLE else "cpu"
    else:
        backend = pains_backend
    if backend == "gpu" and not _NVMOLKIT_AVAILABLE:
        print("  WARNING: --pains-backend gpu requested but nvMolKit not importable. "
              "Falling back to CPU.")
        backend = "cpu"
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
        custom_queries = load_custom_smarts(custom_smarts)
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


def load_custom_smarts(path):
    """Load user-supplied SMARTS rejection patterns."""

    
    queries = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
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
    return queries



# Step 4: stereo


def count_unspecified_stereocentres(mol):
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    stereo_info = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False
    )
    return sum(1 for _, label in stereo_info if label == "?")


def filter_and_enumerate_stereo(df, max_unspecified=2):
    print("\n" + "=" * 60)
    print("STEP 4: STEREO FILTERING + ENUMERATION")
    print("=" * 60)

    filtered_out = 0
    records = []
    opts = StereoEnumerationOptions(tryEmbedding=False, onlyUnassigned=True, maxIsomers=16)

    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol is None:
            continue

        n_unspec = count_unspecified_stereocentres(mol)
        if n_unspec > max_unspecified:
            filtered_out += 1
            continue

        if n_unspec == 0:
            records.append(row.to_dict())
        else:
            for iso_idx, iso_mol in enumerate(EnumerateStereoisomers(mol, options=opts)):
                rec = row.to_dict()
                rec["SMILES"] = Chem.MolToSmiles(iso_mol, isomericSmiles=True)
                rec["ID"] = f"{row['ID']}_iso{iso_idx + 1}"
                records.append(rec)

    result = pd.DataFrame(records)
    print(f"  Removed (>{max_unspecified} unspecified centres): {filtered_out:,}")
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
                    rec["ID"] = f"{row['ID']}_pH{v_idx + 1}"
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
    """Canonicalise and drop exact duplicates (used after ionisation)."""
    before = len(df)
    df = df.copy()
    df["canonical"] = df["SMILES"].apply(
        lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=True)
        if Chem.MolFromSmiles(s) is not None else None
    )
    df = df.dropna(subset=["canonical"]).drop_duplicates(subset=["canonical"])
    df["SMILES"] = df["canonical"]
    df = df.drop(columns=["canonical"])
    print(f"  Re-dedup: {before:,} -> {len(df):,}")
    return df



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


def gpu_smoke_test():
    """Embed one trivial molecule and verify real 3D coordinates come back."""

    
    
    EmbedMolecules, MMFFOptimizeMoleculesConfs, HardwareOptions = _import_nvmolkit()
    from rdkit.Chem.rdDistGeom import ETKDGv3

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    params = ETKDGv3()
    params.useRandomCoords = True
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

    MMFFOptimizeMoleculesConfs([mol], maxIters=50, hardwareOptions=hw)
    print("  GPU smoke test passed (ethanol embedded + minimised).")


def _iter_smiles_file(path):
    """Yield (smiles, mol_id) from the tab-delimited final SMILES file."""
    with open(path) as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            fields = line.split("\t")
            smiles = fields[0].strip()
            mol_id = fields[1].strip() if len(fields) > 1 else f"mol_{line_num}"
            if smiles.upper() in ("SMILES", "CANONICAL_SMILES", "SMI", "SMILE"):
                continue
            yield smiles, mol_id


def _load_chunk(rows):
    mols = []
    parse_fail = 0
    for smiles, mol_id in rows:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            parse_fail += 1
            continue
        m = Chem.AddHs(m)
        m.SetProp("_Name", str(mol_id))
        mols.append(m)
    return mols, parse_fail


def _embed_and_write(mols, writer, n_conformers, mmff_max_iters, hw):
    EmbedMolecules, MMFFOptimizeMoleculesConfs, _ = _import_nvmolkit()
    from rdkit.Chem.rdDistGeom import ETKDGv3

    params = ETKDGv3()
    params.useRandomCoords = True  # required by nvMolKit

    EmbedMolecules(mols, params, confsPerMolecule=n_conformers, hardwareOptions=hw)

   
    mmff_ok, mmff_bad = [], []
    for m in mols:
        if AllChem.MMFFGetMoleculeProperties(m, mmffVariant="MMFF94s") is None:
            mmff_bad.append(m)
        else:
            mmff_ok.append(m)

    energies = MMFFOptimizeMoleculesConfs(mmff_ok, maxIters=mmff_max_iters,
                                          hardwareOptions=hw) if mmff_ok else []

    n_written = 0
    for m, mol_energies in zip(mmff_ok, energies):
        m.SetProp("MMFF_Minimised", "True")
        for cid in range(m.GetNumConformers()):
            if cid < len(mol_energies):
                m.SetProp("MMFF_Energy", f"{mol_energies[cid]:.3f}")
            elif m.HasProp("MMFF_Energy"):
                m.ClearProp("MMFF_Energy")  # never carry a stale energy forward
            writer.write(m, confId=cid)
            n_written += 1

    for m in mmff_bad:
        m.SetProp("MMFF_Minimised", "False")
        for cid in range(m.GetNumConformers()):
            writer.write(m, confId=cid)
            n_written += 1

    return n_written, len(mmff_bad)


def generate_conformers(smi_path, output_sdf, n_conformers=1, chunk_size=100_000,
                        mmff_max_iters=200, batch_size=500, batches_per_gpu=4,
                        preprocessing_threads=8, gpu_ids=None):
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

    print(f"  chunk-size: {chunk_size:,}  batch-size: {batch_size}  "
          f"batches/gpu: {batches_per_gpu}")

    t_total = time.time()
    totals = {"mols": 0, "confs": 0, "parse_fail": 0, "mmff_skipped": 0}
    chunk_idx = 0
    writer = SDWriter(str(output_sdf))

    def _process(rows, final=False):
        nonlocal chunk_idx
        chunk_idx += 1
        t0 = time.time()
        mols, parse_fail = _load_chunk(rows)
        tag = " (final)" if final else ""
        print(f"  Chunk {chunk_idx}{tag}: {len(mols):,} mols ({parse_fail} parse failures)")

        if not mols:
            totals["parse_fail"] += parse_fail
            return

        n_written, n_bad = _embed_and_write(mols, writer, n_conformers, mmff_max_iters, hw)

        
        
        if n_written == 0:
            raise RuntimeError(
                f"Chunk {chunk_idx}: {len(mols):,} molecules in, 0 conformers out.\n"
                "  This is the silent GPU failure mode, usually VRAM exhaustion.\n"
                f"  Retry with a smaller --batch-size (currently {batch_size}) "
                f"and/or --chunk-size (currently {chunk_size:,})."
            )

        totals["mols"] += len(mols)
        totals["confs"] += n_written
        totals["parse_fail"] += parse_fail
        totals["mmff_skipped"] += n_bad

        dt = time.time() - t0
        print(f"    {n_written:,} confs in {dt:.0f}s ({n_written / max(dt, 1e-9):.0f} confs/s)")

        del mols
        gc.collect()

    chunk = []
    for smiles, mol_id in _iter_smiles_file(smi_path):
        chunk.append((smiles, mol_id))
        if len(chunk) >= chunk_size:
            _process(chunk)
            chunk = []
    if chunk:
        _process(chunk, final=True)

    writer.close()

    dt = time.time() - t_total
    print(f"\n  Molecules: {totals['mols']:,}")
    print(f"  Conformers written: {totals['confs']:,}")
    print(f"  Parse failures: {totals['parse_fail']:,}")
    print(f"  MMFF-unparametrisable (written unminimised): {totals['mmff_skipped']:,}")
    print(f"  Total time: {dt:.0f}s ({dt / 3600:.2f}h)")
    if dt > 0:
        print(f"  Throughput: {totals['confs'] / dt:.0f} confs/s")
    return totals



# Pre-flight disk check + manifest


def check_disk_space(output_path, n_molecules, n_conformers, margin=0.90):
    """Refuse to start a run that will plausibly fill the volume.

    Better to fail in one second than six hours in with a half-written SDF.
    """
    est_bytes = n_molecules * n_conformers * BYTES_PER_CONFORMER
    target = Path(output_path).resolve().parent
    free = shutil.disk_usage(target).free

    def gb(x):
        return x / 1024 ** 3

    print(f"\n  Estimated output size: {gb(est_bytes):.1f} GB "
          f"({n_molecules:,} mols x {n_conformers} confs)")
    print(f"  Free space on {target}: {gb(free):.1f} GB")

    if est_bytes > free * margin:
        raise RuntimeError(
            f"Insufficient disk space. Estimated {gb(est_bytes):.1f} GB needed, "
            f"{gb(free):.1f} GB free on {target}.\n"
            "  Reduce --n-conformers, split the input, write to a larger volume, "
            "or pass --skip-disk-check to override."
        )


def _sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(block), b""):
            h.update(buf)
    return h.hexdigest()


def write_manifest(path, args, params, counts, timings, hash_inputs=True):
    """Emit a JSON sidecar: same input + same manifest => same library."""
    import rdkit

    inputs = []
    for p in args.input:
        entry = {"path": str(Path(p).resolve()), "bytes": os.path.getsize(p)}
        if hash_inputs:
            entry["sha256"] = _sha256(p)
        inputs.append(entry)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "2.0",
        "preset": args.preset,
        "inputs": inputs,
        "output_sdf": str(Path(args.output).resolve()),
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
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest: {path}")



# CLI


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
                   help="Max unspecified stereocentres before rejection")
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
                   help="nvMolKit GPU batch size (default: 500; lower for <16GB VRAM)")
    p.add_argument("--batches-per-gpu", type=int, default=4, help="nvMolKit batches per GPU")
    p.add_argument("--preprocessing-threads", type=int, default=8, help="CPU preprocessing threads")
    p.add_argument("--mmff-max-iters", type=int, default=200, help="MMFF94s max iterations")

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

    return params


def main():
    args = build_parser().parse_args()
    params = resolve_params(args)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = str(out_path.with_suffix(""))

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
        gpu_smoke_test()

    def stage(name, fn, *a, **kw):
        t0 = time.time()
        result = fn(*a, **kw)
        timings[name] = round(time.time() - t0, 1)
        return result

    df = stage("merge", merge_suppliers, args.input)
    counts["merged"] = len(df)
    if args.save_intermediates:
        df.to_csv(f"{stem}_01_merged.csv", index=False)

    df = stage("salts", strip_salts, df)
    counts["after_salts"] = len(df)
    if args.save_intermediates:
        df.to_csv(f"{stem}_02_salts_stripped.csv", index=False)

    df, failed_df = stage("filters", apply_filters, df,
                          pains_backend=args.pains_backend,
                          custom_smarts=args.custom_smarts)
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
    if args.save_intermediates:
        df.to_csv(f"{stem}_04_stereo.csv", index=False)

    if params["tautomers"]:
        df = stage("tautomers", enumerate_tautomers, df,
                   max_tautomers=params["max_tautomers"])
        counts["after_tautomers"] = len(df)
        if args.save_intermediates:
            df.to_csv(f"{stem}_04b_tautomers.csv", index=False)

    df = stage("dedup", deduplicate, df)
    counts["after_dedup"] = len(df)
    if args.save_intermediates:
        df.to_csv(f"{stem}_05_deduplicated.csv", index=False)

    if params["ionise"]:
        df = stage("ionise", ionise_molecules, df,
                   ph_min=params["ph_min"], ph_max=params["ph_max"])
        df = canonical_redup(df)
        counts["after_ionise"] = len(df)
        if args.save_intermediates:
            df.to_csv(f"{stem}_06_ionised.csv", index=False)

    
    meta_csv = f"{stem}_final_metadata.csv"
    df.to_csv(meta_csv, index=False)

    final_smi = f"{stem}_final.smi"
    with open(final_smi, "w") as f:
        for smi, mol_id in zip(df["SMILES"], df["ID"]):
            f.write(f"{smi}\t{mol_id}\n")

    n_final = len(df)
    counts["final_2d"] = n_final
    print(f"\n  Final SMILES:   {final_smi}  ({n_final:,} molecules)")
    print(f"  Final metadata: {meta_csv}")

    del df
    gc.collect()

    if not args.skip_conformers:
        if not args.skip_disk_check:
            check_disk_space(args.output, n_final, params["n_conformers"])

        conf_totals = stage(
            "conformers", generate_conformers,
            final_smi, args.output,
            n_conformers=params["n_conformers"],
            chunk_size=args.chunk_size,
            mmff_max_iters=args.mmff_max_iters,
            batch_size=args.batch_size,
            batches_per_gpu=args.batches_per_gpu,
            preprocessing_threads=args.preprocessing_threads,
        )
        counts["conformers_written"] = conf_totals["confs"]
        counts["mmff_unparametrisable"] = conf_totals["mmff_skipped"]

    timings["total"] = round(time.time() - t_total, 1)

    manifest_path = f"{stem}_manifest.json"
    runtime_params = dict(params)
    runtime_params.update({
        "chunk_size": args.chunk_size,
        "batch_size": args.batch_size,
        "batches_per_gpu": args.batches_per_gpu,
        "mmff_max_iters": args.mmff_max_iters,
        "pains_backend": args.pains_backend,
        "custom_smarts": args.custom_smarts,
        "skip_conformers": args.skip_conformers,
    })
    write_manifest(manifest_path, args, runtime_params, counts, timings,
                   hash_inputs=not args.no_hash_inputs)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Runtime: {timings['total']:.0f}s ({timings['total'] / 3600:.2f}h)")
    print(f"Output:  {args.output}")


if __name__ == "__main__":
    main()