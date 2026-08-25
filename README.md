# HoustonLab LibPrep

HoustonLab LibPrep prepares supplier SMILES and CXSMILES catalogues for virtual
screening. It cleans and filters the input, expands selected chemical states,
and generates 3D conformers with nvMolKit on an NVIDIA GPU.

The repository includes a command-line pipeline and a small web app for
uploading files, queueing runs, and downloading results. It is designed for a
single Linux GPU server.

## What it does

- keeps supplier IDs while merging and deduplicating catalogues
- strips salts and applies BRENK, Lipinski, ring, aggregator, PAINS, and custom
  SMARTS filters
- enumerates unspecified atom and E/Z stereochemistry
- optionally enumerates tautomers and protonation states with Dimorphite-DL
- generates conformers in bounded GPU chunks and minimises them with MMFF94s
- writes SDF, SMILES, metadata, diagnostic reports, and a JSON run manifest

Runs are strict by default. If a molecule is still missing conformers after a
retry, the run fails without replacing an earlier completed SDF. Use
`--allow-partial-conformers` only when an incomplete result is acceptable.

## Command-line use

Create the environment:

```bash
conda env create -f environment.yml
conda activate libprep
```

Prepare one or more supplier files:

```bash
python library_pipeline.py \
  --input supplier_a.smi supplier_b.smi \
  --output library.sdf \
  --preset docking
```

The `docking` preset uses pH 7.4, skips tautomer enumeration, and creates one
conformer per molecule. The `enumerate` preset includes tautomers and
protonation states across pH 6.4–8.4.

To run only the 2D preparation stages, add `--skip-conformers`. Run
`python library_pipeline.py --help` for all available settings.

Input can be space- or tab-delimited:

```text
SMILES ID
SMILES<TAB>ID
```

CXSMILES extensions are supported. Common SMILES header rows are skipped.

## Web deployment

The web stack needs Docker Compose, an NVIDIA driver compatible with CUDA 12.6,
and NVIDIA Container Toolkit.

```bash
cp .env.example .env
# Set the domain, secret key, data path, and first admin credentials in .env
docker compose up -d --build
```

The data directory in `.env` must exist and be writable by UID 10001. Caddy
handles HTTPS for a dedicated domain. See [DEPLOYMENT.md](DEPLOYMENT.md) for
server setup, backups, upgrades, and path-based deployment.

## Tests

```bash
python -m pip install -r requirements-web.txt
python -m unittest discover -s tests -v
```

## Main dependencies

[RDKit](https://www.rdkit.org/) ·
[Dimorphite-DL](https://github.com/durrantlab/dimorphite_dl) ·
[nvMolKit](https://github.com/NVIDIA-BioNeMo/nvMolKit)

## License

MIT
