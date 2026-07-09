"""DreaMS-Mol v3 cross-encoder as a FOAM oracle (fitness = v3 (spectrum, candidate) compatibility logit).

In-loop replacement for ICEBERG in the FOAM genetic algorithm, for the "FOAM + DreaMS-Mol v3" experiment.
Unlike ICEBERG (a forward model that predicts a spectrum from a molecule), v3 is a JOINT cross-encoder
that needs per-peak MetFrag fragment annotations for each (spectrum, candidate) pair. Those are produced
ONLINE (dreams_mol.fragmentation.online) with a SMILES-keyed fragment-tree cache so seeds and
rediscovered offspring are fragmented once, not once per (spectrum, generation).

Design choices that keep this importable in the ICEBERG-only (ms-gen / torch-1.9) env and avoid the
ms_pred torch-version conflict in the v3 (modern-torch) env:
  - heavy v3/torch imports are LAZY (inside _get_v3_assets / score paths), so merely importing
    foam.oracles (which lazily imports THIS module only for DreaMSMol_* oracle types) never triggers them;
  - the __init__ is `common`-free (rdkit-only mol setup), so the v3 env needs no ms_pred;
  - the query spectrum is read from MassSpecGym1.5.tsv by spec_id (the representation v3 was trained on),
    NOT FOAM's per-collision-energy .ms spec library.

v3 model config comes from env vars (Karolina deploy overrides the LUMI-release defaults):
  DREAMSMOL_V3_CKPT, DREAMSMOL_V3_CONFIG_MAIN, DREAMSMOL_V3_CONFIG_MODEL, DREAMSMOL_V3_TSV,
  DREAMSMOL_V3_DEVICE (cuda|cpu), DREAMSMOL_V3_CAP (per-peak cap, default 100), DREAMSMOL_V3_FLOOR.

Registered by oracles.build_oracle under --oracle-type DreaMSMol_ (v3 only) and DreaMSMol_SA_ (v3 + the
same SA 2nd objective as Cos_SA_ — the apples-to-apples analogue of FOAM's production ICEBERG oracle).
"""
import os
from functools import partial

import numpy as np
from rdkit import Chem

from foam.oracles import MolOracle, morgan_fp

_DEF = {
    "ckpt": "/scratch/project_465003029/models/DreaMS-Mol-v3/checkpoints/dreams_mol_v3_combo_tani80_step13776.ckpt",
    "config_main": "/scratch/project_465003029/models/DreaMS-Mol-v3/configs/config_main.yaml",
    "config_model": "/scratch/project_465003029/models/DreaMS-Mol-v3/configs/config_dreams_mol_v3_combo_tani80.yaml",
    "tsv": "/scratch/project_465003029/data/MassSpecGym/v1.5/MassSpecGym1.5.tsv",
}

_V3_ASSETS = {}  # ckpt path -> shared assets (model + mol_transform + tree cache + spectrum lookup)


def _env(name, default):
    return os.environ.get(name, default)


def _get_v3_assets():
    """Load (once per process) the v3 model, graph transform, fragment-tree cache, and spectrum lookup.

    Shared across every oracle (spectrum) in the process so the model loads once and the fragment cache
    reuses trees across spectra of the same formula. Heavy imports happen here (lazily)."""
    ckpt = _env("DREAMSMOL_V3_CKPT", _DEF["ckpt"])
    a = _V3_ASSETS.get(ckpt)
    if a is not None:
        return a

    from pathlib import Path

    import pandas as pd
    import torch
    from omegaconf import OmegaConf

    from dreams_mol.config import parse_configs
    from dreams_mol.data.mol_transforms import MolToPyG
    from dreams_mol.models.fragment_cross_encoder import FragmentTokenCrossEncoder
    from dreams_mol.fragmentation.online import BoundedTreeCache

    cfg = parse_configs(Path(_env("DREAMSMOL_V3_CONFIG_MAIN", _DEF["config_main"])),
                        Path(_env("DREAMSMOL_V3_CONFIG_MODEL", _DEF["config_model"])))
    OmegaConf.set_struct(cfg, False)
    ce = cfg.cross_encoder

    want = _env("DREAMSMOL_V3_DEVICE", "cuda")
    device = "cuda" if (want == "cuda" and torch.cuda.is_available()) else "cpu"
    model = FragmentTokenCrossEncoder.load_from_checkpoint(ckpt, map_location=device).eval().to(device)

    mol_transform = MolToPyG(
        pe_laplacian_eigenvectors=0, pe_rand_walks=0, atom_fp_size=None,
        atom_features=ce.gnn.get("atom_features", None), return_dummy_graphs_for_invalid_mols=True)
    n_peaks = ce.get("n_peaks", 100)
    tree_cache = BoundedTreeCache(maxsize=int(_env("DREAMSMOL_V3_TREE_CACHE", "200000")))

    tsv = pd.read_csv(_env("DREAMSMOL_V3_TSV", _DEF["tsv"]), sep="\t", low_memory=False,
                      usecols=["identifier", "mzs", "intensities", "precursor_mz", "instrument_type"])
    tsv["identifier"] = tsv["identifier"].astype(str)
    tsv = tsv.set_index("identifier")

    def get_spec(spec_id):
        r = tsv.loc[str(spec_id)]
        if getattr(r, "ndim", 1) > 1:
            r = r.iloc[0]
        mzs = np.fromstring(str(r["mzs"]), sep=",", dtype=np.float64)
        ins = np.fromstring(str(r["intensities"]), sep=",", dtype=np.float64)
        n = min(len(mzs), len(ins))
        return mzs[:n], ins[:n], float(r["precursor_mz"]), str(r["instrument_type"])

    a = dict(model=model, mol_transform=mol_transform, n_peaks=n_peaks,
             tree_cache=tree_cache, device=device, get_spec=get_spec)
    _V3_ASSETS[ckpt] = a
    return a


class _DreaMSMolOracle(MolOracle):
    """Single-objective v3 oracle: scores[0] = v3 compatibility logit."""

    def __init__(self, smiles, name, **kwargs):
        self.name = name
        mol = None
        if smiles is not None:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                Chem.RemoveStereochemistry(mol)
        # mol=None -> skip MolOracle's common.chem_utils canonicalization; we set mol attrs (rdkit only)
        super().__init__(mol=None, **kwargs)
        self.multiobj = bool(kwargs.get("multiobj", False))
        self.criteria = kwargs.get("criteria", "v3")
        self.colli_engs = []                       # v3 has no collision-energy split (contract attr)
        self._cap = int(_env("DREAMSMOL_V3_CAP", str(kwargs.get("v3_cap_per_peak", 100))))
        self._floor = float(_env("DREAMSMOL_V3_FLOOR", str(kwargs.get("v3_floor", -50.0))))
        if mol is not None:
            self.mol = mol
            self.mol_smiles = Chem.MolToSmiles(mol)
            self.mol_inchikey = Chem.MolToInchiKey(mol)
            self.nbits = 2048
            self.get_morgan_fp = partial(morgan_fp, nbits=2048)
            self.morgan_fp = self.get_morgan_fp(mol)
        self.max_seed_sim = kwargs.get("max_seed_sim", None)

        a = _get_v3_assets()
        self._model = a["model"]
        self._mol_transform = a["mol_transform"]
        self._n_peaks = a["n_peaks"]
        self._tree_cache = a["tree_cache"]
        self._device = a["device"]
        self._mzs, self._ins, self._prec_mz, self._inst = a["get_spec"](name)

    def _score_v3(self, smiles):
        """v3 logit per candidate SMILES (aligned to `smiles`). Unmatched candidates get null tokens
        (via full_candidates); an all-unmatched batch (no peak explained by any candidate) -> floor."""
        import torch

        from dreams_mol.data.fragment_datasets import FragmentCrossEncoderDataset
        from dreams_mol.fragmentation.online import label_index

        index, _ = label_index(self.name, self._mzs, self._ins, self._prec_mz, self._inst,
                               smiles, self._tree_cache, cap_per_peak=self._cap)
        if index is None:
            return np.full(len(smiles), self._floor, dtype=np.float64)

        ds = FragmentCrossEncoderDataset.from_memory(
            index, {self.name: (self._mzs, self._ins, self._prec_mz)}, self._mol_transform,
            n_peaks=self._n_peaks, with_frag_graph=True, full_cands={self.name: list(smiles)})
        batch = ds.collate_fn([ds[0]])
        for k in ("spec", "peak_scores", "has_frag", "frag_scores", "batch_ptr"):
            if k in batch and torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(self._device)
        batch["cand_pyg"] = batch["cand_pyg"].to(self._device)
        with torch.no_grad():
            logit = self._model.score_candidates(batch).float().cpu().numpy().astype(np.float64)

        order = batch["candidates_smiles"]
        if list(order) == list(smiles):
            return logit
        pos = {}
        for i, s in enumerate(order):
            pos.setdefault(s, i)
        return np.array([logit[pos[s]] if s in pos else self._floor for s in smiles], dtype=np.float64)

    def score_valid_mols(self, examples, **kwargs):
        smiles = [Chem.MolToSmiles(m, isomericSmiles=False) for m in examples]
        return self._score_v3(smiles)

    def score_batch(self, examples, **kwargs):
        inds = [i for i, e in enumerate(examples) if e is not None]
        if not inds:
            return np.zeros((1, len(examples)))
        data = self.score_valid_mols([examples[i] for i in inds])
        if isinstance(data, np.ndarray) and data.ndim == 1:
            data = (data,)
        data = tuple(o for o in data if o is not None)
        output = np.zeros((max(1, len(data)), len(examples)))
        for i, obj in enumerate(data):
            output[i, inds] = obj
        return output

    def __call__(self, examples):
        return self.score_batch(examples)

    @staticmethod
    def oracle_name():
        return "DreaMSMolOracle"


class _DreaMSMolWithSAOracle(_DreaMSMolOracle):
    """v3 + synthetic-accessibility, mirroring _ICEBERGWithSAOracle so the only change vs FOAM's
    production Cos_SA_ oracle is the primary scorer (v3 instead of ICEBERG). multiobj=True gives the GA
    two Pareto objectives (v3 logit, SA); scale mismatch is irrelevant under non-dominated sorting."""

    def __init__(self, smiles, name, iceberg_param=0.8, sa_param=0.2, multiobj=False, **kwargs):
        self.iceberg_param = iceberg_param
        self.sa_param = sa_param
        super().__init__(smiles, name, multiobj=multiobj, **kwargs)

    def score_valid_mols(self, examples, **kwargs):
        primary = super().score_valid_mols(examples, **kwargs)
        sa_scores = 1 - self.score_sa(examples) / 10
        if self.multiobj:
            return primary, sa_scores
        return self.iceberg_param * np.asarray(primary) + self.sa_param * np.asarray(sa_scores), None

    @staticmethod
    def oracle_name():
        return "DreaMSMolWithSAOracle"
