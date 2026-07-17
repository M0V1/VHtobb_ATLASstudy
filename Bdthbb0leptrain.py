import os

import numpy as np
import ROOT

ROOT.TH1.AddDirectory(False)
ROOT.TMVA.Tools.Instance()

import HbbBDT0lep as hbb0l 
import processor as base            


EVENT_SPLIT_BRANCH = "eventNumber"  

CATEGORIES = {
    "2j": {
        "selection": "signal_region_2j",
        "vars": hbb0l.BDT_VARS_COMMON_0L,
    },
    "3j": {
        "selection": "signal_region_3j",
        "vars": {**hbb0l.BDT_VARS_COMMON_0L, **hbb0l.BDT_VARS_3J_EXTRA_0L},
    },
}

SIGNAL_LABEL = "signal"
BACKGROUND_LABELS = [lbl for lbl in hbb0l.GROUPS_HBB_0L if lbl != SIGNAL_LABEL]
# Same as in the note
BDT_HYPERPARAMS = (
    "!H:!V"
    ":NTrees=1000"
    ":MinNodeSize=5%"
    ":MaxDepth=4"
    ":BoostType=AdaBoost"
    ":AdaBoostBeta=0.2"
    ":SeparationType=GiniIndex"
    ":nCuts=100"
    ":PruneMethod=NoPruning"
)

TMP_DIR = "bdt_train_trees_0lep"
os.makedirs(TMP_DIR, exist_ok=True)


def _weight_expression_for_entry(entry: base.DSIDEntry, luminosity_fb: float) -> str:
    if entry.is_data:
        return "1.0f"
    files = base._collect_files(entry.file_patterns)
    xsec = base._read_scalar_from_tree(files[0], entry.tree_name, "xsec")
    kfac = base._read_scalar_from_tree(files[0], entry.tree_name, "kfac")
    filteff = base._read_scalar_from_tree(files[0], entry.tree_name, "filteff")
    sow = base._read_scalar_from_tree(files[0], entry.tree_name, "sum_of_weights")
    norm_factor = luminosity_fb * 1000.0 * xsec * kfac * filteff / sow
    return base._build_weight_expression(entry, luminosity_fb, norm_factor)


CLIP_PERCENTILE = 99.0   # clipping


def _snapshot_entry(entry: base.DSIDEntry, sel_cut: str, varmap: dict,
                     luminosity_fb: float, out_path: str,
                     clip_percentile: float | None = CLIP_PERCENTILE) -> str | None:
    files = base._collect_files(entry.file_patterns)
    if not files:
        print(f"  [WARN] {entry.label} (DSID {entry.dsid}) : none, ignored")
        return None

    weight_expr = _weight_expression_for_entry(entry, luminosity_fb)
    rdf = ROOT.RDataFrame(entry.tree_name, files)
    node = rdf.Define("__weight_raw__", weight_expr).Filter(sel_cut)

    n_events = node.Count().GetValue()
    if n_events == 0:
        print(f"  [WARN] {entry.label} (DSID {entry.dsid}) : 0 events after "
              f"selection ignored")
        return None

    if clip_percentile is not None:
        w_arr = node.AsNumpy(["__weight_raw__"])["__weight_raw__"]
        cap = float(np.percentile(np.abs(w_arr), clip_percentile))
        if cap <= 0.0:
            cap = float(np.max(np.abs(w_arr))) or 1.0
        n_clipped = int(np.sum(np.abs(w_arr) > cap))
        print(f"  {entry.label} (DSID {entry.dsid}) : {n_events} evts, "
              f"p{clip_percentile:.0f}={cap:.4g}, "
              f"{n_clipped} evt(s) clipped ({100.0 * n_clipped / n_events:.2f}%)")
        node = node.Define(
            "__weight__",
            f"(double)__weight_raw__ > 0.0 ? std::min((double){cap}, (double)__weight_raw__)"
            f" : std::max(-(double){cap}, (double)__weight_raw__)",
        )
    else:
        node = node.Define("__weight__", "__weight_raw__")

    branch_names = []
    for var_name, expr in varmap.items():
        node = node.Define(var_name, expr)
        branch_names.append(var_name)
    branch_names += ["__weight__", EVENT_SPLIT_BRANCH]

    node.Snapshot("train", out_path, branch_names)
    return out_path


def build_chain(entries: list, sel_cut: str, varmap: dict, luminosity_fb: float,
                 category: str, group_label: str) -> ROOT.TChain:
    chain = ROOT.TChain("train")
    for entry in entries:
        out_path = os.path.join(TMP_DIR, f"{category}_{group_label}_{entry.dsid}.root")
        result = _snapshot_entry(entry, sel_cut, varmap, luminosity_fb, out_path)
        if result is not None:
            chain.Add(result)
    return chain


def train_category(category: str, cfg: dict, luminosity_fb: float = 36.1) -> None:
    print(f"\n{'=' * 70}\nTraining BDT_VH — category{category}\n{'=' * 70}")
    sel_cut = next(s.cut for s in hbb0l.SELECTIONS_HBB_0L if s.name == cfg["selection"])
    varmap = cfg["vars"]
    var_names = list(varmap.keys())

    sig_entries = [e for e in hbb0l.SAMPLES_HBB_0L
                   if not e.is_data and e.label == SIGNAL_LABEL]
    bkg_entries = [e for e in hbb0l.SAMPLES_HBB_0L
                   if not e.is_data and e.label in BACKGROUND_LABELS]

    print(f"  Signal DSIDs     : {[e.dsid for e in sig_entries]}")
    print(f"  Background DSIDs : {[e.dsid for e in bkg_entries]}  "
          f"(labels: {sorted(set(e.label for e in bkg_entries))})")
    print(f"  Variables ({len(var_names)}) : {var_names}")

    chain_sig = build_chain(sig_entries, sel_cut, varmap, luminosity_fb, category, "sig")
    chain_bkg = build_chain(bkg_entries, sel_cut, varmap, luminosity_fb, category, "bkg")

    n_sig_tot, n_bkg_tot = chain_sig.GetEntries(), chain_bkg.GetEntries()
    print(f"\n  [diagnostic] Total signal   ({category}) : {n_sig_tot} events MC")
    print(f"  [diagnostic] Total fond     ({category}) : {n_bkg_tot} events MC")

    for split_name, split_cut in (("A", f"{EVENT_SPLIT_BRANCH} % 2 == 0"),
                                   ("B", f"{EVENT_SPLIT_BRANCH} % 2 == 1")):
        loader_name = f"BDT_VH_0lep_{category}_{split_name}"
        print(f"\n  -- Train {split_name} "
              f"({split_cut}) -> loader '{loader_name}' --")

        out_file = ROOT.TFile(f"TMVA_{loader_name}.root", "RECREATE")
        factory = ROOT.TMVA.Factory(
            loader_name, out_file,
            "!V:!Silent:Color:DrawProgressBar:Transformations=I:AnalysisType=Classification",
        )
        loader = ROOT.TMVA.DataLoader(loader_name)

        for var_name in var_names:
            loader.AddVariable(var_name, "F")

        loader.AddSignalTree(chain_sig, 1.0)
        loader.AddBackgroundTree(chain_bkg, 1.0)
        loader.SetSignalWeightExpression("__weight__")
        loader.SetBackgroundWeightExpression("__weight__")

        loader.PrepareTrainingAndTestTree(
            ROOT.TCut(split_cut), ROOT.TCut(split_cut),
            "SplitMode=Random:NormMode=EqualNumEvents:!V",
        )

        factory.BookMethod(loader, ROOT.TMVA.Types.kBDT, "BDT", BDT_HYPERPARAMS)
        factory.TrainAllMethods()
        factory.TestAllMethods()
        factory.EvaluateAllMethods()

        try:
            roc_auc = factory.GetROCIntegral(loader, "BDT")
            print(f"  [diagnostic] ROC AUC (test, {split_name}) = {roc_auc:.4f}")
        except Exception as exc:
            print(f"  [diagnostic] No ROC AUC ({exc}) ")

        out_file.Close()

        print(f"  -> Weights dataset/{loader_name}/weights/"
              f"{loader_name}_BDT.weights.xml")


if __name__ == "__main__":
    for category, cfg in CATEGORIES.items():
        train_category(category, cfg)

