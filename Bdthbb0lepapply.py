import glob
import os

import ROOT

ROOT.TH1.AddDirectory(False)

import HbbBDT0lep as hbb0l          
import processor as base            
from HbbBDT0lep import (           
    BDT_VARS_COMMON_0L, BDT_VARS_3J_EXTRA_0L, HISTO_BDT_VH,
    HISTOS_HBB_0L, SAMPLES_HBB_0L, GROUPS_HBB_0L,
)
from processor import HTauTauDSIDProcessor, HistoSpec  


EVENT_SPLIT_BRANCH = "eventNumber"   

CATEGORIES = {
    "2j": {
        "selection": "signal_region_2j",
        "vars": BDT_VARS_COMMON_0L,
    },
    "3j": {
        "selection": "signal_region_3j",
        "vars": {**BDT_VARS_COMMON_0L, **BDT_VARS_3J_EXTRA_0L},
    },
}


def _find_weight_file(loader_name: str) -> str:
    matches = glob.glob(f"{loader_name}/weights/*_BDT.weights.xml") \
        + glob.glob(f"dataset/{loader_name}/weights/*_BDT.weights.xml") \
        + glob.glob(f"**/*{loader_name}*_BDT.weights.xml", recursive=True)
    if not matches:
        raise FileNotFoundError(
            f"No weights '{loader_name}'. "
        )
    return matches[0]


def _make_reader(loader_name: str, var_names: list[str]) -> "ROOT.TMVA.Experimental.RReader":
    weight_file = _find_weight_file(loader_name)
    reader = ROOT.TMVA.Experimental.RReader(weight_file)
    trained_vars = list(reader.GetVariableNames())
    if trained_vars != var_names:
        raise RuntimeError(
            f"Weird order'{loader_name}':\n"
        )
    return reader


class Hbb0LepBDTProcessor(HTauTauDSIDProcessor):

    def __init__(self, *args, bdt_categories: dict, **kwargs):
        self.bdt_categories = bdt_categories   # {selection_name: {"vars":..., "reader_A":..., "reader_B":...}}
        super().__init__(*args, **kwargs)

    def book(self) -> None:
        print("[Hbb0LepBDTProcessor] Booking  BDT_VH ...")
        var_meta = self._var_meta()
        total_booked = 0

        for entry in self.dsid_entries:
            files = base._collect_files(entry.file_patterns)
            if not files:
                print(f"  [WARN] {entry.label} (DSID {entry.dsid}) : NONE.")
                continue
            if self.debug_mode:
                files = files[:1]

            norm_factor = 1.0
            if not entry.is_data:
                try:
                    xsec = base._read_scalar_from_tree(files[0], entry.tree_name, "xsec")
                    kfac = base._read_scalar_from_tree(files[0], entry.tree_name, "kfac")
                    filteff = base._read_scalar_from_tree(files[0], entry.tree_name, "filteff")
                    sow = base._read_scalar_from_tree(files[0], entry.tree_name, "sum_of_weights")
                    norm_factor = self.luminosity_fb * 1000.0 * xsec * kfac * filteff / sow
                except Exception as exc:
                    print(f"    [WARN] failed for {entry.label}: {exc}")

            weight_expr = base._build_weight_expression(entry, self.luminosity_fb, norm_factor)
            rdf_root = ROOT.RDataFrame(entry.tree_name, files)
            self._rdfs[entry.dsid] = rdf_root
            node_weighted = rdf_root.Define("__weight__", weight_expr)

            for sel in self.selections:
                node_sel = node_weighted.Filter(sel.cut, sel.name)
                node_vars = node_sel

                if sel.name in self.bdt_categories:
                    bdt_cfg = self.bdt_categories[sel.name]
                    var_names = list(bdt_cfg["vars"].keys())
                    for var_name, expr in bdt_cfg["vars"].items():
                        node_vars = node_vars.Define(var_name, expr)

                    computeA = ROOT.TMVA.Experimental.Compute[len(var_names), "float"](bdt_cfg["reader_A"])
                    computeB = ROOT.TMVA.Experimental.Compute[len(var_names), "float"](bdt_cfg["reader_B"])
                    node_vars = node_vars.Define("__bdt_vec_A__", computeA, var_names)
                    node_vars = node_vars.Define("__bdt_vec_B__", computeB, var_names)
                    node_vars = node_vars.Define(
                        "BDT_VH",
                        f"({EVENT_SPLIT_BRANCH} % 2 == 0) ? __bdt_vec_B__[0] : __bdt_vec_A__[0]",
                    )

                for spec_name, meta in var_meta.items():
                    if meta['is_expression']:
                        node_vars = node_vars.Define(meta['column_name'], meta['expr'])

                for spec in self.histo_specs:
                    key = (entry.dsid, sel.name, spec.name)
                    if key in self._result_ptrs:
                        continue
                    if spec.name == "h_bdt_vh" and sel.name not in self.bdt_categories:
                        continue  

                    col = var_meta[spec.name]['column_name']
                    model = ROOT.RDF.TH1DModel(
                        f"{spec.name}__{entry.dsid}__{sel.name}",
                        f"{spec.title};{spec.title};Events",
                        spec.n_bins, spec.x_low, spec.x_high,
                    )
                    ptr = node_vars.Histo1D(model, col, "__weight__")
                    self._result_ptrs[key] = ptr
                    total_booked += 1

        print(f"[Hbb0LepBDTProcessor] {total_booked} .")


def main() -> None:
    ROOT.TMVA.Tools.Instance()

    bdt_categories = {}
    for category, cfg in CATEGORIES.items():
        var_names = list(cfg["vars"].keys())
        reader_A = _make_reader(f"BDT_VH_0lep_{category}_A", var_names)
        reader_B = _make_reader(f"BDT_VH_0lep_{category}_B", var_names)
        bdt_categories[cfg["selection"]] = {
            "vars": cfg["vars"],
            "reader_A": reader_A,
            "reader_B": reader_B,
        }

    selections = [s for s in hbb0l.SELECTIONS_HBB_0L
                  if s.name in ("signal_region_2j", "signal_region_3j")]
    histo_specs = HISTOS_HBB_0L + [HISTO_BDT_VH]

    proc = Hbb0LepBDTProcessor(
        dsid_entries=SAMPLES_HBB_0L,
        selections=selections,
        histo_specs=histo_specs,
        luminosity_fb=36.1,
        output_file="hbb_0lep_bdt_histograms.root",
        debug_mode=False,
        bdt_categories=bdt_categories,
    )

    proc.book()
    proc.run()
    proc.save()

    hbb0l.save_grouped(
        proc,
        groups=GROUPS_HBB_0L,
        data_dsid=hbb0l.DATA_DSID,
        output_file="hbb_0lep_bdt_grouped.root",
    )
    print("\n[Hbb0LepBDT] Fichier final : hbb_0lep_bdt_grouped.root ")

if __name__ == "__main__":
    main()
