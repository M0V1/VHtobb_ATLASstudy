import multiprocessing
import re
from dataclasses import dataclass, field
from pathlib import Path

import ROOT



@dataclass
class DSIDEntry:
    dsid: int
    label: str                     
    file_patterns: list[str]
    tree_name: str    = "analysis"
    is_data: bool     = False
    extra_weight: str = "1.0"     


@dataclass
class SelectionSpec:
    name: str
    cut: str


@dataclass
class HistoSpec:
    name: str          
    branch: str        
    title: str       
    n_bins: int
    x_low: float
    x_high: float


def _collect_files(patterns: list[str]) -> list[str]:
    files = []
    for pat in patterns:
        if '*' in pat or '?' in pat:
            matched = sorted(Path().glob(pat))
        else:
            matched = [Path(pat)]
        files.extend(str(p) for p in matched if p.exists())
    return files


def _is_expression(expr: str) -> bool:
    return bool(re.search(r'[.()\[\]+\-*/><!=&| ]', expr))


def _sanitize(expr: str) -> str:
    name = expr.replace("-", "m")
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    if not name[0].isalpha():
        name = '_' + name
    return name


def _read_scalar_from_tree(filepath: str, tree_name: str, branch: str) -> float:
    f = ROOT.TFile.Open(filepath)
    if not f or f.IsZombie():
        raise RuntimeError(f"Wrong {filepath}")
    tree = f.Get(tree_name)
    if not tree:
        raise RuntimeError(f"TTree '{tree_name}' not here {filepath}")
    tree.GetEntry(0)
    val = float(getattr(tree, branch))
    f.Close()
    return val


_SF_EXPRESSION = ( "ScaleFactor_ELE" " * ScaleFactor_MUON" " * ScaleFactor_TAU" " * ScaleFactor_LepTRIGGER" " * 
    ScaleFactor_PILEUP" " * ScaleFactor_JVT" " * mcWeight"
)


def _build_weight_expression(
    entry: DSIDEntry,
    luminosity_fb: float,
    norm_factor: float,           # lumi × 1000 × xsec × kfac × filteff / sumofweight
) -> str: 
    """ 
    norm_factor × SF_expression × extra_weight
    """
    if entry.is_data:
        return "1.0f"
    return (
        f"(float)({norm_factor})"
        f" * ({_SF_EXPRESSION})"
        f" * ({entry.extra_weight})"
    )


# Processor principal

class HTauTauDSIDProcessor:
    """
    Parameters
    ----------
    dsid_entries  : list  DSIDEntry
    selections    : list  SelectionSpec
    histo_specs   : list  HistoSpec
    luminosity_fb : luminosity [fb⁻¹]
    output_file   : path
    n_cores       : threads MT (None = auto, max 10)
    debug_mode    :
    """

    def __init__(
        self,
        dsid_entries: list[DSIDEntry],
        selections: list[SelectionSpec],
        histo_specs: list[HistoSpec],
        luminosity_fb: float = 36.1,
        output_file: str = "output_htautau.root",
        n_cores: int | None = None,
        debug_mode: bool = False,
    ) -> None:
        self.dsid_entries   = dsid_entries
        self.selections     = selections
        self.histo_specs    = histo_specs
        self.luminosity_fb  = luminosity_fb
        self.output_file    = output_file
        self.debug_mode     = debug_mode

        self._rdfs: dict[int, ROOT.RDataFrame] = {}
        self._result_ptrs: dict[tuple, object] = {}   # (dsid, sel, histo) → ptr

        #  multithreading
        if n_cores is None:
            n_cores = max(1, multiprocessing.cpu_count() - 1)
        n_cores = min(n_cores, 10)
        print(f"[HTauTauProcessor] MT : {n_cores} threads")
        ROOT.EnableImplicitMT(n_cores)
        ROOT.EnableThreadSafety()

        self._print_summary()


    def _print_summary(self) -> None:
        print("\n" + "=" * 65)
        print(f"  Samples      : {[e.label for e in self.dsid_entries]}")
        print(f"  Selections   : {[s.name  for s in self.selections]}")
        print(f"  Histograms   : {[h.name  for h in self.histo_specs]}")
        print(f"  Luminosity   : {self.luminosity_fb} fb⁻¹")
        print(f"  Output       : {self.output_file}")
        print("=" * 65 + "\n")

    def _var_meta(self) -> dict:
        meta = {}
        for spec in self.histo_specs:
            is_expr = _is_expression(spec.branch)
            meta[spec.name] = {
                'is_expression': is_expr,
                'column_name'  : _sanitize(spec.branch) if is_expr else spec.branch,
                'expr'         : spec.branch,
            }
        return meta

    #  Booking 

    def book(self) -> None:
        print("[HTauTauProcessor] Booking  histograms")
        var_meta = self._var_meta()
        total_booked = 0

        for entry in self.dsid_entries:
            files = _collect_files(entry.file_patterns)
            if not files:
                print(f"  [WARN] {entry.label} (DSID {entry.dsid}) : "
                      "NONE, IGNORED")
                continue
            if self.debug_mode:
                files = files[:1]
                print(f"  [DEBUG] {entry.label} :  debug  1 file")

            print(f"  {entry.label} (DSID {entry.dsid}) : "
                  f"{len(files)} file, tree '{entry.tree_name}'")

            norm_factor = 1.0
            if not entry.is_data:
                try:
                    xsec    = _read_scalar_from_tree(files[0], entry.tree_name, "xsec")
                    kfac    = _read_scalar_from_tree(files[0], entry.tree_name, "kfac")
                    filteff = _read_scalar_from_tree(files[0], entry.tree_name, "filteff")
                    sow     = _read_scalar_from_tree(files[0], entry.tree_name, "sum_of_weights")
                    norm_factor = (
                        self.luminosity_fb * 1000.0   # fb⁻¹ → pb⁻¹
                        * xsec * kfac * filteff / sow
                    )
                    print(f"    xsec={xsec:.4g} pb  kfac={kfac:.4g}  "
                          f"filteff={filteff:.4g}  ΣW={sow:.4g}  "
                          f"→ norm={norm_factor:.4g}")
                except Exception as exc:
                    print(f"    [WARN] failed : {exc}")
                    print("    normalise to 1.0")

            weight_expr = _build_weight_expression(entry, self.luminosity_fb, norm_factor)

            rdf_root = ROOT.RDataFrame(entry.tree_name, files)
            self._rdfs[entry.dsid] = rdf_root

            node_weighted = rdf_root.Define("__weight__", weight_expr)

            for sel in self.selections:
                node_sel = node_weighted.Filter(sel.cut, sel.name)

                node_vars = node_sel
                for spec_name, meta in var_meta.items():
                    if meta['is_expression']:
                        node_vars = node_vars.Define(
                            meta['column_name'],
                            meta['expr']
                        )

                for spec in self.histo_specs:
                    key = (entry.dsid, sel.name, spec.name)
                    if key in self._result_ptrs:
                        continue

                    col = var_meta[spec.name]['column_name']
                    model = ROOT.RDF.TH1DModel(
                        f"{spec.name}__{entry.dsid}__{sel.name}",
                        f"{spec.title};{spec.title};Events",
                        spec.n_bins,
                        spec.x_low,
                        spec.x_high,
                    )
                    ptr = node_vars.Histo1D(model, col, "__weight__")
                    self._result_ptrs[key] = ptr
                    total_booked += 1

        print(f"\n[HTauTauProcessor] {total_booked} all booked\n")


    def run(self) -> None:
        """ROOT::RDF::RunGraphs (lazy execution)"""
        if not self._result_ptrs:
            raise RuntimeError("Nothing to execute, run book()  before")

        for rdf in self._rdfs.values():
            ROOT.RDF.Experimental.AddProgressBar(rdf)

        print("[HTauTauProcessor]…")
        sw = ROOT.TStopwatch()
        ROOT.RDF.RunGraphs(list(self._result_ptrs.values()))
        sw.Stop()
        print("[HTauTauProcessor] End")
        sw.Print()


    def save(self) -> None:
        """
        Structure :  <label>/<selection>/<histo_name>
        """
        dsid_to_label = {e.dsid: e.label for e in self.dsid_entries}

        print(f"[HTauTauProcessor] Saved → '{self.output_file}' …")
        out = ROOT.TFile(self.output_file, "RECREATE")

        for (dsid, sel_name, spec_name), ptr in self._result_ptrs.items():
            label    = dsid_to_label.get(dsid, str(dsid))
            dir_path = f"{label}/{sel_name}"

            tdir = out.GetDirectory(dir_path)
            if not tdir:
                tdir = out.mkdir(dir_path)
            tdir.cd()

            h = ptr.GetPtr().Clone(spec_name)
            h.SetDirectory(tdir)
            h.Write()

        out.Close()
        print(f"[HTauTauProcessor] Write : {self.output_file}")

    def get_histogram(
        self, dsid: int, selection_name: str, histo_name: str
    ) -> ROOT.TH1:
        key = (dsid, selection_name, histo_name)
        ptr = self._result_ptrs.get(key)
        if ptr is None:
            raise KeyError(f"No hist {key}")
        h = ptr.GetPtr().Clone(f"{histo_name}_clone")
        h.SetDirectory(0)
        return h

    def sum_over_dsids(
        self,
        dsids: list[int],
        selection_name: str,
        histo_name: str,
    ) -> ROOT.TH1 | None:
        total = None
        for dsid in dsids:
            try:
                h = self.get_histogram(dsid, selection_name, histo_name)
            except KeyError:
                continue
            total = h if total is None else (total.Add(h), total)[1]
        return total


# Selections
_TRIGGER       = "(trigE || trigM)"
_ONE_SIGNAL_LEP = "n_sig_lep == 1"
_ONE_GOOD_TAU   = (
    "tau_n >= 1"
    " && tau_isTight[0] == true"
    " && tau_pt[0] > 30"         
    " && abs(tau_eta[0]) < 2.5"
)
_OPP_CHARGE    = "lep_charge[0] * tau_charge[0] < 0"

_DEFINE_MVIS = (
    "float(sqrt(2 * lep_pt[0] * tau_pt[0]"
    " * (cosh(lep_eta[0] - tau_eta[0])"
    "  - cos(lep_phi[0]  - tau_phi[0]))))"
)
_DEFINE_MT = (
    "float(sqrt(2 * lep_pt[0] * met"
    " * (1 - cos(lep_phi[0] - met_phi))))"
)
_DEFINE_DR    = (
    "float(sqrt(pow(lep_eta[0]-tau_eta[0],2)"
    " + pow(lep_phi[0]-tau_phi[0],2)))"
)
_DEFINE_DETA  = "float(abs(lep_eta[0] - tau_eta[0]))"

_KINEMATIC_CUTS = (
    "met > 32"       
 #   " && tau_pt>25.f"
   
  
)

SELECTIONS_HTAUTAU = [
    SelectionSpec(
        name = "presel",
        cut  = f"{_TRIGGER} && {_ONE_SIGNAL_LEP} && {_ONE_GOOD_TAU} && {_OPP_CHARGE}",
    ),
    SelectionSpec(
        name = "signal_region",
        cut  = (
            f"{_TRIGGER} && {_ONE_SIGNAL_LEP} && {_ONE_GOOD_TAU}"
            f" && {_OPP_CHARGE} && {_KINEMATIC_CUTS}"
        ),
    ),
]

# VARIABLES
HISTOS_HTAUTAU = [
    # MET
    HistoSpec("h_met",      "met",         "E_{T}^{miss} [MeV]",   50,      0, 300),
    # Lepton
    HistoSpec("h_lep_pt",   "lep_pt[0]",   "Lepton p_{T} [MeV]",   50,  20, 250),
    HistoSpec("h_lep_eta",  "lep_eta[0]",  "Lepton #eta",           50,    -2.5,     2.5),
    HistoSpec("h_lep_phi",  "lep_phi[0]",  "Lepton #phi",           32,  -3.14,    3.14),
    # Tau hadronique
    HistoSpec("h_tau_pt",   "tau_pt[0]",   "#tau_{had} p_{T} [MeV]",50, 30, 200),
    HistoSpec("h_tau_eta",  "tau_eta[0]",  "#tau_{had} #eta",        50,   -2.5,     2.5),
    HistoSpec("h_tau_nTracks","tau_nTracks[0]","#tau_{had} n tracks",  5,      0,       5),
    # Jets
    HistoSpec("h_jet_n",    "jet_n",        "N_{jets}",              10,      0,      10),
    HistoSpec("h_jet_pt",   "jet_pt[0]",    "Leading jet p_{T} [MeV]",50, 40, 400),
    # Variables composées — définies via C++ expression
    HistoSpec("h_mvis",
              _DEFINE_MVIS,
              "m_{vis}(l,#tau) [MeV]",      50,  35, 180),
    HistoSpec("h_mt",
              _DEFINE_MT,
              "m_{T}(l,MET) [MeV]",         50,       0,  70),
    HistoSpec("h_dr",
              _DEFINE_DR,
              "#DeltaR(l,#tau)",             50,       0,     2.5),
    HistoSpec("h_deta",
              _DEFINE_DETA,
              "|#Delta#eta(l,#tau)|",        50,       0,     1.5),
]



if __name__ == "__main__":

#    BASE = "/eos/opendata/atlas/rucio/opendata/"
    BASE = "/afs/cern.ch/user/m/movincen/eos/2026SuMIFIC/"

    SKIM = "2bjets70"
    PFX  = f"{BASE}ODEO_FEB2025_v0_{SKIM}_"

    samples = [
        DSIDEntry(
            dsid=0, label="data",
            file_patterns=[
                f"{PFX}data15_periodD.{SKIM}.root",
                f"{PFX}data15_periodE.{SKIM}.root",
                f"{PFX}data15_periodF.{SKIM}.root",
                f"{PFX}data15_periodG.{SKIM}.root",
                f"{PFX}data15_periodH.{SKIM}.root",
                f"{PFX}data15_periodJ.{SKIM}.root",
             
        
              
            ],
            is_data=True,
        ),

        DSIDEntry(
            dsid=410470, label="ttbar_nonallhad",
            file_patterns=[f"{PFX}mc_410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.{SKIM}.root"],
        ),

        DSIDEntry(
            dsid=700792, label="Ztautau_BFilter",
            file_patterns=[f"{PFX}mc_700792.Sh_2214_Ztautau_maxHTpTV2_BFilter.{SKIM}.root"],
        ),
        DSIDEntry(
            dsid=700793, label="Ztautau_CFilterBVeto",
            file_patterns=[f"{PFX}mc_700793.Sh_2214_Ztautau_maxHTpTV2_CFilterBVeto.{SKIM}.root"],
        ),
        DSIDEntry(
            dsid=700794, label="Ztautau_CVetoBVeto",
            file_patterns=[f"{PFX}mc_700794.Sh_2214_Ztautau_maxHTpTV2_CVetoBVeto.{SKIM}.root"],
        ),

        DSIDEntry(
            dsid=345120, label="ggH125_tautaul13l7",
            file_patterns=[f"{PFX}mc_345120.PowhegPy8EG_NNLOPS_nnlo_30_ggH125_tautaul13l7.{SKIM}.root"],
        ),

        DSIDEntry(
            dsid=700338, label="Wenu_BFilter",
            file_patterns=[f"{PFX}mc_700338.Sh_2211_Wenu_maxHTpTV2_BFilter.{SKIM}.root"],
        ),

        # Add what you need
    ]

    proc = HTauTauDSIDProcessor(
        dsid_entries  = samples,
        selections    = SELECTIONS_HTAUTAU,
        histo_specs   = HISTOS_HTAUTAU,
        luminosity_fb = 36.1,                    # data15+16
        output_file   = "htautau_histograms.root",
        debug_mode    = False,                   # True → 1 DSID (test debug)
    )

    proc.book()
    proc.run()
    proc.save()

