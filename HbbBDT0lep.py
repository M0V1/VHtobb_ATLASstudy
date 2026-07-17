"""
HbbDSIDProcessor — analysis H -> b bbar, canal 0-lepton (Z -> nunu, H -> bb)
"""

import ROOT

ROOT.TH1.AddDirectory(False)

import processor as base
from processor import DSIDEntry, SelectionSpec, HistoSpec, HTauTauDSIDProcessor



base._SF_EXPRESSION = (
    "ScaleFactor_PILEUP"
    " * ScaleFactor_JVT"
    " * ScaleFactor_FTAG"
    " * mcWeight"
)



_HBB_CPP = r"""
#ifndef HBB_HELPERS_0L_DEFINED
#define HBB_HELPERS_0L_DEFINED

#include "ROOT/RVec.hxx"
#include "TLorentzVector.h"
#include <cmath>
#include <algorithm>

using namespace ROOT::VecOps;

// ── Jets ────────────────────────────────────────────────────────────────

inline RVec<bool> Hbb_cjetMask(const RVec<float>& pt, const RVec<float>& eta,
                                const RVec<int>& jvt) {
    RVec<bool> mask(pt.size());
    for (size_t i = 0; i < pt.size(); ++i)
        mask[i] = (jvt[i] == 1) && (pt[i] > 20.0f) && (std::abs(eta[i]) < 2.5f);
    return mask;
}

inline RVec<bool> Hbb_fjetMask(const RVec<float>& pt, const RVec<float>& eta,
                                const RVec<int>& jvt) {
    RVec<bool> mask(pt.size());
    for (size_t i = 0; i < pt.size(); ++i)
        mask[i] = (jvt[i] == 1) && (pt[i] > 30.0f)
                  && (std::abs(eta[i]) >= 2.5f) && (std::abs(eta[i]) < 4.5f);
    return mask;
}

inline RVec<bool> Hbb_btagMask(const RVec<int>& btagq) {
    RVec<bool> mask(btagq.size());
    for (size_t i = 0; i < btagq.size(); ++i)
        mask[i] = (btagq[i] >= 4);
    return mask;
}

inline RVec<int> Hbb_bjetIndices(const RVec<float>& jet_pt, const RVec<float>& jet_eta,
                                  const RVec<int>& jet_jvt, const RVec<int>& jet_btagq) {
    auto cjet = Hbb_cjetMask(jet_pt, jet_eta, jet_jvt);
    auto btag = Hbb_btagMask(jet_btagq);
    RVec<int> idx;
    for (size_t i = 0; i < jet_pt.size(); ++i)
        if (cjet[i] && btag[i]) idx.push_back((int)i);
    return idx;
}

inline bool Hbb_passJetSel(const RVec<float>& jet_pt, const RVec<float>& jet_eta,
                            const RVec<int>& jet_jvt, const RVec<int>& jet_btagq) {
    auto cjet = Hbb_cjetMask(jet_pt, jet_eta, jet_jvt);
    auto fjet = Hbb_fjetMask(jet_pt, jet_eta, jet_jvt);
    auto btag = Hbb_btagMask(jet_btagq);

    int njet = 0, nb = 0, nb45 = 0;
    for (size_t i = 0; i < jet_pt.size(); ++i) {
        if (cjet[i] || fjet[i]) njet++;
        if (cjet[i] && btag[i]) {
            nb++;
            if (jet_pt[i] > 45.0f) nb45++;
        }
    }
    bool njetOK = (njet == 2 || njet == 3);
    bool nbOK   = (nb == 2);
    bool nb45OK = (nb45 == 1);
    return njetOK && nbOK && nb45OK;
}

// H_T > 120 GeV (2 jets) ou > 150 GeV (3 jets)
inline bool Hbb_passHT(const RVec<float>& jet_pt, const RVec<float>& jet_eta,
                        const RVec<int>& jet_jvt) {
    auto cjet = Hbb_cjetMask(jet_pt, jet_eta, jet_jvt);
    auto fjet = Hbb_fjetMask(jet_pt, jet_eta, jet_jvt);
    int njet = 0; float ht = 0.0f;
    for (size_t i = 0; i < jet_pt.size(); ++i) {
        if (cjet[i] || fjet[i]) { njet++; ht += jet_pt[i]; }
    }
    if (njet == 2) return ht > 120.0f;
    if (njet > 2)  return ht > 150.0f;
    return false;
}

inline float Hbb_dphi(float a, float b) {
    float d = std::abs(a - b);
    if (d > M_PI) d = 2.0f * (float)M_PI - d;
    return d;
}
inline float Hbb_dphiRaw(float a, float b) {
    return std::abs(a - b);
}

// Delta_phi(b1,b2) < 140 deg
inline bool Hbb_pass2bPhi(const RVec<float>& jet_pt, const RVec<float>& jet_eta,
                           const RVec<float>& jet_phi, const RVec<int>& jet_jvt,
                           const RVec<int>& jet_btagq) {
    auto idx = Hbb_bjetIndices(jet_pt, jet_eta, jet_jvt, jet_btagq);
    if (idx.size() < 2) return false;
  //  return Hbb_dphi(jet_phi[idx[0]], jet_phi[idx[1]]) < (140.0f * (float)M_PI / 180.0f);
    return Hbb_dphiRaw(jet_phi[idx[0]], jet_phi[idx[1]]) < (140.0f * (float)M_PI / 180.0f);
}

// Delta_phi(MET, bb) > 120 deg
inline bool Hbb_passMetBBPhi(const RVec<float>& jet_pt, const RVec<float>& jet_eta,
                              const RVec<float>& jet_phi, const RVec<int>& jet_jvt,
                              const RVec<int>& jet_btagq, float met_phi) {
    auto idx = Hbb_bjetIndices(jet_pt, jet_eta, jet_jvt, jet_btagq);
    if (idx.size() < 2) return false;
    float px = jet_pt[idx[0]] * std::cos(jet_phi[idx[0]]) + jet_pt[idx[1]] * std::cos(jet_phi[idx[1]]);
    float py = jet_pt[idx[0]] * std::sin(jet_phi[idx[0]]) + jet_pt[idx[1]] * std::sin(jet_phi[idx[1]]);
    float bb_phi = std::atan2(py, px);
//    return Hbb_dphi(met_phi, bb_phi) > (120.0f * (float)M_PI / 180.0f);
    return Hbb_dphiRaw(met_phi, bb_phi) > (120.0f * (float)M_PI / 180.0f);
}

// min Delta_phi(MET, jets) > 20 deg (2 jets) ou > 30 deg (3 jets)
inline bool Hbb_passMinMetJetsPhi(const RVec<float>& jet_pt, const RVec<float>& jet_eta,
                                   const RVec<float>& jet_phi, const RVec<int>& jet_jvt,
                                   float met_phi) {
    auto cjet = Hbb_cjetMask(jet_pt, jet_eta, jet_jvt);
    auto fjet = Hbb_fjetMask(jet_pt, jet_eta, jet_jvt);
    int njet = 0; float minDphi = 1e9f;
    for (size_t i = 0; i < jet_pt.size(); ++i) {
        if (cjet[i] || fjet[i]) {
            njet++;
            float d = Hbb_dphiRaw(jet_phi[i], met_phi);
            if (d < minDphi) minDphi = d;
        }
    }
    if (njet == 2) return minDphi > (20.0f * (float)M_PI / 180.0f);
    if (njet > 2)  return minDphi > (30.0f * (float)M_PI / 180.0f);
    return false;
}

inline float Hbb_ptRecoCorrE(float pt, float e) {
    if (pt < 100.0f) {
        float corr_pct = (-11.0f / 80.0f) * pt + 14.76f;
        return e * (1.0f + corr_pct / 100.0f);
    }
    return e * 1.01f;
}

inline float Hbb_mbb(const RVec<float>& jet_pt, const RVec<float>& jet_eta,
                      const RVec<float>& jet_phi, const RVec<float>& jet_e,
                      const RVec<int>& jet_jvt, const RVec<int>& jet_btagq) {
    auto idx = Hbb_bjetIndices(jet_pt, jet_eta, jet_jvt, jet_btagq);
    if (idx.size() < 2) return -1.0f;
    int i0 = idx[0], i1 = idx[1];
    float e0 = Hbb_ptRecoCorrE(jet_pt[i0], jet_e[i0]);
    float e1 = Hbb_ptRecoCorrE(jet_pt[i1], jet_e[i1]);
    TLorentzVector v0, v1;
    v0.SetPtEtaPhiE(jet_pt[i0], jet_eta[i0], jet_phi[i0], e0);
    v1.SetPtEtaPhiE(jet_pt[i1], jet_eta[i1], jet_phi[i1], e1);
    return (float)((v0 + v1).M());
}

inline bool Hbb_passZeroLep(const RVec<int>& lep_type, const RVec<float>& lep_pt,
                             const RVec<float>& lep_eta, const RVec<float>& lep_d0sig,
                             const RVec<float>& lep_z0, const RVec<int>& lep_isLooseIso,
                             const RVec<int>& lep_isLooseID) {
    int nel = 0, nmu = 0;
    for (size_t i = 0; i < lep_type.size(); ++i) {
        bool isE   = (lep_type[i] == 11);
        bool isMu  = (lep_type[i] == 13);
        float aeta = std::abs(lep_eta[i]);
        bool ptOK  = lep_pt[i] > 7.0f;
        bool isoOK = (lep_isLooseIso[i] == 1);
        bool idOK  = (lep_isLooseID[i] == 1);
        bool z0OK  = (lep_z0[i] < 0.5f);            
        if (isE) {
            bool etaOK = (aeta < 2.47f);
            bool d0OK  = (lep_d0sig[i] < 5.0f);     
            if (ptOK && etaOK && d0OK && z0OK && isoOK && idOK) nel++;
        } else if (isMu) {
            bool etaOK = (aeta < 2.7f);
            bool d0OK  = (lep_d0sig[i] < 3.0f);     
            if (ptOK && etaOK && d0OK && z0OK && isoOK && idOK) nmu++;
        }
    }
    return (nel == 0) && (nmu == 0);
}

#endif // HBB_HELPERS_0L_DEFINED

#ifndef HBB_HELPERS_0L_EXTRA_DEFINED
#define HBB_HELPERS_0L_EXTRA_DEFINED

inline float Hbb_pTbb(const RVec<float>& jet_pt, const RVec<float>& jet_eta,
                       const RVec<float>& jet_phi, const RVec<int>& jet_jvt,
                       const RVec<int>& jet_btagq) {
    auto idx = Hbb_bjetIndices(jet_pt, jet_eta, jet_jvt, jet_btagq);
    if (idx.size() < 2) return -1.0f;
    float px = jet_pt[idx[0]] * std::cos(jet_phi[idx[0]]) + jet_pt[idx[1]] * std::cos(jet_phi[idx[1]]);
    float py = jet_pt[idx[0]] * std::sin(jet_phi[idx[0]]) + jet_pt[idx[1]] * std::sin(jet_phi[idx[1]]);
    return std::sqrt(px * px + py * py);
}

inline float Hbb_dRbb(const RVec<float>& jet_pt, const RVec<float>& jet_eta,
                       const RVec<float>& jet_phi, const RVec<int>& jet_jvt,
                       const RVec<int>& jet_btagq) {
    auto idx = Hbb_bjetIndices(jet_pt, jet_eta, jet_jvt, jet_btagq);
    if (idx.size() < 2) return -1.0f;
    float deta = jet_eta[idx[0]] - jet_eta[idx[1]];
    float dphi = Hbb_dphi(jet_phi[idx[0]], jet_phi[idx[1]]);
    return std::sqrt(deta * deta + dphi * dphi);
}

// Delta_phi(b1,b2) 
inline float Hbb_dPhibb(const RVec<float>& jet_pt, const RVec<float>& jet_eta,
                         const RVec<float>& jet_phi, const RVec<int>& jet_jvt,
                         const RVec<int>& jet_btagq) {
    auto idx = Hbb_bjetIndices(jet_pt, jet_eta, jet_jvt, jet_btagq);
    if (idx.size() < 2) return -1.0f;
    return Hbb_dphi(jet_phi[idx[0]], jet_phi[idx[1]]);
}

// Delta_phi(MET,bb)
inline float Hbb_dPhiMetBB(const RVec<float>& jet_pt, const RVec<float>& jet_eta,
                            const RVec<float>& jet_phi, const RVec<int>& jet_jvt,
                            const RVec<int>& jet_btagq, float met_phi) {
    auto idx = Hbb_bjetIndices(jet_pt, jet_eta, jet_jvt, jet_btagq);
    if (idx.size() < 2) return -1.0f;
    float px = jet_pt[idx[0]] * std::cos(jet_phi[idx[0]]) + jet_pt[idx[1]] * std::cos(jet_phi[idx[1]]);
    float py = jet_pt[idx[0]] * std::sin(jet_phi[idx[0]]) + jet_pt[idx[1]] * std::sin(jet_phi[idx[1]]);
    float bb_phi = std::atan2(py, px);
    return Hbb_dphi(met_phi, bb_phi);
}

// min Delta_phi(MET, jets) 
inline float Hbb_minDPhiMetJets(const RVec<float>& jet_pt, const RVec<float>& jet_eta,
                                 const RVec<float>& jet_phi, const RVec<int>& jet_jvt,
                                 float met_phi) {
    auto cjet = Hbb_cjetMask(jet_pt, jet_eta, jet_jvt);
    auto fjet = Hbb_fjetMask(jet_pt, jet_eta, jet_jvt);
    float minDphi = -1.0f;
    for (size_t i = 0; i < jet_pt.size(); ++i) {
        if (cjet[i] || fjet[i]) {
            float d = Hbb_dphi(jet_phi[i], met_phi);
            if (minDphi < 0.0f || d < minDphi) minDphi = d;
        }
    }
    return minDphi;
}

inline float Hbb_HT(const RVec<float>& jet_pt, const RVec<float>& jet_eta,
                     const RVec<int>& jet_jvt) {
    auto cjet = Hbb_cjetMask(jet_pt, jet_eta, jet_jvt);
    auto fjet = Hbb_fjetMask(jet_pt, jet_eta, jet_jvt);
    float ht = 0.0f;
    for (size_t i = 0; i < jet_pt.size(); ++i)
        if (cjet[i] || fjet[i]) ht += jet_pt[i];
    return ht;
}

inline float Hbb_ptBalance(const RVec<float>& jet_pt, const RVec<float>& jet_eta,
                            const RVec<float>& jet_phi, const RVec<int>& jet_jvt,
                            const RVec<int>& jet_btagq, float met) {
    if (met <= 0.0f) return -999.0f;
    float ptbb = Hbb_pTbb(jet_pt, jet_eta, jet_phi, jet_jvt, jet_btagq);
    if (ptbb < 0.0f) return -999.0f;
    return (ptbb - met) / met;
}

inline float Hbb_cosThetaStar(const RVec<float>& jet_pt, const RVec<float>& jet_eta,
                               const RVec<float>& jet_phi, const RVec<float>& jet_e,
                               const RVec<int>& jet_jvt, const RVec<int>& jet_btagq) {
    auto idx = Hbb_bjetIndices(jet_pt, jet_eta, jet_jvt, jet_btagq);
    if (idx.size() < 2) return -999.0f;
    int i0 = idx[0], i1 = idx[1];
    float e0 = Hbb_ptRecoCorrE(jet_pt[i0], jet_e[i0]);
    float e1 = Hbb_ptRecoCorrE(jet_pt[i1], jet_e[i1]);
    TLorentzVector v0, v1;
    v0.SetPtEtaPhiE(jet_pt[i0], jet_eta[i0], jet_phi[i0], e0);
    v1.SetPtEtaPhiE(jet_pt[i1], jet_eta[i1], jet_phi[i1], e1);
    TLorentzVector vH = v0 + v1;
    TVector3 boostDir = vH.BoostVector();  
    TLorentzVector v0_star = v0;
    v0_star.Boost(-boostDir);              
    if (boostDir.Mag() < 1e-6) return -999.0f;
    return (float)std::cos(v0_star.Vect().Angle(boostDir));
}

#endif // HBB_HELPERS_0L_EXTRA_DEFINED

#ifndef HBB_HELPERS_0L_BDT_DEFINED
#define HBB_HELPERS_0L_BDT_DEFINED

inline int Hbb_njetSel(const RVec<float>& jet_pt, const RVec<float>& jet_eta,
                        const RVec<int>& jet_jvt) {
    auto cjet = Hbb_cjetMask(jet_pt, jet_eta, jet_jvt);
    auto fjet = Hbb_fjetMask(jet_pt, jet_eta, jet_jvt);
    int njet = 0;
    for (size_t i = 0; i < jet_pt.size(); ++i)
        if (cjet[i] || fjet[i]) njet++;
    return njet;
}

inline RVec<int> Hbb_selJetIndices(const RVec<float>& jet_pt, const RVec<float>& jet_eta,
                                    const RVec<int>& jet_jvt) {
    auto cjet = Hbb_cjetMask(jet_pt, jet_eta, jet_jvt);
    auto fjet = Hbb_fjetMask(jet_pt, jet_eta, jet_jvt);
    RVec<int> idx;
    for (size_t i = 0; i < jet_pt.size(); ++i)
        if (cjet[i] || fjet[i]) idx.push_back((int)i);
    return idx;
}

inline float Hbb_pTb1(const RVec<float>& jet_pt, const RVec<float>& jet_eta,
                       const RVec<int>& jet_jvt, const RVec<int>& jet_btagq) {
    auto idx = Hbb_bjetIndices(jet_pt, jet_eta, jet_jvt, jet_btagq);
    return idx.size() >= 1 ? jet_pt[idx[0]] : -1.0f;
}
inline float Hbb_pTb2(const RVec<float>& jet_pt, const RVec<float>& jet_eta,
                       const RVec<int>& jet_jvt, const RVec<int>& jet_btagq) {
    auto idx = Hbb_bjetIndices(jet_pt, jet_eta, jet_jvt, jet_btagq);
    return idx.size() >= 2 ? jet_pt[idx[1]] : -1.0f;
}

// m_eff = H_T + MET 
inline float Hbb_meff(const RVec<float>& jet_pt, const RVec<float>& jet_eta,
                       const RVec<int>& jet_jvt, float met) {
    return Hbb_HT(jet_pt, jet_eta, jet_jvt) + met;
}

inline int Hbb_jet3Index(const RVec<float>& jet_pt, const RVec<float>& jet_eta,
                          const RVec<int>& jet_jvt, const RVec<int>& jet_btagq) {
    auto selIdx = Hbb_selJetIndices(jet_pt, jet_eta, jet_jvt);
    auto bIdx   = Hbb_bjetIndices(jet_pt, jet_eta, jet_jvt, jet_btagq);
    if (selIdx.size() != 3 || bIdx.size() != 2) return -1;
    for (int i : selIdx) {
        if (i != bIdx[0] && i != bIdx[1]) return i;
    }
    return -1;
}

inline float Hbb_jet3Pt(const RVec<float>& jet_pt, const RVec<float>& jet_eta,
                         const RVec<int>& jet_jvt, const RVec<int>& jet_btagq) {
    int i3 = Hbb_jet3Index(jet_pt, jet_eta, jet_jvt, jet_btagq);
    return i3 >= 0 ? jet_pt[i3] : -1.0f;
}

inline float Hbb_mbbj(const RVec<float>& jet_pt, const RVec<float>& jet_eta,
                       const RVec<float>& jet_phi, const RVec<float>& jet_e,
                       const RVec<int>& jet_jvt, const RVec<int>& jet_btagq) {
    auto bIdx = Hbb_bjetIndices(jet_pt, jet_eta, jet_jvt, jet_btagq);
    int i3 = Hbb_jet3Index(jet_pt, jet_eta, jet_jvt, jet_btagq);
    if (bIdx.size() < 2 || i3 < 0) return -1.0f;
    int i0 = bIdx[0], i1 = bIdx[1];
    float e0 = Hbb_ptRecoCorrE(jet_pt[i0], jet_e[i0]);
    float e1 = Hbb_ptRecoCorrE(jet_pt[i1], jet_e[i1]);
    TLorentzVector v0, v1, v2;
    v0.SetPtEtaPhiE(jet_pt[i0], jet_eta[i0], jet_phi[i0], e0);
    v1.SetPtEtaPhiE(jet_pt[i1], jet_eta[i1], jet_phi[i1], e1);
    v2.SetPtEtaPhiE(jet_pt[i3], jet_eta[i3], jet_phi[i3], jet_e[i3]);
    return (float)((v0 + v1 + v2).M());
}

#endif // HBB_HELPERS_0L_BDT_DEFINED
"""

ROOT.gInterpreter.Declare(_HBB_CPP)


_TRIGGER = "trigMET"
_MET30   = "met > 150.0f"
_ZERO_LEP = (
    "Hbb_passZeroLep(lep_type, lep_pt, lep_eta, lep_d0sig, lep_z0,"
    " lep_isLooseIso, lep_isLooseID)"
)
_JET_SEL        = "Hbb_passJetSel(jet_pt, jet_eta, jet_jvt, jet_btag_quantile)"
_HT_SEL         = "Hbb_passHT(jet_pt, jet_eta, jet_jvt)"
_TWO_B_PHI      = "Hbb_pass2bPhi(jet_pt, jet_eta, jet_phi, jet_jvt, jet_btag_quantile)"
_METBB_PHI      = "Hbb_passMetBBPhi(jet_pt, jet_eta, jet_phi, jet_jvt, jet_btag_quantile, met_phi)"
_MIN_METJET_PHI = "Hbb_passMinMetJetsPhi(jet_pt, jet_eta, jet_phi, jet_jvt, met_phi)"

SELECTIONS_HBB_0L = [
    SelectionSpec(
        name="presel",
        cut=f"{_TRIGGER} && {_MET30} && {_ZERO_LEP} && {_JET_SEL}",
    ),
    SelectionSpec(
        name="signal_region",
        cut=(
            f"{_TRIGGER} && {_MET30} && {_ZERO_LEP} && {_JET_SEL}"
            f" && {_HT_SEL} && {_TWO_B_PHI} && {_METBB_PHI} && {_MIN_METJET_PHI}"
        ),
    ),
    SelectionSpec(
        name="multijet_CR",
        # NOT A REAL CR, TESTING PURPOSE ONLY
        cut=(
            f"{_TRIGGER} && {_MET30} && {_ZERO_LEP} && {_JET_SEL}"
            f" && {_HT_SEL} && {_TWO_B_PHI} && {_METBB_PHI}"
            f" && !({_MIN_METJET_PHI})"
        ),
    ),
]

_NJET_SEL = "Hbb_njetSel(jet_pt, jet_eta, jet_jvt)"

SELECTIONS_HBB_0L += [
    SelectionSpec(
        name="signal_region_2j",
        cut=(
            next(s.cut for s in SELECTIONS_HBB_0L if s.name == "signal_region")
            + f" && ({_NJET_SEL} == 2)"
        ),
    ),
    SelectionSpec(
        name="signal_region_3j",
        cut=(
            next(s.cut for s in SELECTIONS_HBB_0L if s.name == "signal_region")
            + f" && ({_NJET_SEL} == 3)"
        ),
    ),
]


_MBB_EXPR = "Hbb_mbb(jet_pt, jet_eta, jet_phi, jet_e, jet_jvt, jet_btag_quantile)"

# variables VH->bb (cf. JHEP12(2017)024, arXiv:2007.02873) ──────
_JETKIN_ARGS  = "jet_pt, jet_eta, jet_phi, jet_jvt, jet_btag_quantile"
_PTBB_EXPR    = f"Hbb_pTbb({_JETKIN_ARGS})"
_DRBB_EXPR    = f"Hbb_dRbb({_JETKIN_ARGS})"
_DPHIBB_EXPR  = f"Hbb_dPhibb({_JETKIN_ARGS})"
_DPHIMETBB_EXPR = f"Hbb_dPhiMetBB({_JETKIN_ARGS}, met_phi)"
_MINDPHI_EXPR = "Hbb_minDPhiMetJets(jet_pt, jet_eta, jet_phi, jet_jvt, met_phi)"
_HT_EXPR      = "Hbb_HT(jet_pt, jet_eta, jet_jvt)"
_PTBAL_EXPR   = f"Hbb_ptBalance({_JETKIN_ARGS}, met)"
_COSTHETA_EXPR = "Hbb_cosThetaStar(jet_pt, jet_eta, jet_phi, jet_e, jet_jvt, jet_btag_quantile)"

# BDT_VH 
# p_T^V (=MET), p_T^b1, p_T^b2, m_bb, DeltaR(b1,b2), Delta_phi(V,bb), m_eff
_PTB1_EXPR   = "Hbb_pTb1(jet_pt, jet_eta, jet_jvt, jet_btag_quantile)"
_PTB2_EXPR   = "Hbb_pTb2(jet_pt, jet_eta, jet_jvt, jet_btag_quantile)"
_MEFF_EXPR   = "Hbb_meff(jet_pt, jet_eta, jet_jvt, met)"
_JET3_PT_EXPR = "Hbb_jet3Pt(jet_pt, jet_eta, jet_jvt, jet_btag_quantile)"
_MBBJ_EXPR    = "Hbb_mbbj(jet_pt, jet_eta, jet_phi, jet_e, jet_jvt, jet_btag_quantile)"

HISTOS_HBB_0L = [
    HistoSpec("h_mbb",         _MBB_EXPR,       "m_{bb} [GeV]",                      24,  20, 500),
    HistoSpec("h_met",         "met",           "E_{T}^{miss} [GeV]",                40,   0, 300),
    HistoSpec("h_jet_n",       "jet_n",         "N_{jets}",                          10,   0,  10),
    HistoSpec("h_jet_pt",      "jet_pt[0]",     "Leading jet p_{T} [GeV]",           40,  20, 400),
    HistoSpec("h_ptbb",        _PTBB_EXPR,      "p_{T}^{bb} [GeV]",                  30,   0, 500),
    HistoSpec("h_drbb",        _DRBB_EXPR,      "#Delta R(b_{1},b_{2})",             25,   0,   5),
    HistoSpec("h_dphibb",      _DPHIBB_EXPR,    "#Delta#phi(b_{1},b_{2})",           32,   0, 3.2),
    HistoSpec("h_dphi_metbb",  _DPHIMETBB_EXPR, "#Delta#phi(E_{T}^{miss},bb)",       32,   0, 3.2),
    HistoSpec("h_mindphi_mj",  _MINDPHI_EXPR,   "min #Delta#phi(E_{T}^{miss},jets)", 32,   0, 3.2),
    HistoSpec("h_ht",          _HT_EXPR,        "H_{T} [GeV]",                       30,   0, 800),
    HistoSpec("h_ptbalance",   _PTBAL_EXPR,     "(p_{T}^{bb}-E_{T}^{miss})/E_{T}^{miss}", 40, -2, 2),
    HistoSpec("h_costhetastar",_COSTHETA_EXPR,  "|cos#theta*|",                      20,  -1,   1),
    HistoSpec("h_jet3_pt",     _JET3_PT_EXPR,   "3rd jet p_{T} [GeV] (si njet==3)",  30,  -1, 300),
    HistoSpec("h_ptb1",        _PTB1_EXPR,      "p_{T}^{b1} [GeV]",                  40,   0, 400),
    HistoSpec("h_ptb2",        _PTB2_EXPR,      "p_{T}^{b2} [GeV]",                  40,   0, 300),
    HistoSpec("h_meff",        _MEFF_EXPR,      "m_{eff} = H_{T}+E_{T}^{miss} [GeV]",30,   0,1000),
    HistoSpec("h_mbbj",        _MBBJ_EXPR,      "m_{bbj} [GeV] (si njet==3)",        30,  -1, 700),
]

HISTO_BDT_VH = HistoSpec("h_bdt_vh", "BDT_VH", "BDT_{VH} score", 15, -1, 1)

BDT_VARS_COMMON_0L = {
    "pT_V":      "met",           # p_T^V == MET canal 0-lepton
    "pT_b1":     _PTB1_EXPR,
    "pT_b2":     _PTB2_EXPR,
    "m_bb":      _MBB_EXPR,
    "dR_bb":     _DRBB_EXPR,
    "dPhi_Vbb":  _DPHIMETBB_EXPR,
    "m_eff":     _MEFF_EXPR,
}
BDT_VARS_3J_EXTRA_0L = {
    "pT_jet3":   _JET3_PT_EXPR,
    "m_bbj":     _MBBJ_EXPR,
}



import glob as _glob

#BASE = "/eos/opendata/atlas/rucio/opendata/"
BASE = "/afs/cern.ch/user/m/movincen/eos/2026SuMIFIC/"
SKIM = "2bjets70"   
PFX  = f"{BASE}" #ODEO_FEB2025_v0_{SKIM}_"


def _resolve(pattern: str) -> list[str]:
    return sorted(_glob.glob(pattern))


def _mc(dsid: int, label: str) -> DSIDEntry:
    return DSIDEntry(
        dsid=dsid,
        label=label,
 #       file_patterns=_resolve(f"{PFX}mc_{dsid}.*.{SKIM}.root"),
       file_patterns=_resolve(f"{PFX}mc_{dsid}.*.{SKIM}.root"),
    )


_PROCESS_DSIDS: dict[str, list[int]] = {
    "ttbar":     [410470, 410471],
    "signal":    [345056, 345058, 345949, 346311, 346312],
    "singletop": [410644, 410645, 410658, 410659, 601624, 601628],
    "Vjets":     [700320, 700321, 700322, 700323, 700324, 700325, 700335, 700336,
                  700337, 700338, 700339, 700340, 700341, 700342, 700343, 700344, 700345,
                  700346, 700347, 700348, 700349, 700467, 700468, 700469, 700470,
                  700471, 700472, 700792, 700793, 700794],
    "diboson":   [700488, 700489, 700490, 700491, 700492,
                  700493, 700494, 700495, 700496, 700195,
                  700196, 700199, 700200, 700201],
    "ttV": [410156],
}
DATA_DSID = 0

SAMPLES_HBB_0L: list[DSIDEntry] = [
    DSIDEntry(
        dsid=0, label="data",
        file_patterns=_resolve(f"{PFX}data1*_period*.{SKIM}.root"),
        is_data=True,
    ),
]
for _process_name, _dsids in _PROCESS_DSIDS.items():
    for _dsid in _dsids:
        SAMPLES_HBB_0L.append(_mc(_dsid, _process_name))

GROUPS_HBB_0L = _PROCESS_DSIDS


def save_grouped(
    proc: HTauTauDSIDProcessor,
    groups: dict[str, list[int]],
    data_dsid: int,
    output_file: str,
) -> None:
    out = ROOT.TFile(output_file, "RECREATE")
    _dir_cache: dict[str, "ROOT.TDirectory"] = {}

    def _get_or_create_dir(dir_path: str) -> "ROOT.TDirectory":
        if dir_path in _dir_cache:
            return _dir_cache[dir_path]
        current = out
        accum = ""
        for part in dir_path.split("/"):
            accum = f"{accum}/{part}" if accum else part
            if accum in _dir_cache:
                current = _dir_cache[accum]
                continue
            sub = current.GetDirectory(part)
            if not sub:
                sub = current.mkdir(part, "", True)
            _dir_cache[accum] = sub
            current = sub
        return current

    def _write(folder: str, sel_name: str, spec_name: str, h) -> None:
        if h is None:
            print(f"  [WARN] hist : {sel_name}/{folder}/{spec_name}")
            return
        dir_path = f"{sel_name}/{folder}" 
        tdir = _get_or_create_dir(dir_path)
        tdir.cd()
        h.SetName(spec_name)
        h.SetDirectory(tdir)
        h.Write()

    for sel in proc.selections:
        for spec in proc.histo_specs:
            try:
                h_data = proc.get_histogram(data_dsid, sel.name, spec.name)
            except KeyError:
                h_data = None
            _write("data", sel.name, spec.name, h_data)

            for process, dsids in groups.items():
                h_grp = proc.sum_over_dsids(dsids, sel.name, spec.name)
                _write(process, sel.name, spec.name, h_grp)

    out.Write()
    out.Close()

if __name__ == "__main__":

    proc = HTauTauDSIDProcessor(
        dsid_entries  = SAMPLES_HBB_0L,
        selections    = SELECTIONS_HBB_0L,
        histo_specs   = HISTOS_HBB_0L,
        luminosity_fb = 36.1,
        output_file   = "hbb_0lep_histograms.root",  
        debug_mode    = False,  
    )

    proc.book()
    proc.run()

    proc.save()

    save_grouped(
        proc,
        groups=GROUPS_HBB_0L,
        data_dsid=DATA_DSID,
        output_file="hbb_0lep_grouped.root",
    )
