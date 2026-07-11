#!/bin/bash
echo 'Welcome to the Hbb analysis package'
echo 'Download it, run it, and get VH(bb) plots.'
# Installation on lxplus 
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase

if [ -z "$CI" ]; then
    source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
    asetup 25.2.50,AnalysisBase
else
    source ~/release_setup.sh
fi
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python -m venv .venv --system-site-packages
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
source .venv/bin/activate

pip install --quiet --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install --quiet -r requirements.txt
fi
# -----------------------------------------------------------------------------
export PYTHONPATH=$PWD:${PYTHONPATH}
export HBB_SKIM="1LMET30"
export HBB_LUMI_FB="36.1"
export HBB_EOS_BASE="/eos/opendata/atlas/rucio/opendata/"
export HBB_OUTPUT_DIR="${PWD}/outputs"
mkdir -p "${HBB_OUTPUT_DIR}"

echo ""
echo "=================================================================="
echo " Hbb analysis package ready"
echo "   PYTHONPATH   : ${PYTHONPATH}"
echo "   Skim         : ${HBB_SKIM}"
echo "   Luminosity   : ${HBB_LUMI_FB} fb-1"
echo "   Output dir   : ${HBB_OUTPUT_DIR}"
echo ""
echo " Run the analysis with:"
echo "   python3 Hbb_rdf.py"
echo " Produce plots with:"
echo "   python3 plot_atlas_style.py"
echo "=================================================================="
