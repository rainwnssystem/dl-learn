#!/bin/bash

sudo dnf install 'dnf-command(config-manager)'
sudo dnf config-manager --add-repo https://cli.github.com/packages/rpm/gh-cli.repo
sudo dnf install gh

sudo yum install -y git python3.12 python3.12-pip

git clone https://github.com/rainwnssystem/dl-learn
cd dl-learn

python3.12 -m venv .venv
. .venv/bin/activate

pip install numpy pandas scikit-learn tensorflow tensorflow[and-cuda] torch torchvision matplotlib seaborn
pip uninstall nvidia-cudnn-cu12


# Keras - GPU 인식 확인
cat >> $VIRTUAL_ENV/bin/activate << 'EOF'
_nvidia_base="$VIRTUAL_ENV/lib/python3.$(python -c 'import sys; print(sys.version_info.minor)')/site-packages/nvidia"
if [ -d "$_nvidia_base" ]; then
    for _pkg_dir in "$_nvidia_base"/*/lib; do
        [ -d "$_pkg_dir" ] && export LD_LIBRARY_PATH="$_pkg_dir:${LD_LIBRARY_PATH:-}"
    done
fi
unset _nvidia_base _pkg_dir
EOF
deactivate
source .venv/bin/activate