#!/bin/bash

sudo dnf install 'dnf-command(config-manager)'
sudo dnf config-manager --add-repo https://cli.github.com/packages/rpm/gh-cli.repo
sudo dnf install gh

sudo yum install -y git python3.12 python3.12-pip

echo "alias python='/usr/bin/python3.12'" >> ~/.bashrc
echo "alias pip='/usr/bin/pip3.12'" >> ~/.bashrc
. ~/.bashrc

git clone https://github.com/rainwnssystem/dl-learn
cd dl-learn

python -m venv .

. ./bin/activate

pip install numpy pandas scikit-learn tensorflow tensorflow[and-cuda] torch torchvision matplotlib seaborn