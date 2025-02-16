add-apt-repository ppa:deadsnakes/ppa -y
apt install python3.8 -y
apt install python3.8-distutils -y
apt install virtualenv -y
virtualenv --python=python3.8 ./server_venv
#./server_venv/bin/python -m pip install -r reqirements.txt
