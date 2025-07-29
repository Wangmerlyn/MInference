source /opt/conda/etc/profile.d/conda.sh 
conda create --name cic python==3.10 -y
conda activate cic
which python

pip install vllm
pip install  minference seaborn absl-py rouge_score html2text bs4