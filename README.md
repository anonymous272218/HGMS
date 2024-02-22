
```bash
conda create --name hgms python=3.8
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
#conda install scikit-learn multiprocess 
conda install nltk=3.5 matplotlib
conda install -c dglteam/label/cu113 dgl
conda install -c conda-forge pyyaml attrdict
conda install -c huggingface transformers
conda update tokenizers
pip install tqdm rouge_score

# https://zhuanlan.zhihu.com/p/586768895
# sudo apt-get install libxml-perl libxml-dom-perl
pip install git+git://github.com/bheinzerling/pyrouge
```
