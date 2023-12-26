# Torch-MGDCF
Source code (PyTorch) and dataset of the paper [MGDCF: Distance Learning via Markov Graph Diffusion for Neural Collaborative Filtering](https://arxiv.org/abs/2204.02338)




## Requirements

I have tested this environment with the following requirements, but please note that you are not restricted to these specific versions and can choose other versions as appropriate, since my code does not utilize many advanced operations:

+ Linux
+ Python 3.7
+ torch=1.12.1+cu113
+ numpy >= 1.17.4
+ tqdm
+ dgl=1.1.1+cu113
+ requests=2.28.1
+ scikit-learn=1.0.2
+ faiss-cpu=1.7.4


## Run MGDCF

```bash
sh run_mgdcf_yelp.sh
sh run_mgdcf_gowalla.sh
sh run_mgdcf_amazon-book.sh
```






## Cite

```
@misc{hu2022mgdcf,
      title={MGDCF: Distance Learning via Markov Graph Diffusion for Neural Collaborative Filtering}, 
      author={Jun Hu and Shengsheng Qian and Quan Fang and Changsheng Xu},
      year={2022},
      eprint={2204.02338},
      archivePrefix={arXiv},
      primaryClass={cs.SI}
}
```