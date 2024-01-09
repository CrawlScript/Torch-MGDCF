<p align="center">
<img src="MGDCF_LOGO.png" width="400"/>
</p>


# Torch-MGDCF
Source code (PyTorch) and dataset of the paper "[MGDCF: Distance Learning via Markov Graph Diffusion for Neural Collaborative Filtering](https://arxiv.org/abs/2204.02338)", which is accepted by IEEE Transactions on Knowledge and Data Engineering (TKDE).



## Implementations and Paper Links

+ PyTorch Implementation: [Torch-MGDCF](https://github.com/CrawlScript/Torch-MGDCF)
+ TensorFlow Implementation: [TensorFlow-MGDCF](https://github.com/hujunxianligong/MGDCF)
+ Paper Access:
    - **IEEE Xplore**: [https://ieeexplore.ieee.org/document/10384729](https://ieeexplore.ieee.org/document/10384729)
    - **ArXiv**: [https://arxiv.org/abs/2204.02338](https://arxiv.org/abs/2204.02338)






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
@ARTICLE{10384729,
  author={Jun Hu and Bryan Hooi and Shengsheng Qian and Quan Fang and Changsheng Xu},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={MGDCF: Distance Learning via Markov Graph Diffusion for Neural Collaborative Filtering}, 
  year={2024},
  volume={},
  number={},
  pages={1-16},
  doi={10.1109/TKDE.2023.3348537}
}
```