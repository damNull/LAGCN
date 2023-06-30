# LA-GCN

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/language-knowledge-assisted-representation/skeleton-based-action-recognition-on-ntu-rgbd-1)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-ntu-rgbd-1?p=language-knowledge-assisted-representation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/language-knowledge-assisted-representation/skeleton-based-action-recognition-on-ntu-rgbd)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-ntu-rgbd?p=language-knowledge-assisted-representation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/language-knowledge-assisted-representation/skeleton-based-action-recognition-on-n-ucla)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-n-ucla?p=language-knowledge-assisted-representation)

Code for LAGCN

> **[Language Knowledge-Assisted Representation Learning for Skeleton-Based Action Recognition](https://arxiv.org/abs/2305.12398)**
>
> Haojun Xu, Yan Gao, Zheng Hui, Jie Li, Xinbo Gao
> 
> *[arXiv 2305.12398](https://arxiv.org/abs/2305.12398)*

## TODO List

* [x] Upload NW-UCLA configs
* [x] Add ensemble code
* [x] Upload pretrained weights
* [ ] Add code of generating CPR graph (loss relevant)
* [ ] Add code of generating GPR graph (input data relevant)

## Data Preparation

### Download datasets.

#### There are 3 datasets to download:

- NTU RGB+D 60 Skeleton
- NTU RGB+D 120 Skeleton
- NW-UCLA

#### NTU RGB+D 60 and 120

1. Request dataset here: https://rose1.ntu.edu.sg/dataset/actionRecognition
2. Download the skeleton-only datasets:
   1. `nturgbd_skeletons_s001_to_s017.zip` (NTU RGB+D 60)
   2. `nturgbd_skeletons_s018_to_s032.zip` (NTU RGB+D 120)
   3. Extract above files to `./data/nturgbd_raw`

#### NW-UCLA

1. Download dataset from [here](https://www.dropbox.com/s/10pcm4pksjy6mkq/all_sqe.zip?dl=0)
2. Move `all_sqe` to `./data/NW-UCLA`

### Data Processing

#### Directory Structure

Put downloaded data into the following directory structure:

```
- data/
  - NW-UCLA/
    - all_sqe
      ... # raw data of NW-UCLA
  - ntu/
  - ntu120/
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
      ...
```

#### Generating Data

- Generate NTU RGB+D 60 or NTU RGB+D 120 dataset:

```
 cd ./data/ntu # or cd ./data/ntu120
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
```



# Training & Testing

### Training

- Change the config file depending on what you want.

```
# Example: training LAGCN on NTU RGB+D 120 cross subject with GPU 0
python main.py --config configs/ntu120-xsub/joint.yaml --work-dir work_dir/ntu120/csub/lagcn_joint --device 0
```

- To train your own model, put model file `your_model.py` under `./model` and run:

```
# Example: training your own model on NTU RGB+D 120 cross subject
python main.py --config config/ntu120-xsub/joint.yaml --model model.your_model.Model --work-dir work_dir/ntu120/xsub/your_model --device 0
```

### Testing

- To test the trained models saved in <work_dir>, run the following command:

```
python main.py --config <work_dir>/config.yaml --work-dir <work_dir> --phase test --save-score True --weights <work_dir>/xxx.pt --device 0
```

- To ensemble the results of different modalities, run 
```
# Example: ensemble four modalities of LAGCN on NTU RGB+D 120 cross subject
python ensemble.py --datasets ntu120/xsub --joint work_dir/ntu120/xsub/j.pkl --bone work_dir/ntu120/xsub/b.pkl --joint-motion work_dir/ntu120/xsub/jm.pkl --bone-motion work_dir/ntu120/xsub/bm.pkl
# Ensemble six modalities of LAGCN on NTU RGB+D 120 cross subject
python ensemble_6s.py --datasets ntu120/xsub --joint work_dir/ntu120/xsub/j.pkl --bone work_dir/ntu120/xsub/b.pkl --joint-motion work_dir/ntu120/xsub/jm.pkl --bone-motion work_dir/ntu120/xsub/bm.pkl --prompt work_dir/ntu120/xsub/p2.pkl --prompt2 work_dir/ntu120/xsub/p5.pkl
```

### Pretrained Models

Pretrained weights and validation set inference results are provided in the [link](https://drive.google.com/file/d/1Yz86jwjj_EAeqf8-KVBPsM9lVQWz-mCM/view?usp=drive_link) and [link](https://drive.google.com/file/d/1fOfhQGV8N6kJvGAmD02Kyigs58Ytrie0/view?usp=drive_link) respectively.


The performance of NW-UCLA dataset is slightly different from the article table. The reason is that the MHA-GC used in NW-UCLA experiment is the numerical approximation version (Fig. 7b right). We will modify the relevant table of artivle in next version.

## Acknowledgements

This repo is based on [2s-AGCN](https://github.com/lshiwjx/2s-AGCN). The data processing is borrowed from [SGN](https://github.com/microsoft/SGN) and [HCN](https://github.com/huguyuehuhu/HCN-pytorch).

Thanks to the original authors for their work!

# Citation

Please cite this work if you find it useful:
```
@article{xu2023language,
      title={Language Knowledge-Assisted Representation Learning for Skeleton-Based Action Recognition}, 
      author={Haojun Xu and Yan Gao and Zheng Hui and Jie Li and Xinbo Gao},
      year={2023},
      primaryClass={cs.CV},
      journal={CoRR},
      volume={abs/2305.12398},
      url={https://arxiv.org/abs/2305.12398}
}
```

# Contact
For any questions, feel free to contact: `damnull@outlook.com`
