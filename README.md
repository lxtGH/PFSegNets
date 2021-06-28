# PFSegNets
This repo contains the the implementation of Our CVPR-2021 work: PointFlow: Flowing Semantics Through Points for Aerial Image Segmentation

The master branch works with PyTorch 1.5 and python 3.7.6.
# DataSet preparation
1. Downloading [iSAID](https://captain-whu.github.io/iSAID/), [Potsdam](https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-potsdam/) and
 [Vahihigen](https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-vaihingen/) dataset.
2. Using scripts to crop [iSAID](tools/split_iSAID.py) and [Potsdam, Vaihigen](tools/split_isprs.py) into patches.
3. Using scripts to convert the original mask of [iSAID](tools/convert_iSAID_mask2graymask.py) and [Potsdam, Vaihigen](tools/convert_isprs_mask2graymask.py) 
into gray mask for training and evaluating.
4. Finally, you can either change the `config.py` or do the soft link according to the default path in config.

For example, suppose you store your iSAID dataset at `~/username/data/iSAID`, please update the dataset path in `config.py`,
```
__C.DATASET.iSAID_DIR = '~/username/data/iSAID'
``` 
Or, you can link the data path into current folder.

```
mkdir data 
cd data
ln -s your_iSAID_root_data_path iSAID
```

Actually, the order of steps 2 and 3 is interchangeable.

## Pretrained Models

Baidu Pan Link: https://pan.baidu.com/s/1MWzpkI3PwtnEl1LSOyLrLw  4lwf 

Google Drive Link: https://drive.google.com/drive/folders/1C7YESlSnqeoJiR8DWpmD4EVWvwf9rreB?usp=sharing

After downloading the pretrained ResNet, you can either change the model path of `network/resnet_d.py` or do the soft link according to the default path in `network/resnet_d.py`.

For example, 
Suppose you store the pretrained ResNet50 model at `~/username/pretrained_model/resnet50-deep.pth`, please update the 
dataset path in Line315 of `config.py`,
```
model.load_state_dict(torch.load("~/username/pretrained_model/resnet50-deep.pth", map_location='cpu'))
```
Or, you can link the pretrained model path into current folder.
```
mkdir pretrained_models
ln -s your_pretrained_model_path path_to_pretrained_models_folder
```

# Model Checkpoints

  <table><thead><tr><th>Dataset</th><th>Backbone</th><th>mIoU</th><th>Model</th></tr></thead><tbody>
<tr><td>iSAID</td><td>ResNet50</td><td>66.9</td><td><a href="https://drive.google.com/file/d/1igB0y-5IybcIxf0cALFoqh0Pg36OxWR-/view?usp=sharing" target="_blank" rel="noopener noreferrer">Google Drive</a>&nbsp;|&nbsp;<a href="https://pan.baidu.com/s/1xX2DXdQ5SdpKA3w2EAdZUA" target="_blank" rel="noopener noreferrer">Baidu Pan</a>(v3oj)</td></tr>
<tr><td>Potsdam</td><td>ResNet50</td><td>75.4</td><td><a href="https://drive.google.com/file/d/1tVvPLaMLBp55HfyDhRgmRcMOW44CSc6s/view?usp=sharing" target="_blank" rel="noopener noreferrer">Google Drive</a>&nbsp;|&nbsp;<a href="https://pan.baidu.com/s/1NX1k80NBIrA_G03AsmzZ1w" target="_blank" rel="noopener noreferrer">Baidu Pan</a>(lhlf)</td></tr>
<tr><td>Vaihigen</td><td>ResNet50</td><td>70.4</td><td><a href="https://drive.google.com/file/d/1C3FrXPo8-LuBGUJcC6PCcMP-FP8zVXXb/view?usp=sharing" rel="noopener noreferrer">Google Drive</a>&nbsp;|&nbsp;<a href="https://pan.baidu.com/s/1LSOViE817pS2XpzMPCBbwA" target="_blank" rel="noopener noreferrer">Baidu Pan</a>(54qm)</td></tr>
</tbody></table>

# Evaluation

For example, when evaluating PFNet on validation set of iSAID dasaset:
```bash
sh scripts/pointflow/test/test_iSAID_pfnet_R50.pth path_to_checkpoint path_to_save_results
```
If you want to save images during evaluating, add args: `dump_images`, which will take more time.

# Training

To be note that, our models are trained on 8 V-100 GPUs with 32GB memory.
 **It is hard to reproduce such best results if you do not have such resources.**
For example, when training PFNet on iSAID dataset:
```bash
sh scripts/pointflow/train_iSAID_pfnet_r50.sh
```

# Citation
If you find this repo is helpful to your research. Please consider cite our work.

```
@inproceedings{li2021pointflow,
  title={PointFlow: Flowing Semantics Through Points for Aerial Image Segmentation},
  author={Li, Xiangtai and He, Hao and Li, Xia and Li, Duo and Cheng, Guangliang and Shi, Jianping and Weng, Lubin and Tong, Yunhai and Lin, Zhouchen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4217--4226},
  year={2021}
}
```

# Acknowledgement
This repo is based on NVIDIA segmentation [repo](https://github.com/NVIDIA/semantic-segmentation). 
We fully thank their open-sourced code.