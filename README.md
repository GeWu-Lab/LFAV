# Towards Long Form Audio-visual Video Understanding. 

[Project Page](https://gewu-lab.github.io/LFAV/)

### Dataset & Features

#### YouTube ID

The dataset is collected from [YouTube](https://www.youtube.com/), you can find the ID of each video in [annotation files](https://github.com/GeWu-Lab/LFAV/tree/main/LFAV_dataset). 

#### Features

We use VGGish to extract audio features, use ResNet18 and R(2+1)D-18 to extract visual features.

VGGish feature: [Google Drive](https://drive.google.com/file/d/1bvTBotLHnPGIeIAkkgMWK7wcWjZ5xbfo/view), [Baidu Drive](https://pan.baidu.com/share/init?surl=nSdhEilGxGFs-7FOgsDoFw) (pwd: lfav), (~662M).

ResNet18 feature: [Google Drive](https://drive.google.com/file/d/14p4jgDo-tteeZPzRBbEq1982tT-uxviZ/view), [Baidu Drive](https://pan.baidu.com/s/1GAstblAMXbhlUj_8QD_ONg) (pwd: lfav), (~2.6G).

R(2+1)D-18feature: [Google Drive](https://drive.google.com/file/d/1FfLpS0PLPXNJ28SqqYLb_vBATlUWnDvK/view), [Baidu Drive](https://pan.baidu.com/share/init?surl=-jRD7MQ0RT0lAN5DP40syA) (pwd: lfav), (~2.6G).

### Annotations

#### training set 

```
# LFAV training set annotations
cd LFAV_dataset
cd ./train
train_audio_weakly.csv: video-level audio annotaions of training set
train_visual_weakly.csv: video-level visual annotaions of training set
train_weakly.csv: video-level annotations (union of video-level audio annotations and visual annotations) of training set
```

#### validation set

```
# LFAV validation set annotations
cd LFAV_dataset
cd ./val
val_audio_weakly.csv: video-level audio annotaions of validation set
val_visual_weakly.csv: video-level visual annotaions of validation set
val_weakly_av.csv: video-level annotations (union of video-level audio annotations and visual annotations) of validation set
val_audio.csv: event-level audio annotaions of validation set
val_visual.csv: event-level visual annotaions of validation set
```

#### testing set

```
# LFAV testing set annotations
cd LFAV_dataset
cd ./test
test_audio_weakly.csv: video-level audio annotaions of testing set
test_visual_weakly.csv: video-level visual annotaions of testing set
test_weakly_av.csv: video-level annotations (union of video-level audio annotations and visual annotations) of testing set
test_audio.csv: event-level audio annotaions of testing set
test_visual.csv: event-level visual annotaions of testing set
```

### Train and test

The script of training all three phases is in:

```
src/scripts/train_s3.sh
```

If you want to train one or two phases, just edit the arg "num_stages" to 1 or 2.

The script of testing all three phases is in:

```
src/scripts/test_s3.sh
```

We also provide our trained weights of the complete method (three phases): [Google Drive](https://drive.google.com/file/d/10v-1WnUhHf-0ehH8yXJ0pSkGDYtKVdPy/view?usp=sharing),  [Baidu Drive](https://pan.baidu.com/s/1-wki3AfPAz3YnzNmGrC0wA?pwd=lfav) (pwd: lfav).

### Publication(s)

If you find our work useful in your research, please cite our paper.

```
 @ARTICLE{hou2023towards,
          title={Towards Long Form Audio-visual Video Understanding},
          author={Hou, Wenxuan and li, Guangyao and Tian, Yapeng and Hu, Di},
          journal={arXiv preprint arXiv:2306.09431},
          year={2023},
        }
```

### Acknowledgement

The source code referenced [AVVP-ECCV20](https://github.com/YapengTian/AVVP-ECCV20).

### License

This project is released under the [CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/).
