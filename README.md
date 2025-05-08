# DifferSketching: How Differently Do People Sketch 3D Objects?
> [Chufeng Xiao](https://scholar.google.com/citations?user=2HLwZGYAAAAJ&hl=en), [Wanchao Su](https://ansire.github.io/), [Jing Liao](https://liaojing.github.io/html/), [Zhouhui Lian](https://www.icst.pku.edu.cn/zlian/), [Yi-Zhe Song](http://personal.ee.surrey.ac.uk/Personal/Y.Song/), [Hongbo Fu](https://sweb.cityu.edu.hk/hongbofu/)
> 
> [[Project Page]](https://chufengxiao.github.io/DifferSketching/) [[Paper]](https://arxiv.org/abs/2209.08791) [[Dataset]](https://chufengxiao.github.io/DifferSketching/#dataset) [[Supplemental Material]](https://github.com/chufengxiao/DifferSketching/tree/project-page/Supplemental_Material)
>
> Accepted by [SIGGRAPH Asia 2022](https://sa2022.siggraph.org/) (Journal Track)

## To-do List
  - [x] [Dataset](https://chufengxiao.github.io/DifferSketching/#dataset)
  - [ ] Multi-level Registration Method
  - [x] Freehand-style Sketch Synthesis
    - [x] Pre-trained models and inference code
    - [x] Training code and training dataset

## Freehand-style Sketch Synthesis
### Quick Test
Please run the below commands to visualize the pipeline of our method for sketch synthesis. The pre-trained models of our method are located at the directory `./sketch_synthesis/model/`. There are two examples in the directory `./sketch_synthesis/input_sketch_json/` for testing, and you can also pick up other data from `<category>/reg_json/` under the release dataset directory. The visualization result will be save in the directory `./sketch_synthesis/results/`.

```bash
cd ./sketch_synthesis
pip install -r requirements.txt
python test.py
```



### Training Dataset for Sketch Synthesis

Please download the latest version of our DifferSketching Dataset (updated in 8 May 2025) via [Google Drive](https://drive.google.com/file/d/1A_3RVc8Y4YdI7nhyM7tb-q7dQw4zTcCO/view) and put it at the root directory. Run the below commands to prepare data for training three MLP disturbers:
```bash
cd ./sketch_synthesis
pip install -r requirements.txt

# The sketch dataset should be located at root_dir="../DifferSketching_Dataset"

python ./prepare_data/getExtrinsicData.py # data for training extrinsic disturber
python ./prepare_data/getIntrinsicData.py # data for training intrinsic disturber
python ./prepare_data/getCurveNoiseData.py # data for training point disturber

# The extracted training data will be save at ./data
```
Please check the codes to switch the dataset from novices or professionals via the variable `NP=N` or `NP=P`. 

### Training
You can train three MLP disturbers using the corresponding data via the below commands:
```bash
cd ./sketch_synthesis

python train_extrinsic.py # train extrinsic disturber
python train_intrinsic.py # train intrinsic disturber
python train_curveNoise.py # train point disturber

# The trained models will be saved at ./train_models. Please check more details in the codes.
```


