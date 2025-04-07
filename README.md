# DifferSketching: How Differently Do People Sketch 3D Objects?
> [Chufeng Xiao](https://scholar.google.com/citations?user=2HLwZGYAAAAJ&hl=en), [Wanchao Su](https://ansire.github.io/), [Jing Liao](https://liaojing.github.io/html/), [Zhouhui Lian](https://www.icst.pku.edu.cn/zlian/), [Yi-Zhe Song](http://personal.ee.surrey.ac.uk/Personal/Y.Song/), [Hongbo Fu](https://sweb.cityu.edu.hk/hongbofu/)
> 
> [[Project Page]](https://chufengxiao.github.io/DifferSketching/) [[Paper]](https://arxiv.org/abs/2209.08791) [[Dataset]](https://chufengxiao.github.io/DifferSketching/#dataset) [[Supplemental Material]](https://github.com/chufengxiao/DifferSketching/tree/project-page/Supplemental_Material)
>
> Accepted by [SIGGRAPH Asia 2022](https://sa2022.siggraph.org/) (Journal Track)

## To-do List
  - [x] [Dataset](https://chufengxiao.github.io/DifferSketching/#dataset)
  - [ ] Multi-level Registration Method
  - [ ] Freehand-style Sketch Synthesis
    - [x] Pre-trained models and inference code
    - [ ] Training code and training dataset

## Freehand-style Sketch Synthesis

Please run the below commands to visualize the pipeline of our method for sketch synthesis. The pre-trained models of our method are located at the directory `./sketch_synthesis/model/`. There are two examples in the directory `./sketch_synthesis/input_sketch_json/` for testing, and you can also pick up other data from `<category>/reg_json/` under the release dataset directory. The visualization result will be save in the directory `./sketch_synthesis/results/`.

```bash
cd ./sketch_synthesis
pip install -r requirements.txt
python test.py
```

