# DifferSketching: How Differently Do People Sketch 3D Objects?

###### [Chufeng Xiao](https://scholar.google.com/citations?user=2HLwZGYAAAAJ&hl=en)$^{1,*}$ &nbsp; [Wanchao Su](https://ansire.github.io/)$^{1,2,*}$ &nbsp; [Jing Liao]([LIAO Jing](https://liaojing.github.io/html/))$^2$ &nbsp; [Zhouhui Lian](https://www.icst.pku.edu.cn/zlian/)$^3$ &nbsp; [Yi-Zhe Song](http://personal.ee.surrey.ac.uk/Personal/Y.Song/)$^{4}$ &nbsp; [Hongbo Fu](https://sweb.cityu.edu.hk/hongbofu/)$^{1,\dagger}$

###### 1 School of Creative Media, City University of Hong Kong

###### 2 Department of Computer Science, City University of Hong Kong

###### 3 Wangxuan Institute of Computer Technology, Peking University

###### 4 SketchX, CVSSP, University of Surrey

###### * Joint first authors

###### $\dagger$ Corresponding author

###### Accepted by [SIGGRAPH Asia 2022](https://sa2022.siggraph.org/) (Journal Track)

###### [[Paper]](#) &nbsp;&nbsp;&nbsp;&nbsp; [[Dataset]](#Dataset) &nbsp;&nbsp;&nbsp;&nbsp; [[Code]](#Code) &nbsp;&nbsp;&nbsp;&nbsp; [[Supplemental Material]](#)

![Teaser](index.assets/1662644164790.png)

**Fig 1:** We present DifferSketching, a new dataset of freehand sketches to understand how differently professional and novice users sketch 3D objects (d). We perform three-level data analysis through (a) sketch-level registration, (b) stroke-level registration, and (c) pixel-level registration, to understand the difference in drawings from the two skilled groups.

## Abstract

Multiple sketch datasets have been proposed to understand how people draw 3D objects. However, such datasets are often of small scale and cover a small set of objects or categories. In addition, these datasets contain freehand sketches mostly from expert users, making it difficult to compare the drawings by expert and novice users, while such comparisons are critical in informing more effective sketch-based interfaces for either user groups. These observations motivate us to analyze how differently people with and without adequate drawing skills sketch 3D objects. We invited 70 novice users and 38 expert users to sketch 136 3D objects, which were presented as 362 images rendered from multiple views. This leads to a new dataset of 3,620 freehand multi-view sketches, which are registered with their corresponding 3D objects under certain views. Our dataset is an order of magnitude larger than the existing datasets. We analyze the collected data at three levels, i.e., sketch-level, stroke-level, and pixel-level, under both spatial and temporal characteristics, and within and across groups of creators. We found that the drawings by professionals and novices show significant differences at stroke-level, both intrinsically and extrinsically. We demonstrate the usefulness of our dataset in two applications: (i) freehand-style sketch synthesis, and (ii) posing it as a potential benchmark for sketch-based 3D reconstruction. 

## Dataset

Coming soon!


## Citation

```tex
@article{xiao2022differsketching,
      title={DifferSketching: How Differently Do People Sketch 3D Objects?}, 
      author={Xiao, Chufeng and Su, Wanchao and Liao, Jing and Lian, Zhouhui and Song, Yi-Zhe and Fu, Hongbo},
      journal = {ACM Transactions on Graphics (Proceedings of ACM SIGGRAPH Asia 2022)},
      volume={41},
      number={4},
      pages={1--16},
      year={2022},
      publisher={ACM New York, NY, USA}
}
```



<center>
    <img src="./index.assets/cityu_logo.jpg" width="55%"><br/>
    <img src="./index.assets/pk_logo.png" width="35%"> &nbsp;&nbsp;&nbsp;
    <img src="./index.assets/surrey_logo2.png" width="33%">
</center>



