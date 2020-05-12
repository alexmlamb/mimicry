# RAVEN

This repo contains code for our CVPR 2019 paper.

[RAVEN: A Dataset for <u>R</u>elational and <u>A</u>nalogical <u>V</u>isual r<u>E</u>aso<u>N</u>ing](http://wellyzhang.github.io/attach/cvpr19zhang.pdf)  


![framework](http://wellyzhang.github.io/img/in-post/RAVEN/process.jpg)

# Dataset

The dataset is generated using the attributed stochastic image grammar. An example is shown below.

![grammar](http://wellyzhang.github.io/img/in-post/RAVEN/prologue.jpg)

The grammatical design makes the dataset flexible and extendable. In total, we come up with 7 different figural configurations. 

![configurations](http://wellyzhang.github.io/img/in-post/RAVEN/peek_view.png)

The dataset formatting document is in ```assets/README.md```. To download the dataset, please check [our project page](http://wellyzhang.github.io/project/raven.html#dataset).

# Performance

We show performance of models in the following table. For details, please check our [paper](http://wellyzhang.github.io/attach/cvpr19zhang.pdf).


| Method     | Acc        | Center     | 2x2Grid    | 3x3Grid    | L-R        | U-D        | O-IC       | O-IG       |
| :---       | :---:      | :---:      | :---:      | :---:      | :---:      | :---:      | :---:      | :---:      |
| LSTM       | 13.07%     | 13.19%     | 14.13%     | 13.69%     | 12.84%     | 12.35%     | 12.15%     | 12.99%     |
| WReN       | 14.69%     | 13.09%     | 28.62%     | 28.27%     | 7.49%      | 6.34%      | 8.38%      | 10.56%     |
| CNN        | 36.97%     | 33.58%     | 30.30%     | 33.53%     | 39.43%     | 41.26%     | 43.20%     | 37.54%     |
| ResNet     | 53.43%     | 52.82%     | 41.86%     | 44.29%     | 58.77%     | 60.16%     | 63.19%     | 53.12%     |
| LSTM+DRT   | 13.96%     | 14.29%     | 15.08%     | 14.09%     | 13.79%     | 13.24%     | 13.99%     | 13.29%     |
| WReN+DRT   | 15.02%     | 15.38%     | 23.26%     | 29.51%     | 6.99%      | 8.43%      | 8.93%      | 12.35%     |
| CNN+DRT    | 39.42%     | 37.30%     | 30.06%     | 34.57%     | 45.49%     | 45.54%     | 45.93%     | 37.54%     |
| ResNet+DRT | **59.56%** | **58.08%** | **46.53%** | **50.40%** | **65.82%** | **67.11%** | **69.09%** | **60.11%** |
| Human      | 84.41%     | 95.45%     | 81.82%     | 79.55%     | 86.36%     | 81.81%     | 86.36%     | 81.81%     |
| Solver     | 100%       | 100%       | 100%       | 100%       | 100%       | 100%       | 100%       | 100%       |




## Dataset Generation

Code to generate the dataset resides in the ```src/dataset``` folder. To generate a dataset, run

```
python src/dataset/main.py --num-samples <number of samples per configuration> --save-dir <directory to save the dataset>
```

Check the ```main.py``` file for a full list of arguments you can adjust.

## Benchmarking

Code to benchmark the dataset resides in ```src/model```. To run the code, first put ```assets/embedding.npy``` in the dataset folder as specified in the ```src/model/utility/dataset_utility.py```. Then run

```
python src/model/main.py --model <model name> --path <path to the dataset>
```

You can check the ```main.py``` file for a full list of arguments. This repo only supports ```Resnet18_MLP```, ```CNN_MLP```, and ```CNN_LSTM```. For WReN, please check the implementation in [the WReN repo](https://github.com/Fen9/WReN).

Note that for batch processing, we implement the DRT as a maximum tree of all possible tree structures and prune the branches during training based on an indicator.

