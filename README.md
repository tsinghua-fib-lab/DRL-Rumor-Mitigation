# DRL-Rumor-Mitigation

### Overall Framework

![image-20231219145858355](C:\Users\randommm\AppData\Roaming\Typora\typora-user-images\image-20231219145858355.png)

# Installation 

### Environment
* **Tested OS: **Linux
* Python >= 3.8
* PyTorch == 1.13.0
* Tensorboard
### Dependencies:
1. Install [PyTorch 1.13.0](https://pytorch.org/get-started/previous-versions/) with the correct CUDA version.
2. Set the following environment variable to avoid problems with multiprocess trajectory sampling:
    ```
    export OMP_NUM_THREADS=1
    ```

# Training

You can train your own models using the provided config in `metro/cfg`:

```
python -m news.train --cfg cfg_name --global_seed 0 --num_threads 1 --gpu_index 2 --agent rl-gnn3 
```
You can replace `cfg_name` to train other cfgs.

The results are saved in path `result/platform/method/cfg/seed`

### Algorithm Framework 

![image-20231219151336377](C:\Users\randommm\AppData\Roaming\Typora\typora-user-images\image-20231219151336377.png)

### Result

![image-20231219151803656](C:\Users\randommm\AppData\Roaming\Typora\typora-user-images\image-20231219151803656.png)



![image-20231219150103216](C:\Users\randommm\AppData\Roaming\Typora\typora-user-images\image-20231219150103216.png)

- We compared DRLE to other baselines at different social media platforms, with the metrics being the Total Infectious Rate.
- Extensive experiments demonstrate that DRLE yields impressive effects on the mitigation of rumors, exhibiting an improvement of **over 20%** compared to baseline methods.

### Transferability 

![image-20231219152907333](C:\Users\randommm\AppData\Roaming\Typora\typora-user-images\image-20231219152907333.png)

- The model trained on small social media platforms can be directly applied to larger networks with only a marginal decrease in metrics. Importantly, this performance remains superior to the optimal baselines.

### For Vulnerable Populations

![image-20231219153348348](C:\Users\randommm\AppData\Roaming\Typora\typora-user-images\image-20231219153348348.png)

- DRLE can also offer effective protection for specific populations within social media platforms.


# License
Please see the [license](LICENSE) for further details.

## Note

The implemention is based on *[Transform2Act](https://github.com/Khrylx/Transform2Act)*.
