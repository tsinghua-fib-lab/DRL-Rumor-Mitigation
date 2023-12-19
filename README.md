# DRL-Rumor-Mitigation

### Overall Framework

![transfer](.\img\frame.png)

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

![transfer](.\img\alg.png)

### Result

![transfer](.\img\table.png)



![transfer](.\img\lines.png)

- We compared DRLE to other baselines at different social media platforms, with the metrics being the Total Infectious Rate.
- Extensive experiments demonstrate that DRLE yields impressive effects on the mitigation of rumors, exhibiting an improvement of **over 20%** compared to baseline methods.

### Transferability 

![transfer](.\img\transfer.png)

- The model trained on small social media platforms can be directly applied to larger networks with only a marginal decrease in metrics. Importantly, this performance remains superior to the optimal baselines.

### For Vulnerable Populations

![transfer](.\img\4good.png)

- DRLE can also offer effective protection for specific populations within social media platforms.


# License
Please see the [license](LICENSE) for further details.

## Note

The implemention is based on *[Transform2Act](https://github.com/Khrylx/Transform2Act)*.
