import subprocess

# 定义要执行的命令列表
commands = [
    'python -m news.train  --cfg twitter_h --global_seed 0 --num_threads 20 --gpu_index 1 --agent rl-gnn3',
    'python -m news.train  --cfg twitter_h --global_seed 0 --num_threads 20 --gpu_index 1 --agent rl-gnn3 --infer 1'
]

# 依次执行命令
for command in commands:
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {e}")
    except Exception as e:
        print(f"发生错误: {e}")

print("所有命令执行完成")
