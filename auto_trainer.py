import os

BASE_PATH = "/jsphilip54/healthcare_baseline"
CONFIG_PATH = os.path.join(BASE_PATH, "configs")
EXPERIMENT_PATH = os.path.join(BASE_PATH, "outputs")

CONFIG_QUEUE_PATH = os.path.join(CONFIG_PATH, "queue")
CONFIG_ENDS_PATH = os.path.join(CONFIG_PATH, "ends")

configs = [file for file in os.listdir(CONFIG_QUEUE_PATH) if file != ".gitkeep"]
configs.sort()

if configs:
    print(f"current config file: {configs[0]}")

    # train.py
    os.system(f"nohup python train.py --config {os.path.join(CONFIG_QUEUE_PATH, configs[0])} &")
    os.system(f"sleep 60")
    os.system(f"mv {os.path.join(CONFIG_QUEUE_PATH, configs[0])} {os.path.join(CONFIG_ENDS_PATH, configs[0])}")

    # predict.py
    # import json
    # with open(os.path.join(CONFIG_ENDS_PATH, configs[0]), "r") as exp:
    #     config = json.load(exp)
    # os.system(f"poetry run python inference.py --exp {os.path.join(EXPERIMENT_PATH, config['name'])}")