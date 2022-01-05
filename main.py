import fit_model
import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 사용 가능한 gpu가 있으면 사용
    print(f'Using {device} device')  # 현재 사용하고 있는 device 출력

    fit_model.fit_model(device)
