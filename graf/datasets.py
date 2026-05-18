import os, json, torch, glob, argparse, importlib
import numpy as np
import pandas as pd
from PIL import Image
import os
import torch

from torchvision.datasets.vision import VisionDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt

class ImageDataset(VisionDataset):
    """
    Load images from multiple data directories.
    Folder structure: data_dir/filename.png
    """

    def __init__(self, data_dirs, extractor=None, extractor_args=None, result_dir="results", transforms=None, **kwargs):
        # Use multiple root folders
        if not isinstance(data_dirs, list):
            data_dirs = [data_dirs]

        # initialize base class
        VisionDataset.__init__(self, root=data_dirs, transform=transforms)

        self.filenames = []
        root = []
        self.labels = {}
        self.extractor = extractor
        self.extractor_args = extractor_args
        self.result_dir = result_dir
        self.exp_list = ['RS307', 'RS315', 'RS330','RS615'] #,'RS315'
        self.hysteresis = {}
        self.hidden_state = self.get_hidden_state(extractor, self.exp_list, result_dir=result_dir)

         # ── 新增：每個 specimen 的 7 個正規化材料特徵 ──────────────
        # 欄位順序: displacement, force, corner_σ, corner_ε, core_σ, core_ε, cover_ε
        self.material_features = {
            "RS307_n": [0.000000,0.156717,0.918975,0.339361,0.498170,0.310937,0.360617],
            "RS315_n": [0.006158,0.411209,0.538894,0.725360,0.017321,0.727661,0.7510072],
            "RS330_n": [0.008831,1.000000,1.000000,1.000000,0.000000,1.000000,1.000000],
            "RS615_n": [1.000000,0.000000,0.000000,0.000000,1.000000,0.000000,0.000000],
        }

        self.height_map = {
            "0_": 0,    
            "0.5_": 1,  
            "1_": 2,    
            "1.5_": 3,  
        }

        target_specimens = ["RS307_n", "RS315_n", "RS330_n",'RS615_n'] #, "RS615_n" , "RS315_n"
        self.specimen_props = {
            "RS307_n": {
                "AR": [1, 0],       # 3.0
                "LR": [1, 0, 0],    # 0.742
                "TR": [0, 1]        # 1.282
            },

            "RS315_n": {
                "AR": [1, 0],       # 6.0
                "LR": [0, 1, 0],    # 1.444
                "TR": [1, 0]        # 1.106
            },

            "RS330_n": {
                "AR": [1, 0],       # 3.0
                "LR": [0, 0, 1],    # 2.889
                "TR": [1, 0]        # 1.106
            },
            "RS615_n": {
                "AR": [0, 1],       # 6.0
                "LR": [0, 1, 0],    # 1.444
                "TR": [1, 0]        # 1.106
            }
        }
        for ddir in data_dirs:
            all_files = self._get_files(ddir)
            
            for filename in all_files:
                # [篩選] 只保留 RS307 和 RS330
                specimen_name = None
                for name in target_specimens:
                    if name in filename:
                        specimen_name = name
                        break
                
                if specimen_name is None:
                    continue 

                self.filenames.append(filename)

                props = self.specimen_props[specimen_name]
                vec_AR = props["AR"]
                vec_LR = props["LR"]
                vec_TR = props["TR"]

                # ── 新增：取出該 specimen 的材料特徵 ──
                vec_mat = self.material_features.get(specimen_name, [0.0] * 7)

                for category_prefix, category_idx in self.height_map.items():
                    if filename.startswith(f"{ddir}/{category_prefix}"):
                        num_part = filename.split('_')[-1].replace('.jpg', '')
                        file_idx = int(num_part)
                        angle_idx = [category_idx, file_idx]
                        # ── 修改：label 從 9-dim 擴充到 16-dim ──
                        final_label = vec_AR + vec_LR + vec_TR + vec_mat + angle_idx
                        self.labels[filename] = final_label
                        break 
            root.append(ddir)


    def __len__(self):
        return len(self.filenames)

    @staticmethod
    def _get_files(root_dir):
        return glob.glob(f'{root_dir}/*.png') + glob.glob(f'{root_dir}/*.jpg' )+ glob.glob(f'{root_dir}/*.PNG')

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        # 改為檢查完整的路徑 (filename) 裡面有沒有包含 RS307 等字眼
        exp = next((e for e in self.exp_list if e in filename), None)
        
        # 加上一個安全防護：如果真的找不到，讓它印出路徑方便我們除錯
        if exp is None:
            raise ValueError(f"嚴重錯誤：無法從路徑 '{filename}' 中找到對應的實驗名稱 (預期包含 {self.exp_list})")
            
        img = Image.open(filename).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels.get((filename), 0)
        label = torch.tensor(label, dtype=torch.float32)
        device = img.device
        label = label.to(device)
        hidden_state = self.hidden_state[exp]
        return img, label, hidden_state
    
    def get_hidden_state(self, extractor, exp_list, result_dir):
        hidden_state = {}
        
        test_ds = HystereticDataset(root="/Data/home/vicky/graf260108_im64/Data/Hysteresis", specified=exp_list, **vars(self.extractor_args))
        test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=test_ds.collate_fn)
        device = extractor.device
        
        with torch.no_grad():
            for j, batch in enumerate(tqdm(test_dataloader)):
                simulated_loop, test_force, _, maximun_simulated_force, exp = batch
                simulated_loop = simulated_loop.to(device)
                output, _ = extractor.L(simulated_loop)
                hidden_state[exp[0]] = output[0, -1, :].cpu().squeeze()
                
                predicted_force = extractor.P(output)
                self.draw(predicted_force, simulated_loop, test_force, maximun_simulated_force, exp[0], result_dir)
        
        return hidden_state
    
    def draw(self, predicted_force, simulated_loop, test_force, maximun_simulated_force, exp, result_dir):
        predicted_force = predicted_force.cpu().squeeze().detach().numpy()
        simulated_loop = simulated_loop.cpu().squeeze().detach().numpy()
        drift = simulated_loop[:, -2]
        simulated_force = simulated_loop[:, -1]
        test_force = test_force.cpu().squeeze().detach().numpy()
        max_simulated_force = maximun_simulated_force.cpu().detach().numpy()
        self.hysteresis[exp] = {"predicted": predicted_force, "simulated": simulated_force, "test": test_force, "drift": drift, "max_simulated_force": max_simulated_force}
        
        fig = plt.figure()
        plt.plot(drift, test_force * max_simulated_force, label="Experiment", color="tab:blue", alpha=0.7)
        plt.plot(drift, predicted_force * max_simulated_force, label="Predicted", linestyle="--", color="tab:red", alpha=0.7)
        plt.plot(drift, simulated_force * max_simulated_force, label="Simulated", linestyle="-.", color="tab:green", alpha=0.3)
        plt.xlabel("Drift (%)")
        plt.ylabel("Force (kN)")
        plt.legend()
        plt.title(f"{exp}")
        
        save_dir = os.path.join(result_dir, "data_check")
        os.makedirs(save_dir, exist_ok=True)

        # 移除了 self.mode
        plt.savefig(os.path.join(result_dir, "data_check", f"{exp}.svg"))
        plt.close()

class Carla(ImageDataset):
    def __init__(self, **kwargs):
        super(Carla, self).__init__(**kwargs)
    
class RS307_0_i2(ImageDataset):
    def __init__(self, **kwargs):
        super(RS307_0_i2, self).__init__(**kwargs)

class HystereticDataset(Dataset):
    """Dataset class for the hystereticDataset dataset."""
    
    # 移除了 mode
    def __init__(self, root, data_size="big", specified=[], simulate=True, scale=True, maximum=None, **kwargs):
        self.path = root
        self.simulate = simulate
        self.scale = scale
        self.maximum = maximum
        
        self.hysteresis_loop = {}
        if len(specified) == 0:
            json_data = json.load(open(os.path.join(self.path, f"id_list_{data_size}.json")))
            self.exp_list = []
            # 自動處理：如果 JSON 裡面還是字典(有train/val)，就把所有 list 合併；如果是 list 就直接用
            if isinstance(json_data, dict):
                for v in json_data.values():
                    self.exp_list.extend(v)
            else:
                self.exp_list = json_data
        else:
            self.exp_list = specified
            
        self.columns = ["id", "concreteStrength", "transverse_yield", "transverse_strength", "longitudinal_corner_yield", "longitudinal_corner_strength", "longitudinal_intermediate_yield", 
                    "longitudinal_intermediate_strength", "lMeasured", "axialLoad", "fcc", "epscc", "fcu", "epscu", "Vn", "Vst", "reinforcementRatio", "volTransReinfRatio"]
        self.info = pd.read_csv(os.path.join(self.path, f"info_w_mander_{data_size}.csv")).loc[:, self.columns]
        statistics = pd.read_csv(os.path.join(root, f"statistics_{data_size}.csv"), index_col=0)
        self.info = standardize(self.info, statistics)
        
        for exp in self.exp_list:
            self.hysteresis_loop[exp] = {"simulate": 0, "exp": 0, "info": 0}
            
        for exp in self.exp_list:
            df_test = pd.read_csv(os.path.join(self.path, "Experiment", f"{exp}.csv"))
            df_simulate = pd.read_csv(os.path.join(self.path, "Simulate", f"{exp}.csv"))
            info = self.info[self.info["id"]==exp].iloc[:, 1:].values
            simulation_values = df_simulate.loc[:, ["Drift(%)", "Force(kN)"]].values
            test_values = df_test.loc[:, ["Force(kN)"]].values
            
            self.hysteresis_loop[exp]["simulate"] = simulation_values
            self.hysteresis_loop[exp]["exp"] = test_values
            self.hysteresis_loop[exp]["info"] = info

    def __getitem__(self, index):
        exp = self.exp_list[index]
        simulated_loop = torch.FloatTensor(self.hysteresis_loop[exp]["simulate"])
        test_force = torch.FloatTensor(self.hysteresis_loop[exp]["exp"])
        info = torch.FloatTensor(self.hysteresis_loop[exp]["info"])
        if self.simulate and self.scale:
            maximun_simulated_force = simulated_loop[:, 1].max()
        else:
            maximun_simulated_force = torch.FloatTensor([self.maximum]) 
        
        simulated_loop[:, 1] = simulated_loop[:, 1] / maximun_simulated_force
        if self.simulate:
            input = torch.cat([info.repeat(simulated_loop.shape[0], 1), simulated_loop], dim=1)
        else:
            input = torch.cat([info.repeat(simulated_loop.shape[0], 1), simulated_loop[:, 0:1]], dim=1)
            
        test_force[:, 0] = test_force[:, 0] / maximun_simulated_force
        original_length = simulated_loop.shape[0]
        
        return input, test_force, original_length, maximun_simulated_force, exp
    
    def __len__(self):
        return len(self.hysteresis_loop)

    def collate_fn(self, batch):
        simulated_loop, test_force, original_length, maximun_simulated_force, exp = zip(*batch)
        simulated_loop = torch.nn.utils.rnn.pad_sequence(simulated_loop, batch_first=True, padding_value=0)
        test_force = torch.nn.utils.rnn.pad_sequence(test_force, batch_first=True, padding_value=0)
        original_length = torch.tensor(original_length)
        maximun_simulated_force = torch.stack(maximun_simulated_force)
        exp = list(exp)
        return simulated_loop, test_force, original_length, maximun_simulated_force, exp
    
class PatternLoopDataset(Dataset):
    
    # 移除了 mode
    def __init__(self, root, transform, extractor, extractor_args, result_dir, data_path, column_info, **kwargs):
        super().__init__()
        self.root = root
        self.transform = transform
        self.extractor_args = extractor_args
        
        # 修改 JSON 讀取邏輯
        json_data = json.load(open(os.path.join(self.root, "DamagePattern", data_path)))
        self.image_list = []
        if isinstance(json_data, dict):
            for v in json_data.values():
                self.image_list.extend(v)
        else:
            self.image_list = json_data
            
        self.label = pd.read_csv(os.path.join(self.root, "DamagePattern", "label.csv"))
        self.label = self.label[self.label["filename"].isin(self.image_list)]
        self.exp_list = self.label['exp'].unique()
        print("Extracting hidden state...")
        self.info = pd.read_csv(os.path.join(self.root, "Hysteresis", f"info_w_mander_big.csv")).loc[:, column_info]
        statistics = pd.read_csv(os.path.join(self.root, "Hysteresis", f"statistics_big.csv"), index_col=0)
        self.info = standardize(self.info, statistics)
        self.hysteresis = {}
        self.hidden_state = self.get_hidden_state(extractor, self.exp_list, result_dir=result_dir)
        print("Done!")
        
    def __getitem__(self, index):
        filename, exp, direction, timestep, DI = self.label.iloc[index, :].values
        image = Image.open(os.path.join(self.root, "DamagePattern", "images", filename))
        info = self.info[self.info["id"]==exp].iloc[:, 1:].values.squeeze()
        info = torch.FloatTensor(info)
        hidden_state = self.hidden_state[exp][timestep, :]
        image = self.transform(image)
        DI = torch.FloatTensor([DI])
        if direction == "Pull":
            direction = torch.FloatTensor([0.0, 1.0])
        else:
            direction = torch.FloatTensor([1.0, 0.0])
        return image, hidden_state, direction, exp, timestep, DI, info
        
    def __len__(self):
        return self.label.shape[0]
    
    def get_hidden_state(self, extractor, exp_list, result_dir):
        hidden_state = {}
        test_ds = HystereticDataset(root=os.path.join(self.root, "Hysteresis"), specified=exp_list, **vars(self.extractor_args))
        test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=test_ds.collate_fn)
        device = extractor.device
        
        with torch.no_grad():
            for j, batch in enumerate(tqdm(test_dataloader)):
                simulated_loop, test_force, _, maximun_simulated_force, exp = batch
                simulated_loop = simulated_loop.to(device)
                output, _ = extractor.L(simulated_loop)
                hidden_state[exp[0]] = output.cpu().squeeze()
                
                predicted_force = extractor.P(output)
                self.draw(predicted_force, simulated_loop, test_force, maximun_simulated_force, exp[0], result_dir)
        
        return hidden_state
    
    def draw(self, predicted_force, simulated_loop, test_force, maximun_simulated_force, exp, result_dir):
        predicted_force = predicted_force.cpu().squeeze().detach().numpy()
        simulated_loop = simulated_loop.cpu().squeeze().detach().numpy()
        drift = simulated_loop[:, -2]
        simulated_force = simulated_loop[:, -1]
        test_force = test_force.cpu().squeeze().detach().numpy()
        max_simulated_force = maximun_simulated_force.cpu().detach().numpy()
        self.hysteresis[exp] = {"predicted": predicted_force, "simulated": simulated_force, "test": test_force, "drift": drift, "max_simulated_force": max_simulated_force}
        
        fig = plt.figure()
        plt.plot(drift, test_force * max_simulated_force, label="Experiment", color="tab:blue", alpha=0.7)
        plt.plot(drift, predicted_force * max_simulated_force, label="Predicted", linestyle="--", color="tab:red", alpha=0.7)
        plt.plot(drift, simulated_force * max_simulated_force, label="Simulated", linestyle="-.", color="tab:green", alpha=0.3)
        plt.xlabel("Drift (%)")
        plt.ylabel("Force (kN)")
        plt.legend()
        plt.title(f"{exp}")
        
        # 移除了 self.mode
        plt.savefig(os.path.join(result_dir, "data_check", f"{exp}.svg"))
        plt.close()

def standardize(df, statistics):
    for column in df.columns:
        if column in statistics.columns:
            df[column] = (df[column] - statistics[column]["mean"]) / (statistics[column]["std"])   
    return df