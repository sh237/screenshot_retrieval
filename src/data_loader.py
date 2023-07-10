import torch
import numpy as np
import random
import pandas as pd
import torch.multiprocessing as mp
BATCH_SIZE = 1

# 乱数seedの固定
def fix_seed(seed):
    # random
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


fix_seed(seed=42)

# データローダーのサブプロセスの乱数seedが固定
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
# データセットの作成
class ScreenShotDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, eval=False):
        self.X = X
        self.y = y
        self.eval = eval
        if eval:
            self.X = [self.X] * len(self.X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # print("getitem", self.X[index])
        
        # self.X[index]はdatasetsizeの長さのリストで、その各要素は2つの要素を持つリストで、その各要素はtensorになっている.
        # それらを一つ目のtensorと二つ目のtensorに分ける
        label =  self.y[index].squeeze()

        if self.eval:
            cleaned_content = self.X[index][0][0].unsqueeze(0)
            predicted_label = self.X[index][0][1].unsqueeze(0)
            for i in range(1, len(self.X[index])):
                cleaned_content = torch.cat([cleaned_content, self.X[index][i][0].unsqueeze(0)], dim=0)
                predicted_label = torch.cat([predicted_label, self.X[index][i][1].unsqueeze(0)], dim=0)
        else:
            cleaned_content = self.X[index][0]
            predicted_label = self.X[index][1]
        # cleaned_content = self.X[index][0]
        # predicted_label = self.X[index][1]
        # label = self.y[index].squeeze()
        """
        前処理を書く
        現状はcleaned_contentには整形済みの文字列が入っている
        """
        return cleaned_content, predicted_label, label
    
def main():
    # mp.set_start_method('spawn')
    train_X = pd.read_pickle("./data/output/train_X.pkl")
    val_X = pd.read_pickle("./data/output/val_X.pkl")
    test_X = pd.read_pickle("./data/output/test_X.pkl")
    train_y = pd.read_pickle("./data/output/train_y.pkl")
    val_y = pd.read_pickle("./data/output/val_y.pkl")
    test_y = pd.read_pickle("./data/output/test_y.pkl")

    train_dataset = ScreenShotDataset(train_X, train_y)
    val_dataset = ScreenShotDataset(val_X, val_y)
    test_dataset = ScreenShotDataset(test_X, test_y)


    # データローダーの作成
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=BATCH_SIZE, 
                                            shuffle=True,  # データシャッフル
                                            num_workers=0,  
                                            pin_memory=True,  
                                            worker_init_fn=worker_init_fn
                                            )
    val_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=BATCH_SIZE,  
                                            shuffle=True,  
                                            num_workers=0, 
                                            pin_memory=True,
                                            worker_init_fn=worker_init_fn
                                            )
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=False,
                                            num_workers=0,
                                            pin_memory=True,
                                            worker_init_fn=worker_init_fn
                                            )

    # 動作確認
    count = 0
    for data, predicted_label, label in train_loader:
        print("count", count)
        print("data", data.shape, "predicted_label", len(predicted_label), predicted_label[0].shape, "label", label.shape)
        count += 1

if __name__ == "__main__":
    main()