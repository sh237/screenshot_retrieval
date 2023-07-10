import os
import glob
import pickle
import json
import torch
from transformers import AutoModel, AutoTokenizer

class Embedding():
    """deberta v3のモデルを用いてテキストの埋め込み表現を取得するクラス"""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.deberta_tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")
        self.deberta_model = AutoModel.from_pretrained("microsoft/mdeberta-v3-base")
        for param in self.deberta_model.parameters():
            param.requires_grad = False
        self.deberta_model.eval()
        self.deberta_model.to(self.device)

    def tokenize(self, text):
        return self.deberta_tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)
    
    def get_embedding(self, text):
        input_ids = self.tokenize(text)
        output = self.deberta_model(input_ids)
        return output[0]

def get_subfolders(folder_path):
    subfolders = []
    for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            subfolder_path = os.path.join(root, dir)
            subfolders.append(subfolder_path)
    return subfolders

# アノテーションのラベルのリストを6:2:2に分割 (train, val, testに分割するため)
def split_list(input_list, ratio= [9, 1]):
    """input_listをratioに従って2つに分割する"""
    split_index = int(len(input_list) * ratio[0] / sum(ratio))
    return input_list[:split_index], input_list[split_index:]

def main():
    train_X = []
    val_X = []
    test_X = []
    train_y = []
    val_y = []
    test_y = []
    folder_path = "./sample"  
    output_path = "./output"

    train_info = {"data": []}
    val_info = {"data": []}
    test_info = {"data": []}
    if os.path.exists(output_path) == False:
        os.mkdir(output_path)
    subfolders_list = get_subfolders(folder_path)
    subfolders_list.sort()
    count = 0
    embedding = Embedding()
    train_i, val_i, test_i = 0, 0, 0
    train_subfolders_list, test_subfolders_list = split_list(subfolders_list, ratio=[9, 1])

    # train, valデータの作成
    for doc_id, subfolder in enumerate(train_subfolders_list):
        cleaned_txt_path = glob.glob(os.path.join(subfolder, "cleaned_transcription.txt"))
        predicted_label_path = glob.glob(os.path.join(subfolder, "predicted_label.txt"))
        if len(cleaned_txt_path) != 1:
            assert False, "txt_path must be one"
        if len(predicted_label_path) != 1:
            assert False, "predicted_label_path must be one"
        with open(cleaned_txt_path[0], "r", encoding="utf-8") as file:
            cleaned_content = file.read()
        # predicted_labelは改行区切りの文字列.これを配列に変換
        with open(predicted_label_path[0], "r", encoding="utf-8") as file:
            predicted_label = file.read().split("\n")
        emb_cleaned_content = embedding.get_embedding(cleaned_content).squeeze(0).to("cpu")
        #emb_cleaned_contentの次元は(768*2, 768)とする.そうでない場合は0埋めする
        if emb_cleaned_content.shape[0] < 768*2:
            emb_cleaned_content = torch.cat([emb_cleaned_content, torch.zeros(768*2-emb_cleaned_content.shape[0], 768)], dim=0)
        elif emb_cleaned_content.shape[0] > 768*2:
            emb_cleaned_content = emb_cleaned_content[:768*2]
        emb_predicted_label = []
        for label in predicted_label:
            #emb_predicted_labelの各要素の次元は(10, 768)とする.そうでない場合は0埋めする
            embedding_plabel = embedding.get_embedding(label).squeeze(0).to("cpu")
            print("inner embedding_plabel", embedding_plabel.shape)
            if embedding_plabel.shape[0] < 10:
                embedding_plabel = torch.cat([embedding_plabel, torch.zeros(10-embedding_plabel.shape[0], 768)], dim=0)
            elif embedding_plabel.shape[0] > 10:
                embedding_plabel = embedding_plabel[:10]
            emb_predicted_label.append(embedding_plabel)
        #emb_predicted_labelは(10, 768)のtensorを要素とする配列である.この配列の要素数を10個とする
        if len(emb_predicted_label) < 10:
            for i in range(10-len(emb_predicted_label)):
                emb_predicted_label.append(torch.zeros(10, 768))
        elif len(emb_predicted_label) > 10:
            emb_predicted_label = emb_predicted_label[:10]
        emb_predicted_label = torch.stack(emb_predicted_label, dim=0)
        with open(f"{subfolder}/labels.txt", "r", encoding="utf-8") as file:
            labels = file.read().split("\n")

        num_labels = len(labels)
        train_labels, val_labels = split_list(labels)
        #emb_train_labelsのshapeは例えば(4, 3, 768)のようになっており、この4つをそれぞれ配列に格納する
        for i, label in enumerate(train_labels):
            emb_label = embedding.get_embedding(label).squeeze(0).to("cpu")
            if emb_label.shape[0] < 10:
                emb_label = torch.cat([emb_label, torch.zeros(10-emb_label.shape[0], 768)], dim=0)
            elif emb_label.shape[0] > 10:
                emb_label = emb_label[:10]
            train_y.append(emb_label)
            train_X.append([emb_cleaned_content, emb_predicted_label])
            train_info["data"].append({"document_id" : doc_id + 1, "id" : train_i + 1, "label_id" : i + 1, "label" : label })
            train_i += 1
        for i, label in enumerate(val_labels):
            emb_label = embedding.get_embedding(label).squeeze(0).to("cpu")
            if emb_label.shape[0] < 10:
                emb_label = torch.cat([emb_label, torch.zeros(10-emb_label.shape[0], 768)], dim=0)
            elif emb_label.shape[0] > 10:
                emb_label = emb_label[:10]
            val_y.append(emb_label)
            val_X.append([emb_cleaned_content, emb_predicted_label])
            val_info["data"].append({"document_id" : doc_id + 1, "id" : val_i + 1, "label_id" : i + 1, "label" : label })
            val_i += 1

    # testデータの作成
    for doc_id, subfolder in enumerate(test_subfolders_list):
        cleaned_txt_path = glob.glob(os.path.join(subfolder, "cleaned_transcription.txt"))
        predicted_label_path = glob.glob(os.path.join(subfolder, "predicted_label.txt"))
        if len(cleaned_txt_path) != 1:
            assert False, "txt_path must be one"
        if len(predicted_label_path) != 1:
            assert False, "predicted_label_path must be one"
        with open(cleaned_txt_path[0], "r", encoding="utf-8") as file:
            cleaned_content = file.read()
        # predicted_labelは改行区切りの文字列.これを配列に変換
        with open(predicted_label_path[0], "r", encoding="utf-8") as file:
            predicted_label = file.read().split("\n")
        emb_cleaned_content = embedding.get_embedding(cleaned_content).squeeze(0).to("cpu")
        #emb_cleaned_contentの次元は(768*2, 768)とする.そうでない場合は0埋めする
        if emb_cleaned_content.shape[0] < 768*2:
            emb_cleaned_content = torch.cat([emb_cleaned_content, torch.zeros(768*2-emb_cleaned_content.shape[0], 768)], dim=0)
        elif emb_cleaned_content.shape[0] > 768*2:
            emb_cleaned_content = emb_cleaned_content[:768*2]
        emb_predicted_label = []
        for label in predicted_label:
            #emb_predicted_labelの各要素の次元は(10, 768)とする.そうでない場合は0埋めする
            embedding_plabel = embedding.get_embedding(label).squeeze(0).to("cpu")
            print("inner embedding_plabel", embedding_plabel.shape)
            if embedding_plabel.shape[0] < 10:
                embedding_plabel = torch.cat([embedding_plabel, torch.zeros(10-embedding_plabel.shape[0], 768)], dim=0)
            elif embedding_plabel.shape[0] > 10:
                embedding_plabel = embedding_plabel[:10]
            emb_predicted_label.append(embedding_plabel)
        #emb_predicted_labelは(10, 768)のtensorを要素とする配列である.この配列の要素数を10個とする
        if len(emb_predicted_label) < 10:
            for i in range(10-len(emb_predicted_label)):
                emb_predicted_label.append(torch.zeros(10, 768))
        elif len(emb_predicted_label) > 10:
            emb_predicted_label = emb_predicted_label[:10]
        emb_predicted_label = torch.stack(emb_predicted_label, dim=0)

        with open(f"{subfolder}/labels.txt", "r", encoding="utf-8") as file:
            labels = file.read().split("\n")
        for i, label in enumerate(labels):
            emb_label = embedding.get_embedding(label).squeeze(0).to("cpu")
            if emb_label.shape[0] < 10:
                emb_label = torch.cat([emb_label, torch.zeros(10-emb_label.shape[0], 768)], dim=0)
            elif emb_label.shape[0] > 10:
                emb_label = emb_label[:10]
            test_y.append(emb_label)
            test_X.append([emb_cleaned_content, emb_predicted_label])
            test_info["data"].append({"document_id" : doc_id + 1, "id" : test_i + 1, "label_id" : i + 1, "label" : label, "document_id_fixed" : len(train_subfolders_list) + doc_id + 1})
            test_i += 1

    test = """コンテナのログを出力する
    Docker Hubに公開されているDockerイメージを検索する
    Docker Hubからイメージをダウンロード"""
    test = """私は昨日公園で犬を散歩させた"""
    print("len test", len(test))
    my_embedding = Embedding()
    print("tokenize", my_embedding.tokenize(test).shape)
    print("get_embedding", my_embedding.get_embedding(test).shape)

    # pickleで保存
    with open(os.path.join(output_path, "train_X.pkl"), "wb") as file:
        pickle.dump(train_X, file)
    with open(os.path.join(output_path, "val_X.pkl"), "wb") as file:
        pickle.dump(val_X, file)
    with open(os.path.join(output_path, "test_X.pkl"), "wb") as file:
        pickle.dump(test_X, file)
    with open(os.path.join(output_path, "train_y.pkl"), "wb") as file:
        pickle.dump(train_y, file)
    with open(os.path.join(output_path, "val_y.pkl"), "wb") as file:
        pickle.dump(val_y, file)
    with open(os.path.join(output_path, "test_y.pkl"), "wb") as file:
        pickle.dump(test_y, file)
    with open(os.path.join(output_path, "train_info.json"), "w") as file:
        json.dump(train_info, file, indent=4, ensure_ascii=False)
    with open(os.path.join(output_path, "val_info.json"), "w") as file:
        json.dump(val_info, file, indent=4, ensure_ascii=False)
    with open(os.path.join(output_path, "test_info.json"), "w") as file:
        json.dump(test_info, file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()