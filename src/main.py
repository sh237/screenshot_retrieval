# Latest Update : 31 May 2022, 09:55 GMT+7

# TO ADD :
# Gradient Checkpointing
# Filter out bias from weight decay
# Decaying learning rate with cosine schedule
# Half-precision Adam statistics
# Half-precision stochastically rounded text encoder weights were used

#BATCH_SIZE must larger than 1

from os import path
import os
import time
from random import sample
import random
import glob
import re
import torch
import clip
import json
import sys
import numpy as np
from tqdm import tqdm
import wandb
import argparse
import collections
import pandas as pd

from PIL import Image

from model import ScreenShotModel
from data_loader import ScreenShotDataset
from model import Embedding
import callback_server

import warnings
warnings.simplefilter('ignore')



def calc_score(probs, index):
    mrr = 0
    recall1 = 0
    recall5 = 0
    recall10 = 0
    recall20 = 0
    # probsを大きい順にソートし、ソートする前のindexの順位を取得.重複がある場合には最大の順位を取得
    #ソートする前のprobsのindexの値が、ソート後に何番目に来るかを取得
    rank = sorted(probs[0], reverse=True).index(probs[0, index])
    
    # find top 20
    top20 = np.argsort(probs)[-20:][::-1]
    
    # for i, rank in enumerate(sorted(ranks)):
    if rank < 20:
        recall20 += 1
    if rank < 10:
        recall10 += 1
    if rank < 5:
        recall5 += 1
    if rank < 1:
        recall1 += 1

    # if i == 0: # first time
    mrr = 100 / (rank+1) 

    recall20 = 100 * recall20 
    recall10 = 100 * recall10
    recall5 = 100 * recall5
    recall1 = 100 * recall1
    
    return mrr, recall1, recall5, recall10, recall20, rank, top20

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 


def train_epoch(model, dataloader, optimizer):
    model.train()
    t_loss = 0
    n_ex = 0
    # Train
    for document, predicted_label, label in tqdm(dataloader):
        # print("data :", data.shape, predicted_label.shape, label.shape)
        optimizer.zero_grad()
        loss = model(document.to("cuda:0"), predicted_label.to("cuda:0"), label.to("cuda:0"))
        t_loss += loss
        loss.backward()
        optimizer.step()
        n_ex += 1

    return loss/n_ex


@torch.no_grad()
def eval_epoch_loss(model, dataloader, optimizer, frcnn_flag=False):
    model.eval()
    v_loss = 0
    n_ex = 0
    # Train
    # for images, texts, frcnn in tqdm(dataloader):
    for bbox_image_feature, entire_image_feature, tokenized_instruction, tokenized_np, left_image_feature, right_image_feature in tqdm(dataloader):
        optimizer.zero_grad()
        loss = model(bbox_image_feature.to("cuda:0"), entire_image_feature.to("cuda:0"), tokenized_instruction.to("cuda:0"),
                        tokenized_np.to("cuda:0"), left_image_feature.to("cuda:0"), right_image_feature.to("cuda:0"))
        v_loss += loss
        n_ex += 1

    return loss/n_ex

@torch.no_grad()
def eval(model, dataloader, split="val"):
    
    model.eval()
    info_path = "../data/output/{}_info.json".format(split)
    with torch.no_grad():
        mrr, recall1, recall5, recall10, recall20 = 0,0,0,0,0
        print(f"==========  {split.upper()}  ===========")

        env_mrr, env_recall1, env_recall5, env_recall10, env_recall20 = 0,0,0,0,0

        n_ex = 0
        for index, (document, predicted_label, label) in enumerate(tqdm(dataloader)):
            
            document = torch.unique(document, dim=1)
            predicted_label = torch.unique(predicted_label, dim=1)
            logits_per_text, _, _ = model.calc_logits(document.squeeze(0).to("cuda"), predicted_label.squeeze(0).to("cuda"), label.to("cuda"))
            # logits_per_textは[1, 8]のtensor.これを[8]に変換し、リストに変換する
            logits_per_text = logits_per_text
            # _mrr, _recall1, _recall5, _recall10, _recall20, ranks = calc_score(probs, gt_img_id, imageId_list, bbox=args.bbox)
            gt_doc_id = get_doc_id(index, info_path)
            _mrr, _recall1, _recall5, _recall10, _recall20, rank, top20 = calc_score(logits_per_text.cpu().numpy(), gt_doc_id)
            
            n_ex += 1

            env_mrr += _mrr
            env_recall1 += _recall1
            env_recall5 += _recall5
            env_recall10 += _recall10
            env_recall20 += _recall20
        
            
        mrr += env_mrr / n_ex
        recall1 += env_recall1 / n_ex
        recall5 += env_recall5 / n_ex
        recall10 += env_recall10 / n_ex
        recall20 += env_recall20 / n_ex

    return mrr, recall1, recall5, recall10, recall20

def get_doc_id(index, info_path):
    with open(info_path, "r") as f:
        info = json.load(f)
    return int(info["data"][index]["document_id"]) - 1

def sample_eval(model, preprocess_clip):
    image = preprocess_clip(Image.open("CLIP.png")).unsqueeze(0).to("cuda:0")
    text = clip.tokenize(["a diagram", "a dog", "a cat"]).to("cuda:0")

    with torch.no_grad():
        print(image.shape, text.shape)
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    

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

# データローダーのサブプロセスの乱数seedが固定
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

@torch.no_grad()
def embed_document(data_path, embedding, model, embed=True):
    """data_pathにあるcleaned_transcript.txtとpredicted_label.txt
    読み込み、それらを用いてembeddingを計算し、embedding.pthとして保存"""
    if embed:
        predicted_label_path = os.path.join(data_path, "predicted_label.txt")
        cleaned_transcript_path = os.path.join(data_path, "cleaned_transcription.txt")
        with open(predicted_label_path, "r", encoding="utf-8") as file:
            predicted_label = file.read().split("\n")
        with open(cleaned_transcript_path, "r", encoding="utf-8") as file:
            cleaned_content = file.read()
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
            # print("inner embedding_plabel", embedding_plabel.shape)
            if embedding_plabel.shape[0] < 10:
                embedding_plabel = torch.cat([embedding_plabel, torch.zeros(10-embedding_plabel.shape[0], 768)], dim=0)
            elif embedding_plabel.shape[0] > 10:
                embedding_plabel = embedding_plabel[:10]
            emb_predicted_label.append(embedding_plabel)
        if len(emb_predicted_label) < 10:
            for i in range(10-len(emb_predicted_label)):
                emb_predicted_label.append(torch.zeros(10, 768))
        elif len(emb_predicted_label) > 10:
            emb_predicted_label = emb_predicted_label[:10]
        emb_predicted_label = torch.stack(emb_predicted_label, dim=0)
        model.to("cuda")
        
        with torch.no_grad():
            docment_embedding = model.document_encoder(emb_cleaned_content.to("cuda"), emb_predicted_label.to("cuda"))

        torch.save(docment_embedding, os.path.join(data_path, "embedding.pth"))

def embed_documents(embedding, model):
    data_paths = glob.glob("../data/sample/*")
    data_paths.sort()
    for data_path in data_paths:
        print("data_path", data_path)
        embed_document(data_path, embedding, model, args.embed)

def load_document_embeddings():
    """document特徴量を読み込む"""
    document_embeddings = []
    data_paths = glob.glob("../data/sample/*")
    data_paths.sort()
    for data_path in data_paths:
        document_embeddings.append(torch.load(os.path.join(data_path, "embedding.pth")))
    return document_embeddings

def main(args):
    SEED = int(args.seed)
    np.random.seed(SEED)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    fix_seed(SEED)
    BATCH_SIZE = 30

    # if args.log_wandb:
    #     wandb.init(project="clip-reverie", name=args.wandb_name)

    model = ScreenShotModel().to("cuda:0")
    if args.server:
        """serverとして立ち上げるときの処理"""
        #モデルを読み込む
        model.load_state_dict(torch.load("./model/0.pth"))
        model.eval()
        if args.overwrite_embedding:
            print("embedding documents ...")
            embedding  = Embedding()
            embed_documents(embedding, model)
        
        document_embeddings = load_document_embeddings()
        print("image length", len(document_embeddings))
        #サーバーを立ち上げる
        with open("../config/server_config.json", "r", encoding="utf-8") as server_conf:
            conf = json.load(server_conf)
        callback_server.start(conf, document_embeddings, model.predict_oneshot)
    
    else:
        print("Currently loading train dataset ... ")
        train_X = pd.read_pickle("../data/output/train_X.pkl")
        val_X = pd.read_pickle("../data/output/val_X.pkl")
        test_X = pd.read_pickle("../data/output/test_X.pkl")
        train_y = pd.read_pickle("../data/output/train_y.pkl")
        val_y = pd.read_pickle("../data/output/val_y.pkl")
        test_y = pd.read_pickle("../data/output/test_y.pkl")

        train_dataset = ScreenShotDataset(train_X, train_y)
        val_dataset = ScreenShotDataset(val_X, val_y, eval=True)
        test_dataset = ScreenShotDataset(test_X, test_y, eval=True)
        # train_dataloader = DataLoader(dataset, batch_size=int(args.bs)) 
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=BATCH_SIZE, 
                                                shuffle=True,  # データシャッフル
                                                num_workers=0,  
                                                pin_memory=True,  
                                                worker_init_fn=worker_init_fn
                                                )
        val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=1,  
                                                shuffle=False,  
                                                num_workers=0, 
                                                pin_memory=True,
                                                worker_init_fn=worker_init_fn
                                                )
        test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=0,
                                                pin_memory=True,
                                                worker_init_fn=worker_init_fn
                                                )
        optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), betas=(0.9,0.98), eps=1e-6, weight_decay=0.2) #Params from paper

        best_score = 0
        best_epoch = 0
        for epoch in range(int(args.epochs)):
            print(f"==== Epoch {epoch} ================")
            loss = train_epoch(model, train_dataloader, optimizer)
            print(f"Epoch: {epoch},  Loss: {loss:.10f}")

            # val_unseen_mrr, val_unseen_recall1, val_unseen_recall5, val_unseen_recall10, val_unseen_recall20 = eval(model, dataloader=val_dataloader, split="val")
            # print(f"{val_unseen_mrr:.2f}, {val_unseen_recall1:.2f}, {val_unseen_recall5:.2f}, {val_unseen_recall10:.2f}, {val_unseen_recall20:.2f}")  

            test_mrr, test_recall1, test_recall5, test_recall10, test_recall20 = eval(model, dataloader=test_dataloader, split="test")
            print(f"{test_mrr:.2f}, {test_recall1:.2f}, {test_recall5:.2f}, {test_recall10:.2f}, {test_recall20:.2f}")

            eval_score = test_mrr

            if eval_score > best_score:
                best_epoch = epoch
                best_score = eval_score
                #前のモデルを削除する
                if os.path.exists(os.path.join(args.model_output_path, "best_model.pth")):
                    os.remove(os.path.join(args.model_output_path, "best_model.pth"))
                torch.save(model.state_dict(), os.path.join(args.model_output_path, "best_model.pth"))
            if os.path.exists(os.path.join(args.model_output_path, f"{epoch}.pth")):
                os.remove(os.path.join(args.model_output_path, f"{epoch}.pth"))
            torch.save(model.state_dict(), os.path.join(args.model_output_path, f"{epoch}.pth"))
        # if args.log_wandb:
        #     wandb.log({"mrr":test_mrr, "R@1":test_recall1, "R@5":test_recall5, "R@10":test_recall10, "R@20":test_recall20})

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--lr", default="1e-5")
    parser.add_argument("--bs", default=128)
    parser.add_argument("--epochs", default=50) 
    parser.add_argument("--loss_weight", default=2) 
    parser.add_argument("--model_output_path", default="./model")
    parser.add_argument("--seed", default="42")
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--overwrite_embedding", action="store_true")

    args = parser.parse_args()
    main(args)