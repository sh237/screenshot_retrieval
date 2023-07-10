from email.mime import image
from turtle import left
import time
import torch
import clip
import torch.nn.functional as F
import math
import copy
from torch.nn import CosineSimilarity
from transformers import AutoModel, AutoTokenizer
#torchinfo
from torchinfo import summary

class ScreenShotModel(torch.nn.Module):
    """
    Baseline : CLIP ViT-L/14 を用いて評価
    """
    def __init__(self):
        super(ScreenShotModel, self).__init__()
        # self.clip_model, self.preprocess_clip = clip.load("ViT-L/14", device="cuda:0")
        # 最後の2層の重み固定を解除(17)
        # for param in self.deberta_model.encoder.layer[-2:].parameters():
        #     param.requires_grad = True
        
        # for params in self.clip_model.parameters():
        #     params.requires_grad = False
        self.temperature = 1

        self.cossim = CosineSimilarity(dim=1, eps=1e-6)

        # id 5003 txt transformer
        h_q = 4
        d_ff_q = 768*2
        dropout_q = 0.4
        d_model = 768
        N_q = 2
        c = copy.deepcopy
        attn_q = MultiHeadedAttention(h_q, d_model)
        ff_q = PositionwiseFeedForward(d_model, d_ff_q, dropout_q)
        self.q_encoder = Encoder(N_q, EncoderLayer(
            d_model, c(attn_q), c(ff_q), dropout=dropout_q))

        h_llm = 3
        d_ff_llm = 768*2
        dropout_llm = 0.4
        N_llm = 2
        c = copy.deepcopy
        attn_llm = MultiHeadedAttention(h_llm, d_model)
        ff_llm = PositionwiseFeedForward(d_model, d_ff_llm, dropout_llm)
        self.llm_encoder = Encoder(N_llm, EncoderLayer(
            d_model, c(attn_llm), c(ff_llm), dropout=dropout_llm))
        
        h_doc = 3
        d_ff_doc = 768*2
        dropout_doc = 0.4
        N_doc = 2
        c = copy.deepcopy
        attn_doc = MultiHeadedAttention(h_doc, d_model)
        ff_doc = PositionwiseFeedForward(d_model, d_ff_doc, dropout_doc)
        self.doc_encoder = Encoder(N_doc, EncoderLayer(
            d_model, c(attn_doc), c(ff_doc), dropout=dropout_doc))

        h_doc_ca = 3
        d_ff_doc_ca = 768*2
        dropout_doc_ca = 0.4
        N_doc_ca = 2
        c = copy.deepcopy
        attn_doc_ca = MultiHeadedAttention(h_doc_ca, d_model)
        ff_doc_ca = PositionwiseFeedForward(d_model, d_ff_doc_ca, dropout_doc_ca)
        self.encoder_doc_ca = CrossAttentionEncoder(N_doc_ca, CrossAttentionEncoderLayer(
            d_model, c(attn_doc_ca), c(ff_doc_ca), dropout=dropout_doc_ca))
        
        self.fc_doc1  = torch.nn.Linear(768 * 2, 1000)
        self.fc_doc2  = torch.nn.Linear(1000, 768)
        self.fc_q1  = torch.nn.Linear(768, 1000)
        self.fc_q2  = torch.nn.Linear(1000, 768)
        self.fc6  = torch.nn.Linear(768, 512)
        self.fc7  = torch.nn.Linear(512, 768)
        self.bn_doc = torch.nn.BatchNorm1d(1000) 
        self.bn_q = torch.nn.BatchNorm1d(1000) 
        self.bn4 = torch.nn.BatchNorm1d(1024)
        self.dropout = torch.nn.Dropout(0.4)
        self.dropout1 = torch.nn.Dropout(0.6)
        self.relu = torch.nn.ReLU()

    def document_encoder(self, document, predicted_label):
        """documentの次元は[batch, 1536, 768], predicted_labelの次元は[batch, 10, 10, 768]である.
        これらは0埋めされているので、それらを除去し、encodeする.
        """
        # print("document.shape : ", document.shape)
        # print("predicted_label : ", len(predicted_label))
        # print("predicted_label[0].shape : ", predicted_label[0].shape)
        # predicted_labelの次元は[batch, 10, 10, 768]であるが、これを[batch, 100, 768]に変換する
        # print("documnt.shape : ", document.shape)
        # print("predicted_label.shape : ", predicted_label.shape)
        predicted_label = predicted_label.view(predicted_label.shape[0], -1, predicted_label.shape[-1])
        # print("predicted_label.shape : ", predicted_label.shape)
        
        # print("llm_encoder", self.llm_encoder(predicted_label).shape)
        llm_embedding = self.llm_encoder(predicted_label)[:, 0, :]
        # print("llm_embedding.shape : ", llm_embedding.shape)

        # print("document", document.shape)
        # time.sleep(10000)
        document_embedding = self.doc_encoder(document)[:, 0, :]
        # documentとllm_embeddingをconcat
        # print("document_embedding.shape : ", document_embedding.shape)
        # print("llm_embedding.shape : ", llm_embedding.shape)
        embedding = self.encoder_doc_ca(document_embedding.unsqueeze(1), llm_embedding)[:, 0, :].unsqueeze(1).squeeze(1)

        embedding = torch.cat([embedding, llm_embedding], dim=1)
        # document = torch.cat([llm_embedding.unsqueeze(1), document], dim=1)
        # print("document.shape : ", document.shape)
        # print("embedding.shape : ", embedding.shape)
        # print("document_embedding.shape : ", document_embedding.shape)
        # print("document_embedding.shape : ", document_embedding)
        embedding = self.dropout(self.bn_doc(self.relu(self.fc_doc1(embedding))))
        embedding = self.fc_doc2(embedding)

        # print("embedding.shape : ", embedding.shape)

        return embedding
    

    def query_encoder(self, query):
        query_embedding = self.q_encoder(query)[:, 0, :].squeeze(1)
        # print("query_embedding.shape : ", query_embedding)
        # print("query_embedding.shape : ", query_embedding.shape)
        query_embedding = self.dropout(self.bn_q(self.relu(self.fc_q1(query_embedding))))
        query_embedding = self.fc_q2(query_embedding)
        return query_embedding

        # left_image_embeddings = left_image_feature[:, 0, :].unsqueeze(1)
        # center_image_embeddings = image[:, 0, :].unsqueeze(1)
        # right_image_embeddings = right_image_feature[:, 0, :].unsqueeze(1)

        # lr_img = torch.cat([left_image_embeddings, right_image_embeddings], dim=1)
        # image_embeddings = self.encoder_img(lr_img)
        # image_embeddings = image_embeddings[:, 0, :]

        # sam_feature = torch.cat([left_image_feature, image, right_image_feature], dim=1)

        # sam_feature = self.sam_encoder(sam_feature)[:, 0, :]

        # image_embeddings = torch.cat([image_embeddings, bbox_image_feature.squeeze(1), center_image_embeddings.squeeze(1), sam_feature], dim=1)
        # image_embeddings = self.relu(self.fc8(image_embeddings))
        # image_embeddings = self.dropout1(self.bn5(image_embeddings))
        # image_embeddings = self.fc9(image_embeddings)

        
        # return image_embeddings.half()
    
    def calc_logits(self, document, predicted_label, label):
        # print("tokenized_np.shape : ", tokenized_np_clip.shape)
        document_embeddings = self.document_encoder(document, predicted_label) 
        query_embeddings = self.query_encoder(label)
        logits = (query_embeddings @ document_embeddings.T) / self.temperature #[128,128]

        document_similarity = document_embeddings @ document_embeddings.T
        query_similarity = query_embeddings @ query_embeddings.T
        # print("document_similarity.shape : ", document_similarity)
        # print("query_similarity.shape : ", query_similarity)
        return logits, document_similarity, query_similarity
        
    def forward(self, document, predicted_label, label):
        logits, document_similarity, query_similarity = self.calc_logits(document, predicted_label, label)
        # print("logits.shape : ", logits)
        # print("document_similarity.shape : ", document_similarity)
        # print("query_similarity.shape : ", query_similarity)
        # print('(document_similarity + query_similarity) / 2, : ', (document_similarity + query_similarity) / 2)
        targets = F.softmax(
            (document_similarity + query_similarity) / 2 * self.temperature, dim=-1
        )
        # print("targets.shape : ", targets)
        document_loss = cross_entropy(logits, targets)
        # print("document_loss.shape : ", document_loss)
        query_loss = cross_entropy(logits.T, targets.T)
        # print("query_loss.shape : ", query_loss)
        loss =  (document_loss + query_loss) / 2.0 # shape: (batch_size)
        # print("loss.shape : ", loss)
        return loss.mean()
    
    def preprocess(self, x):
        return self.preprocess_clip(x)
    

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


class Encoder(torch.nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, N, layer):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x
    
class CrossAttentionEncoder(torch.nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, N, layer):
        super(CrossAttentionEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, kv):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, kv)
        x = self.norm(x)
        return x


def clones(module, N):
    "Produce N identical layers."
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class SublayerConnection(torch.nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(torch.nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = torch.nn.Parameter(torch.ones(features))
        self.b_2 = torch.nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(torch.nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(d_model, d_ff)
        self.w_2 = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(torch.nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class EncoderLayer(torch.nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)
    
class CrossAttentionEncoderLayer(torch.nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, cross_attn, feed_forward, dropout):
        super(CrossAttentionEncoderLayer, self).__init__()
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, kv):
        "Follow Figure 1 (left) for connections."
        # print("hello")
        x = self.sublayer[0](x, lambda x: self.cross_attn(x, kv, kv))
        return self.sublayer[1](x, self.feed_forward)

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
    
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
        / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
