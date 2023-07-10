    def document_encoder(self, document, predicted_label):
        predicted_label = torch.cat(predicted_label, dim=1).to("cuda:0")
        llm_embedding = self.llm_encoder(predicted_label)[:, 0, :].squeeze(1)
        document = torch.cat([llm_embedding.unsqueeze(0), document], dim=1)

        document_embedding = self.doc_encoder(document)[:, 0, :].squeeze(1)
        return document_embedding.half()

    def query_encoder(self, query):
        query_embedding = self.q_encoder(query)[:, 0, :].squeeze(1)
        return query_embedding.half()
    
    def calc_logits(self, document, predicted_label, label):
        # print("tokenized_np.shape : ", tokenized_np_clip.shape)
        document_embeddings = self.document_encoder(document, predicted_label) 
        query_embeddings = self.query_encoder(label)
        
        logits = (document_embeddings @ query_embeddings.T) / self.temperature #[128,128]

        document_similarity = document_embeddings @ document_embeddings.T
        query_similarity = query_embeddings @ query_embeddings.T
        # print("document_similarity.shape : ", document_similarity)
        # print("query_similarity.shape : ", query_similarity)
        return logits, document_similarity, query_similarity
        
    def forward(self, document, predicted_label, label):
        logits, document_similarity, query_similarity = self.calc_logits(document, predicted_label, label)
        print("logits.shape : ", logits)
        print("document_similarity.shape : ", document_similarity)
        print("query_similarity.shape : ", query_similarity)
        print('(document_similarity + query_similarity) / 2, : ', (document_similarity + query_similarity) / 2)
        targets = F.softmax(
            (document_similarity + query_similarity) / 2 * self.temperature, dim=-1
        )
        print("targets.shape : ", targets)
        document_loss = cross_entropy(logits, targets)
        print("document_loss.shape : ", document_loss)
        query_loss = cross_entropy(logits.T, targets.T)
        print("query_loss.shape : ", query_loss)
        loss =  (document_loss + query_loss) / 2.0 # shape: (batch_size)
        print("loss.shape : ", loss)
        return loss.mean()
    
    def preprocess(self, x):
        return self.preprocess_clip(x)