# configurations -------------------------------------------------------------------------------
# CACHE_DIR = r'D:\NLP_Project\models'
CACHE_DIR = ''
API_KEY = 'AIzaSyAnVKQx0OY53BX0AFlviPDu7T6OYsFOCc8'

# ----------------------------------------------------------------------------------------------
import torch
from torchinfo import summary
from transformers import AutoModel, AutoTokenizer
from langchain_core.embeddings import Embeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings

class ViEmbeddingModel(Embeddings):
    def __init__(self, model_name, cache_dir, device=None):
        super().__init__()
        self.device = torch.device('cpu' if device is None else 'cuda:0')
        self.model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_name,
            cache_dir=cache_dir
        )
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name,
            cache_dir=cache_dir
        )
    # callable methods --------------------------------------------------
    def __call__(self, texts):
        input_dict = self.tokenizer(
            text=texts, 
            padding=True,
            truncation=True, 
            max_length=self.get_max_seq_len(),
            return_tensors='pt'
        )
        input_dict = {
            name: value.to(self.device) for name, value in input_dict.items()
        }
        return self.model(**input_dict).pooler_output.tolist()

    def embed_documents(self, texts):
        return self(texts)
    
    def embed_query(self, text):
        return self(text)[0]

    # attribute methods --------------------------
    def get_max_seq_len(self):
        return self.model.embeddings.position_embeddings.num_embeddings - 2

    def get_embedding_dim(self):
        return self.model.pooler.dense.out_features

    def summary(self):
        summary(self.model, verbose=True)


# initialize embedding models ----------------------------------------------------------------
biencoder_embedding = ViEmbeddingModel(
    model_name='bkai-foundation-models/vietnamese-bi-encoder',
    cache_dir=CACHE_DIR
)
phobert_embedding = ViEmbeddingModel(
    model_name='VoVanPhuc/sup-SimCSE-VietNamese-phobert-base',
    cache_dir=CACHE_DIR
)
sbert_embedding = ViEmbeddingModel(
    model_name='keepitreal/vietnamese-sbert',
    cache_dir=CACHE_DIR
)
# genai_embedding = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=API_KEY)
# if __name__ == '__main__':
#     embed_model.summary()