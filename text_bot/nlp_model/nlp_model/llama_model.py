from llama_index import MockEmbedding
from llama_index.embeddings.base import BaseEmbedding

from sentence_transformers import SentenceTransformer
import torch

import numpy as np
from numpy import ndarray
from tqdm import tqdm
from text_bot.nlp_model.nlp_model import NlpModel
from typing import Union, Generator, Any
from text_bot.nlp_model.openai_model import OpenaiModel
from llama_cpp import Llama


# Llama 2 models are 4096


llm = Llama(model_path="./models/llama-2-7b-chat.ggmlv3.q2_K.bin",
            n_ctx=1024, n_batch=128, verbose=False)
instruction = input("User: ")
# put together the instruction in the prompt template for Orca models
prompt = f"[INST] <<SYS>><</SYS>>\n{instruction} [/INST]"
output = llm(prompt,temperature  = 0.7, max_tokens=1024, top_k=20, top_p=0.9,
            repeat_penalty=1.15)
res = output['choices'][0]['text'].strip()
print('Llama2-7b: ' + res)


llm = Llama(model_path="./models/orca-mini-3b.ggmlv3.q4_1.bin", n_ctx=512, n_batch=32, verbose=False)
instruction = input("User: ")
# put together the instruction in the prompt template for Orca models
system = 'You are an AI assistant that follows instruction extremely well. Help as much as you can.'
prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"
output = llm(prompt,temperature  = 0.7,max_tokens=512,top_k=20, top_p=0.9,
                    repeat_penalty=1.15)
res = output['choices'][0]['text'].strip()



llm.create_chat_completion(
    messages = [
        {"role": "system", "content": "You are an assistant who perfectly describes images."},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://.../image.png"}},
                {"type" : "text", "text": "Describe this image in detail please."}
            ]
        }
    ]
)



SENTENCE_TRANSFORMER_NLP_MODEL = "msmarco-MiniLM-L-6-v3"
CUDA = "cuda"
MPS = "mps"
CPU = "cpu"

class LLamaModel(NlpModel):

    VECTOR_PARAMS_SIZE = 384

    def __init__(self):



        # self.embedding_model = HuggingFaceEmbedding(
        #     model_name=embedding_hf_model_name,
        #     cache_folder=str(models_cache_path),
        # )

        self.llm_model = Llama(model_path="./models/llama-2-7b-chat.ggmlv3.q2_K.bin",
                    n_ctx=1024, n_batch=128, verbose=False)
        instruction = input("User: ")
        # put together the instruction in the prompt template for Orca models
        prompt = f"[INST] <<SYS>><</SYS>>\n{instruction} [/INST]"
        output = llm(prompt, temperature=0.7, max_tokens=1024, top_k=20, top_p=0.9,
                     repeat_penalty=1.15)


        self.model = SentenceTransformer(
            SENTENCE_TRANSFORMER_NLP_MODEL,
            device=CUDA
            if torch.cuda.is_available()
            else MPS
            if torch.backends.mps.is_available()
            else CPU,
        )
        self.nlp_prompt_model = OpenaiModel()



    def get_embeddings(self, sentences: list[str]) -> ndarray:

        vectors = []
        batch_size = 512
        batch = []

        for doc in tqdm(sentences):
            batch.append(doc)

            if len(batch) >= batch_size:
                vectors.append(self.model.encode(batch))
                batch = []

        if len(batch) > 0:
            vectors.append(self.model.encode(batch))
            batch = []

        vectors = np.concatenate(vectors)

        return vectors

    def get_embedding(self, text:str):
        self.model.encode(text)


    def send_prompt( self, system_msg:str, user_prompt:str ):
        return self.nlp_prompt_model.send_prompt(system_msg, user_prompt)