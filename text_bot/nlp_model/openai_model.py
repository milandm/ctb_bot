import openai

from typing import Union, Generator, Any
from text_bot.nlp_model.nlp_model import NlpModel

from text_bot.nlp_model.config import (
    OPENAI_API_KEY,
)

from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI


SYSTEM_MSG = 'Ti si ekspert za zakone u oblasti klinickih istrazivanja.'
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-3.5-turbo"

class OpenaiModel(NlpModel):

    VECTOR_PARAMS_SIZE = 1536

    def __init__(self):
        openai.api_key = OPENAI_API_KEY
        self.llm = ChatOpenAI(temperature=0, model_name=LLM_MODEL)
        self.open_ai_embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    def send_prompt( self, system_msg:str, user_prompt:str ):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system_msg},
                       {"role": "user", "content": user_prompt}],
            max_tokens=250,
            temperature=0,
        )
        return response

    def get_embedding(self, text):
        return self.open_ai_embeddings.embed_query(text)


    def get_embeddings(self, sentences: list[str]) -> list[list[float]]:
        return self.open_ai_embeddings.embed_documents(sentences)









    # def send_prompt( self, user_prompt:str ) -> Union[Generator[Union[list, openai.OpenAIObject, dict], Any, None], list, OpenAIObject, dict]:
    #     response = openai.ChatCompletion.create(
    #         model="gpt-3.5-turbo",
    #         messages=[{"role": "system", "content": SYSTEM_MSG},
    #                    {"role": "user", "content": user_prompt}],
    #         max_tokens=250,
    #         temperature=0.2,
    #     )
    #     return response
    #
    #     # # Define the system message
    #     # system_msg = 'You are a helpful assistant who understands data science.'
    #     #
    #     # # Define the user message
    #     # user_msg = 'Create a small dataset about total sales over the last year. The format of the dataset should be a data frame with 12 rows and 2 columns. The columns should be called "month" and "total_sales_usd". The "month" column should contain the shortened forms of month names from "Jan" to "Dec". The "total_sales_usd" column should contain random numeric values taken from a normal distribution with mean 100000 and standard deviation 5000. Provide Python code to generate the dataset, then provide the output in the format of a markdown table.'
    #     #
    #     # # Create a dataset using GPT
    #     # response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
    #     #                                         messages=[{"role": "system", "content": system_msg},
    #     #                                                   {"role": "user", "content": user_msg}])
    #
    #     # return {
    #     #     "response": response["choices"][0]["message"]["content"],
    #     #     "references": references,
    #     # }
    #
    # def get_embedding(self, text) -> Union[Generator[Union[list, openai.OpenAIObject, dict], Any, None], list, openai.OpenAIObject, dict]:
    #     # text = text.replace("\n", " ")
    #
    #     if isinstance(text, str):
    #         text = [text]
    #     embedding = openai.Embedding.create(input=text, model=EMBEDDING_MODEL)
    #
    #     embeddings = [row["embedding"]
    #         for row in embedding['data']
    #     ]
    #
    #     if len(embeddings) == 1:
    #         return embeddings[0]
    #
    #     return embeddings
    #
    #     # df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
    #     # df.to_csv('output/embedded_1k_reviews.csv', index=False)
    #
    #     # import pandas as pd
    #     #
    #     # df = pd.read_csv('output/embedded_1k_reviews.csv')
    #     # df['ada_embedding'] = df.ada_embedding.apply(eval).apply(np.array)
    #
    # def get_embeddings(self, sentences: list[str]) -> ndarray:
    #     vectors = []
    #     batch_size = 512
    #     batch = []
    #
    #     for doc in tqdm(sentences):
    #         batch.append(doc)
    #
    #         if len(batch) >= batch_size:
    #             vectors.append(self.get_embedding(batch))
    #             batch = []
    #
    #     if len(batch) > 0:
    #         vectors.append(self.get_embedding(batch))
    #         batch = []
    #
    #     vectors = np.concatenate(vectors)
    #
    #     return vectors