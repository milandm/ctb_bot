from django.db import models
from sentence_transformers import SentenceTransformer
from pgvector.django import CosineDistance
from text_bot.ai_utils import get_distance_scores, check_distance_scores_out

class TopChatQuestionsManager(models.Manager):

    def add_chat_question_to_top_chat_pool(self, question_text: str, history_key:str):
        # paraphrase-multilingual-mpnet-base-v2 Dimensions:	768
        # multi-qa-distilbert-cos-v1 Dimensions: 768
        # dimension all-MiniLM-L12-v2 Dimensions: 384

        model = SentenceTransformer('multi-qa-distilbert-cos-v1')
        question_embedding = model.encode([question_text], normalize_embeddings=True).tolist()[0]

        # add_new_anchor_db(dataset_groups[16]['text'], groups_sentences_embeddings[16])
        sentence_rows_list = self.query_embedding_in_db(question_embedding)
        sentence_to_update_list = list()
        is_similar_to_existing = False
        if sentence_rows_list:
            for sentence in sentence_rows_list:
                distance_scores = get_distance_scores(question_embedding, sentence.embedding)
                if not check_distance_scores_out(distance_scores):
                    is_similar_to_existing = True
                    sentence_to_update_list.append({"sentence": sentence, "distance_scores": distance_scores})

        if is_similar_to_existing:
            if sentence_to_update_list:
                sentence_to_update = \
                max(sentence_to_update_list, key=lambda x: x["distance_scores"]['cosine_scores'])["sentence"]
                self.update_group_count_db(sentence_to_update)
        else:
            self.add_new_anchor_db(question_text, question_embedding, history_key)

    def add_new_anchor_db(self, question_text, question_embedding, history_key):
        self.create(history_key = history_key,
                    text=question_text,
                    embedding=question_embedding,
                    appearance_count=1)

    def query_embedding_in_db(self, embedding):
        return self.order_by(L2Distance('embedding', embedding))[:5]

    def update_group_count_db(self, chat_question):
        chat_question.appearance_count = chat_question.appearance_count+1
        chat_question.save()

class UserHistoryManager(models.Manager):

    def get_user_history(self, user_id):
        return self.filter(creator=user_id)


# COSINE_DISTANCE_TRESHOLD = 0.15
COSINE_DISTANCE_TRESHOLD = 0.2

class CTDocumentSplitManager(models.Manager):



    # def add_document_embedding_db(self, document_title, document_text, document_embedding):
    #     self.create(document_title = document_title,
    #                 text=document_text,
    #                 embedding=document_embedding)

    def query_embedding_in_db(self, embedding):
        return self.order_by(CosineDistance('embedding', embedding)).all()[:5]

    def query_embedding_and_filter_out_in_db(self, document_title, embedding):
        items = self.filter(document_title=document_title).order_by(CosineDistance('embedding', embedding))[:5]
        return items

    def query_embedding_by_distance(self, embedding):
        return self.alias(distance=CosineDistance('embedding', embedding)).filter(distance__lt=COSINE_DISTANCE_TRESHOLD).order_by('distance')
