from django.db import models
from pgvector.django import VectorField
from pgvector.django import IvfflatIndex, HnswIndex
from text_bot.views.managers import TopChatQuestionsManager, CTDocumentSplitManager


# { input: searchQuery, history_key: historyKey }

class TextbotOutput(models.Model):
    history_key = models.CharField(max_length=100)

    class Meta:
        ordering = ['-id']

class UserHistory(models.Model):
    history_key = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    creator = models.ForeignKey('auth.User', related_name='history_keys', on_delete=models.CASCADE)

    class Meta:
        ordering = ['-id']

# multi-qa-distilbert-cos-v1
MULTI_QA_DISTILBERT_COS_V1_VECTOR_SIZE = 1536
class ChatQuestion(models.Model):

    class Meta:
        indexes = [
            IvfflatIndex(
                name='my_index',
                fields=['embedding'],
                lists=100,
                opclasses=['vector_l2_ops']
            )
        ]

    history_key = models.CharField(max_length=100)
    embedding = VectorField(dimensions=MULTI_QA_DISTILBERT_COS_V1_VECTOR_SIZE)
    appearance_count = models.IntegerField()
    text = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)

    objects = TopChatQuestionsManager()

    class Meta:
        ordering = ['-id']

class CTDocument(models.Model):
    class Meta:
        ordering = ['-id']

    document_version = models.IntegerField()
    document_title = models.CharField(max_length=100)
    document_filename = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

class CTDocumentPage(models.Model):
    class Meta:
        ordering = ['-id']

    ct_document = models.ForeignKey(CTDocument, on_delete=models.CASCADE, related_name='document_pages')
    document_page_text = models.CharField(max_length=6000)
    document_page_number = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)



# extract topics, extract keywords, extract possible questions
# Counter Hypothetical Document Embeddings (HyDE)
# CREATE QUESTIONS FOR CONTEXT
class CTDocumentSplit(models.Model):
    class Meta:
        ordering = ['-id']
        indexes = [
            IvfflatIndex(
                name='my_index',
                fields=['embedding'],
                lists=100,
                # probes = 10,
                opclasses=['vector_cosine_ops']
            ),

            # or
            HnswIndex(
                name='my_hnsw_index',
                fields=['embedding'],
                m=16,
                ef_construction=64,
                opclasses=['vector_l2_ops']
            )
        ]

    ct_document = models.ForeignKey(CTDocument, on_delete=models.CASCADE, related_name='document_splits')
    document_title = models.CharField(max_length=100)
    document_filename = models.CharField(max_length=100)
    document_page = models.IntegerField()
    split_text = models.CharField(max_length=1500)
    split_text_compression = models.CharField(max_length=1500)
    split_number = models.IntegerField()
    embedding = VectorField(dimensions=MULTI_QA_DISTILBERT_COS_V1_VECTOR_SIZE)
    created_at = models.DateTimeField(auto_now_add=True)

    objects = CTDocumentSplitManager()


