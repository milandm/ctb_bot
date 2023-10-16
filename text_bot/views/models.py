from django.db import models
from pgvector.django import VectorField
from pgvector.django import IvfflatIndex
from text_bot.views.managers import TopChatQuestionsManager, DocumentSplitManager


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

class DocumentSplit(models.Model):
    class Meta:
        indexes = [
            IvfflatIndex(
                name='my_index',
                fields=['embedding'],
                lists=100,
                opclasses=['vector_l2_ops']
            )
        ]

    filename = models.CharField(max_length=100)
    title = models.CharField(max_length=100)
    embedding = VectorField(dimensions=MULTI_QA_DISTILBERT_COS_V1_VECTOR_SIZE)
    text = models.CharField(max_length=1000)
    text_compression = models.CharField(max_length=1000)
    page = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    objects = DocumentSplitManager()

    class Meta:
        ordering = ['-id']


