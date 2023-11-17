from django.db import models
from pgvector.django import VectorField
from pgvector.django import IvfflatIndex
# from pgvector.django import HnswIndex

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

    # find all main topics/contexts
    # find all splits with main context explained
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
                name='document_ivf_flat_index',
                fields=['embedding'],
                lists=100,
                # probes = 10,
                opclasses=['vector_cosine_ops']
            ),

            # or
            # HnswIndex(
            #     name='document_hnsw_index',
            #     fields=['embedding'],
            #     m=16,
            #     ef_construction=64,
            #     opclasses=['vector_l2_ops']
            # )
        ]

    # find split with main context
    # find semantically connected splits
    # is this split related to previous split context
    # is next split related to this slit content
    # what is this split related to in bigger context
    # pick up all splits related with bigger context

    # main context split
    # semantically connected splits

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


# extract topics, extract keywords, extract possible questions
# Counter Hypothetical Document Embeddings (HyDE)
# CREATE QUESTIONS FOR CONTEXT
class CTDocumentSection(models.Model):
    class Meta:
        ordering = ['-id']
        indexes = [
            IvfflatIndex(
                name='document_section_ivf_flat_index',
                fields=['title_embedding',
                        'text_embedding',
                        'content_summary_embedding',
                        'references_embedding',
                        'topics_embedding'],
                lists=100,
                # probes = 10,
                opclasses=['vector_cosine_ops',
                           'vector_cosine_ops',
                           'vector_cosine_ops',
                           'vector_cosine_ops',
                           'vector_cosine_ops']
            ),

            # or
            # HnswIndex(
            #     name='document_section_hnsw_index',
            #     fields=['title_embedding',
            #             'text_embedding',
            #             'content_summary_embedding',
            #             'references_embedding',
            #             'topics_embedding'],
            #     m=16,
            #     ef_construction=64,
            #     opclasses=['vector_l2_ops',
            #                'vector_l2_ops',
            #                'vector_l2_ops',
            #                'vector_l2_ops',
            #                'vector_l2_ops']
            # )
        ]

    # find split with main context
    # find semantically connected splits
    # is this split related to previous split context
    # is next split related to this slit content
    # what is this split related to in bigger context
    # pick up all splits related with bigger context

    # main context split
    # semantically connected splits

    ct_document = models.ForeignKey(CTDocument, on_delete=models.CASCADE, related_name='document_sections')
    document_title = models.CharField(max_length=100)
    document_filename = models.CharField(max_length=100)
    document_page = models.IntegerField()

    section_title = models.CharField(max_length=100)
    section_text = models.CharField(max_length=1500)
    section_content_summary = models.CharField(max_length=1500)
    section_references = models.CharField(max_length=1500)
    section_topics = models.CharField(max_length=1500)
    section_number = models.IntegerField()

    # 'title_embedding', 'text_embedding', 'content_summary_embedding', 'references_embedding', 'topics_embedding'

    title_embedding = VectorField(dimensions=MULTI_QA_DISTILBERT_COS_V1_VECTOR_SIZE)
    text_embedding = VectorField(dimensions=MULTI_QA_DISTILBERT_COS_V1_VECTOR_SIZE)
    content_summary_embedding = VectorField(dimensions=MULTI_QA_DISTILBERT_COS_V1_VECTOR_SIZE)
    references_embedding = VectorField(dimensions=MULTI_QA_DISTILBERT_COS_V1_VECTOR_SIZE)
    topics_embedding = VectorField(dimensions=MULTI_QA_DISTILBERT_COS_V1_VECTOR_SIZE)

    created_at = models.DateTimeField(auto_now_add=True)
    objects = CTDocumentSplitManager()


# extract topics, extract keywords, extract possible questions
# Counter Hypothetical Document Embeddings (HyDE)
# CREATE QUESTIONS FOR CONTEXT
class CTDocumentSubsection(models.Model):
    class Meta:
        ordering = ['-id']
        indexes = [
            IvfflatIndex(
                name='document_subsection_ivf_flat_index',
                fields=['title_embedding',
                        'text_embedding',
                        'content_summary_embedding',
                        'references_embedding',
                        'topics_embedding'],
                lists=100,
                # probes = 10,
                opclasses=['vector_cosine_ops',
                           'vector_cosine_ops',
                           'vector_cosine_ops',
                           'vector_cosine_ops',
                           'vector_cosine_ops']
            ),

            # or
            # HnswIndex(
            #     name='document_subsection_hnsw_index',
            #     fields=['title_embedding',
            #             'text_embedding',
            #             'content_summary_embedding',
            #             'references_embedding',
            #             'topics_embedding'],
            #     m=16,
            #     ef_construction=64,
            #     opclasses=['vector_l2_ops',
            #                'vector_l2_ops',
            #                'vector_l2_ops',
            #                'vector_l2_ops',
            #                'vector_l2_ops']
            # )
        ]

    # find split with main context
    # find semantically connected splits
    # is this split related to previous split context
    # is next split related to this slit content
    # what is this split related to in bigger context
    # pick up all splits related with bigger context

    # main context split
    # semantically connected splits

    ct_document_section = models.ForeignKey(CTDocumentSection, on_delete=models.CASCADE, related_name='section_subsections')
    document_title = models.CharField(max_length=100)
    document_filename = models.CharField(max_length=100)
    document_page = models.IntegerField()

    subsection_title = models.CharField(max_length=100)
    subsection_text = models.CharField(max_length=1500)
    subsection_content_summary = models.CharField(max_length=1500)
    subsection_references = models.CharField(max_length=1500)
    subsection_topics = models.CharField(max_length=1500)
    subsection_number = models.IntegerField()


    title_embedding = VectorField(dimensions=MULTI_QA_DISTILBERT_COS_V1_VECTOR_SIZE)
    text_embedding = VectorField(dimensions=MULTI_QA_DISTILBERT_COS_V1_VECTOR_SIZE)
    content_summary_embedding = VectorField(dimensions=MULTI_QA_DISTILBERT_COS_V1_VECTOR_SIZE)
    references_embedding = VectorField(dimensions=MULTI_QA_DISTILBERT_COS_V1_VECTOR_SIZE)
    topics_embedding = VectorField(dimensions=MULTI_QA_DISTILBERT_COS_V1_VECTOR_SIZE)

    created_at = models.DateTimeField(auto_now_add=True)

    objects = CTDocumentSplitManager()

    # {
    #     "Section Title": "I. УВОДНЕ ОДРЕДБЕ",
    #     "Section Content Summary": "Introduction to the regulation specifying the content and labeling of external and internal packaging of medicines, additional labeling, and the content of the medicine instructions.",
    #     "Section Text": "I. УВОДНЕ ОДРЕДБЕ\nСадржина правилника\nЧлан 1.\nОвим правилником прописује се садржај и начин обележавања спољњег и унутрашњег паковања\nлека, додатно обележавање лека, као и садржај упутства за лек.",
    #     "Section References": ["правилник", "лек", "спољње паковање", "унутрашње паковање", "обележавање",
    #                            "упутство за лек"],
    #     "Section Topics": ["Садржина правилника", "обележавање", "упутство за лек"],
    #     "Subsections": [
    #         {
    #             "Subsection Title": "Садржина правилника",
    #             "Subsection Content Summary": "Defines the regulation of the content and labeling of external and internal packaging of medicines, additional labeling, and the content of the medicine instructions.",
    #             "Subsection Text": "Члан 1.\nОвим правилником прописује се садржај и начин обележавања спољњег и унутрашњег паковања\nлека, додатно обележавање лека, као и садржај упутства за лек.",
    #             "Subsection References": ["правилник", "лек", "спољње паковање", "унутрашње паковање", "обележавање",
    #                                       "упутство за лек"],
    #             "Subsection Topics": ["правилник", "обележавање", "упутство за лек"]
    #         }
    #     ]
    # }