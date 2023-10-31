from string import Template


# system messages describe the behavior of the AI assistant. A useful system message for data science use cases is "You are a helpful assistant who understands data science."
# user messages describe what you want the AI assistant to say. We'll cover examples of user messages throughout this tutorial
# assistant messages describe previous responses in the conversation. We'll cover how to have an interactive conversation in later tasks


question_template = """
QUESTION: {question}
=========
=========
ANSWER:

"""


combine_template = """
PREVIOUS:
{previous}

Da li si siguran da ANSWER sadrzi sve informacije koje se pominju u dokumentaciji vezano za QUESTION.
Kompletan odgovor treba da bude u dole zadatom formatu:

```
QUESTION: <the question>
=========
<Source of information 1>
...
<Source of information N>
=========
ANSWER: <you provide your answer here. Always use bullet points.>

SOURCES: <list the sources used from those provided above>
```

"""


synopsis_template = """

Ti si ekspert za zakone u oblasti klinickih istrazivanja.
Tvoj zadatak je da pruzis informacije iz datih izvora.
Treba da navedes dokument u kom si pronasao odgovor.
Odgovor treba da bude u dole zadatom formatu:

```
QUESTION: <the question>
=========
<Source of information 1>
...
<Source of information N>
=========
ANSWER: <you provide your answer here. Always use bullet points.>

SOURCES: <list the sources used from those provided above>
```

QUESTION: {question}
=========
{summaries}
=========
ANSWER:"""


combine_template = """
PREVIOUS:
{previous}

Da li si siguran da PREVIOUS sadrzi sve informacije koje se pominju u dokumentaciji vezano za QUESTION.
Kompletan odgovor treba da bude u dole zadatom formatu:

```
QUESTION: <the question>
=========
<Source of information 1>
...
<Source of information N>
=========
{summaries}
=========
ANSWER: <you provide your answer here. Always use bullet points.>

SOURCES: <list the sources used from those provided above>
```

"""






QUESTION_PROMPT_TEMPLATE = """
        You propose closest meaning sentences : $question

        Cite them in your answer.

        References:

        $references

        \nHow to cite a reference: This is a citation [1]. This one too [3]. And this is sentence with many citations [2][3].\nAnswer:
        """

RECOMMEND_PROMPT_TEMPLATE = """
        You propose closest meaning sentences : $questions

        Cite them in your answer.

        References:

        $references

        \nHow to cite a reference: This is a citation [1]. This one too [3]. And this is sentence with many citations [2][3].\nAnswer:
        """




SYSTEM_MSG_EXPERT = """

Ti si ekspert za zakone u oblasti klinickih istrazivanja.
Tvoj zadatak je da pruzis informacije iz datih izvora.
Treba da navedes dokument u kom si pronasao odgovor.
Odgovor treba da bude u dole zadatom formatu:

```
RESULT: {
    "question": <the question>,
    "answer": <you provide your answer here. Always use bullet points.>,
    "sources": [
        <list the sources used from those provided above>
        <Source of information 1>,
        ...
        <Source of information N>
    ]}
```
"""


SYSTEM_MSG_TITLE = """
Izdvoj naslov iz zadatog teksta.
Kompletan odgovor treba da bude u dole zadatom formatu:

```
TITLE: <title you extracted>
```
"""

TITLE_EXTRACT_KEY = "TITLE:"


TITLE_TEMPLATE = """

DOCUMENT_SPLIT: $document_split

Izdvoj naslov iz DOCUMENT_SPLIT teksta
"""




DOCUMENT_SYSTEM_MSG_COMPRESSION_V1 = """
Compress the following text in a way that fits in a tweet - 280 characters (ideally)
and such that you (GPT-4) can reconstruct the intention of the human 
who wrote text as close as possible to the original intention. 
This is for yourself. It does not need to be human readable or understandable. 
Abuse of language mixing, abbreviations, symbols (unicode and emoji), 
or any other encodings or internal representations is all permissible, 
as long as it, if pasted in a new inference cycle, 
will yield near-identical results as the original text: 

Complete answer should be formatted this way:

```
TEXT_COMPRESSION: <text you compressed>
```
"""


DOCUMENT_SYSTEM_MSG_COMPRESSION_V2 = """
Compress the given text following rules specified below sorted by priority:
    1. Mandatory keep all enlisted items!!!
    2. Highest priority is to preserve all key information and entities in the text.
    3. Very high priority is to compress the following text in a way that you (GPT-4) 
    can reconstruct the intention of the human who wrote text as close as possible to the original intention. 
    4. If it is possible to keep all key information and entities it is preferable that compressed text fits 
    in a tweet(280) characters.
    If it is not possible to keep all key information and entities it is preferable that compressed text fits 
    in a tweet(280) characters, compress the given text in more then 280 characters.

    5. This is for yourself. 
    It does not need to be human readable or understandable. 
    Abuse of language mixing, abbreviations, symbols (unicode and emoji), 
    or any other encodings or internal representations is all permissible, 
    as long as it, if pasted in a new inference cycle, 
    will yield near-identical results as the original text. 

Complete answer should be formatted this way:

```
TEXT_COMPRESSION: <text you compressed>
```
"""


DOCUMENT_SYSTEM_MSG_COMPRESSION_V2 = """
Compress the given text following rules specified below sorted by priority:
    1. It is mandatory to keep all enlisted items!!!
    2. Highest priority is to preserve all key information and entities in the text.
    3. Very high priority is to compress the following text in a way that you (GPT-4) 
    can reconstruct the intention of the human who wrote text as close as possible to the original intention. 
    4. Compress text size to as much as possible low count of characters

    5. This is for yourself. 
    It does not need to be human readable or understandable. 
    Abuse of language mixing, abbreviations, symbols (unicode and emoji), 
    or any other encodings or internal representations is all permissible, 
    as long as it, if pasted in a new inference cycle, 
    will yield near-identical results as the original text. 

Complete answer should be formatted this way:

```
TEXT_COMPRESSION: <text you compressed>
```
"""


DOCUMENT_SYSTEM_MSG_COMPRESSION_V3 = """
Compress the given text following rules specified below sorted by priority:
    1. It is mandatory to keep all enlisted items!!!
    2. Highest priority is to preserve all key information and entities in the text.
    3. Very high priority is to compress the following text in a way that you (GPT-4) 
    can reconstruct the intention of the human who wrote text as close as possible to the original intention. 
    4. Compress text size to as much as possible low count of characters

    5. This is for yourself. 
    It does not need to be human readable or understandable. 
    Abuse of language mixing, abbreviations, symbols (unicode and emoji), 
    or any other encodings or internal representations is all permissible, 
    as long as it, if pasted in a new inference cycle, 
    will yield near-identical results as the original text. 

Complete answer should be formatted this way:

```
TEXT_COMPRESSION: <text you compressed>
```
"""


DOCUMENT_COMPRESSION_EXTRACT_KEY = "TEXT_COMPRESSION:"


DOCUMENT_COMPRESSION_TEMPLATE_V1 = """
This is text that should be compressed: 
$text_to_compress
"""

DOCUMENT_COMPRESSION_TEMPLATE_V2 = """
This is the given text that should be compressed: 
$text_to_compress
"""





DOCUMENT_SYSTEM_MSG_COMPRESSION_CHECK_V1 = """
You are expert for clinical trial research and you should check if given response is correct.
"""

DOCUMENT_COMPRESSION_CHECK_TEMPLATE_V1 = """

GIVEN_REQUEST:

    ```
    Compress the given text following rules specified below sorted by priority:
        1. It is mandatory to keep all enlisted items!!!
        2. Highest priority is to preserve all key information and entities in the text.
        3. Very high priority is to compress the following text in a way that you (GPT-4) 
        can reconstruct the intention of the human who wrote text as close as possible to the original intention. 
        4. Compress text size to as much as possible low count of characters
    
        5. This is for yourself. 
        It does not need to be human readable or understandable. 
        Abuse of language mixing, abbreviations, symbols (unicode and emoji), 
        or any other encodings or internal representations is all permissible, 
        as long as it, if pasted in a new inference cycle, 
        will yield near-identical results as the original text. 
    
        This is the given text that should be compressed: 
        $text_to_compress
    
    ```

PREVIOUS_RESPONSE: $previous_response
    
    Please check if PREVIOUS_RESPONSE for GIVEN_REQUEST is complete and correct.
    If PREVIOUS_RESPONSE is complete and correct, its MANDATORY!!! that new response should be just "YES".
    If PREVIOUS_RESPONSE is not complete or not correct, please provide as short as possible comment on PREVIOUS_RESPONSE,
    and complete and correct new response.
    
    Your new response should be formatted this way:

```
{
    "comment": <comment>,
    "new_response": <new response>
}
```
    
"""



# For this specific domain CRI chatGPT cant do reconstruction of high extent of comression in proper way.
# Domain fine-tuning is needed
BETTER_COMPRESSION_TEMPLATE = """

The PREVIOUS_RESPONSE is readable, well-formatted, and maintains the essence of the GIVEN_REQUEST. However, the goal was to compress it further using any permissible encodings or representations that I can later reconstruct for a new inference cycle.

Given this, I will compress the text even further.



NEW_RESPONSE: UZpkQI📋. OpšKI:👤sponz,🔀CRO,📖stud,🔢protokol,📊faza,💊lek,🏢ZU&👥istraž. TokKI:📅kvart,👥iskr&rndm,🚫ispitKI,🔒bez(🚫🏥RS&🔗IMP). 18.07.17.
"""




# Rewrite-Retrieve-Read


template = """Answer the users question based only on the following context:

<context>
{context}
</context>

Question: {question}
"""



template = """Provide a better search query for \
web search engine to answer the given question, end \
the queries with ’**’. Question: \
{x} Answer:"""





# Semi-structured RAG
#
# Many documents contain a mixture of content types, including text and tables.


template = """Provide a better search query for \
web search engine to answer the given question, end \
the queries with ’**’. Question: \
{x} Answer:"""


# table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})


# RAG Fusion
#
# vectorstore = Pinecone.from_existing_index("rag-fusion", OpenAIEmbeddings())
# retriever = vectorstore.as_retriever()
#
# from langchain.load import dumps, loads
#
#
# def reciprocal_rank_fusion(results: list[list], k=60):
#     fused_scores = {}
#     for docs in results:
#         # Assumes the docs are returned in sorted order of relevance
#         for rank, doc in enumerate(docs):
#             doc_str = dumps(doc)
#             if doc_str not in fused_scores:
#                 fused_scores[doc_str] = 0
#             previous_score = fused_scores[doc_str]
#             fused_scores[doc_str] += 1 / (rank + k)
#
#     reranked_results = [(loads(doc), score) for doc, score in
#                         sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
#     return reranked_results




# Rules Based Checker
# from langchain_experimental.tot.checker import ToTChecker
# from langchain_experimental.tot.thought import ThoughtValidity

# Step-Back Prompting (Question-Answering)

response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

{normal_context}

Original Question: {question}
Answer:"""

from langchain.chains.qa_with_sources.refine_prompts import DEFAULT_REFINE_PROMPT_TMPL


# # Few Shot Examples
# examples = [
#     {
#         "input": "Could the members of The Police perform lawful arrests?",
#         "output": "what can the members of The Police do?"
#     },
#     {
#         "input": "Jan Sindel’s was born in what country?",
#         "output": "what is Jan Sindel’s personal history?"
#     },
# ]
# # We now transform these to example messages
# example_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("human", "{input}"),
#         ("ai", "{output}"),
#     ]
# )
# few_shot_prompt = FewShotChatMessagePromptTemplate(
#     example_prompt=example_prompt,
#     examples=examples,
# )

# Hypothetical Document Embeddings (HyDE)

prompt_template = """Please answer the user's question about the most recent state of the union address
Question: {question}
Answer:"""


# Learned Prompt Variable Injection via RL


# Summarization checker chain

# EmbeddingsRedundantFilter


# k: Optional[int] = 20
# """The number of relevant documents to return. Can be set to None, in which case
# `similarity_threshold` must be specified. Defaults to 20."""
# similarity_threshold: Optional[float]
# """Threshold for determining when two documents are similar enough
# to be considered redundant. Defaults to None, must be specified if `k` is set
# to None."""


# similarity_fn: Callable = cosine_similarity
# """Similarity function for comparing documents. Function expected to take as input
# two matrices (List[List[float]]) and return a matrix of scores where higher values
# indicate greater similarity."""
# similarity_threshold: float = 0.95
# """Threshold for determining when two documents are similar enough
# to be considered redundant."""


# Counter Hypothetical Document Embeddings (HyDE)
# CREATE QUESTIONS FOR CONTEXT

# langchain_experimental.smart_llm import SmartLLMChain

class PromptTemplateCreator:


    def __init__(self):
        print()

    # def create_similar_sentences_prompt(self, question:str, references_list: list[ScoredPoint]) -> tuple[str, str]:
    #
    #     references_text = ""
    #
    #     for i, reference in enumerate(references_list, start=1):
    #         text = reference.payload["text"].strip()
    #         references_text += f"\n[{i}]: {text}"
    #
    #     key_value_to_change ={
    #         "question": question.strip(),
    #         "references": references_text,
    #     }
    #
    #     prompt = self.prepare_template(QUESTION_PROMPT_TEMPLATE, key_value_to_change)
    #
    #     return prompt, references_text

    #
    # def create_recommended_sentences_prompt(self, questions_list:str, references_list: list[ScoredPoint]) -> tuple[str, str]:
    #
    #     questions_text = ""
    #
    #     for i, question in enumerate(questions_list, start=1):
    #         text = question.payload["question"].strip()
    #         questions_text += f"\n[{i}]: {text}"
    #
    #     references_text = ""
    #
    #     for i, reference in enumerate(references_list, start=1):
    #         text = reference.payload["text"].strip()
    #         references_text += f"\n[{i}]: {text}"
    #
    #     key_value_to_change ={
    #         "questions": questions_text,
    #         "references": references_text,
    #     }
    #
    #     prompt = self.prepare_template(RECOMMEND_PROMPT_TEMPLATE, key_value_to_change)
    #
    #     return prompt, references_text


    def prepare_template(self, template: str, **kwargs) -> str:
        prompt_template = Template(template)
        try:
            prepared_prompt = prompt_template.safe_substitute(kwargs)
        except KeyError as e:
            print(e)
        except ValueError as e:
            print(e)

        # mapping = defaultdict(str, key_value_to_change)
        # prepared_prompt = template.format_map(mapping=mapping)
        return prepared_prompt


    def get_title_extract_prompt(self, document_split: str) -> str:
        user_prompt = self.prepare_template(TITLE_TEMPLATE, document_split=document_split)
        return user_prompt

    def get_query_based_text_compression_prompt(self, query: str, document_split: str) -> str:
        user_prompt = self.prepare_template(QUERY_BASED_COMPRESSION_TEMPLATE_V2, query = query, text_to_compress=document_split)
        return user_prompt

    def get_document_text_compression_prompt(self, document_split: str) -> str:
        user_prompt = self.prepare_template(DOCUMENT_COMPRESSION_TEMPLATE_V2, text_to_compress=document_split)
        return user_prompt

    def get_document_text_compression_check_prompt(self, document_split: str, previous_response: str) -> str:
        user_prompt = self.prepare_template(DOCUMENT_COMPRESSION_CHECK_TEMPLATE_V1,
                                            text_to_compress=document_split,
                                            previous_response = previous_response)
        return user_prompt