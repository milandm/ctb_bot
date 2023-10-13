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




SYSTEM_MSG_COMPRESSION_V1 = """
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


SYSTEM_MSG_COMPRESSION_V2 = """
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


SYSTEM_MSG_COMPRESSION_V2 = """
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


COMPRESSION_EXTRACT_KEY = "TEXT_COMPRESSION:"


COMPRESSION_TEMPLATE_V1 = """
This is text that should be compressed: 
$text_to_compress
"""

COMPRESSION_TEMPLATE_V2 = """
This is the given text that should be compressed: 
$text_to_compress
"""

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


    def get_text_compression_prompt(self, document_split: str) -> str:
        user_prompt = self.prepare_template(COMPRESSION_TEMPLATE_V2, text_to_compress=document_split)
        return user_prompt