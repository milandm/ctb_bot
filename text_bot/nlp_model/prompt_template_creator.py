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



title_template = """

DOCUMENT_SPLIT: $document_split

Izdvoj naslov iz DOCUMENT_SPLIT teksta
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
        user_prompt = self.prepare_template(title_template, document_split=document_split)
        return user_prompt