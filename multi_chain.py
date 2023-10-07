from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain import PromptTemplate
import time
from langchain.chains import LLMChain
from langchain.chains import SequentialChain


# please put your openai api key here
#for best practice put it on environment variable 
openai_api_key = "your openai api key"

llm = ChatOpenAI(temperature=0,
                 openai_api_key=openai_api_key,
                 max_tokens=1000,
                 model='gpt-3.5-turbo'
                 )

question_prompt = """
You are a question generator .Given the sentence , it is your job to create questions from that sentence,and the question should be {hardness_level} level. 
sentence : {sentence}
generator : This is your questions for the above sentence:
"""

answer_prompt = """
You are a answer generator .Given the questions , it is your job to create answers from that questions from the sourse{source}. 
questions : {questions}
generator : This is your answers for the above questions:
"""

question_prompt_template = PromptTemplate(template=question_prompt, input_variables=["sentence",'hardness_level'])

Question_chain = LLMChain(llm=llm,prompt=question_prompt_template,output_key='questions')

answer_prompt_template = PromptTemplate(input_variables=["questions","source"], template=answer_prompt)
answer_chain = LLMChain(llm=llm, prompt=answer_prompt_template, output_key="answer")


overall_chain = SequentialChain(
    chains=[Question_chain, answer_chain],
    input_variables=["sentence","hardness_level","source"],
    output_variables=["questions", "answer"],
    verbose=True)


def get_question_answer(filepath,start_page = 1,end_page = 0,hardness_level = 'easy or simple'):
   
   
    loader = PyPDFLoader(filepath)
    pages = loader.load()
    if end_page == 0:
        end_page = len(pages)

    print(f"number of pages in pdf file :{len(pages)}")

    if start_page > len(pages):
         return False,False
    
    if end_page > len(pages):
         return True,True

    limit = end_page

    text = ""
    index = 1
    for page in pages: 
            if index <= limit:
                if index >= start_page: 
                    text += page.page_content
                index += 1

    text = text.replace('\t', ' ')

    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=7000, chunk_overlap=1500)
    docs = text_splitter.create_documents([text])

    ques_list = []
    ans_list = []
    for i,doc in enumerate(docs):
        result = overall_chain({'sentence' : doc,'hardness_level' : hardness_level,'source' : [doc]})

        ques_list.append(result['questions'])
        ans_list.append(result['answer'])

        print(f"---------------{i}---------------")

    return ques_list,ans_list

def split_sentence(questions):
    question = []
    for q1 in questions:
        sentences = q1.split('\n')
        for q in sentences:
            question.append(q)
    return question
     


if __name__ == "__main__":
    st = time.time()

    questions,answers = get_question_answer("pdf/quiz.pdf",start_page=1,end_page=2,hardness_level="easy")

    if questions == False:
        print("start page should be less then or equal to the number of pages on pdf.")
    
    elif questions == True:
        print("end page should be less than or equal to the number of pages on pdf.")
    
    else:       
        questions_list = split_sentence(questions)
        answers_list = split_sentence(answers)
        print(len(questions_list))
        print(len(answers_list))
        q_n_a = []
        for q, a in zip(questions_list, answers_list):
            q_n_a.append({"question": q, "answer": a})
        print(q_n_a)     
    print(f'total time taken is : {time.time() - st}')