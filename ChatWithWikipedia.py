import os
import openai

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.environ['OPENAI_API_KEY']

import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from css import bot_template, user_template, css
import wikipedia
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper


def wiki_to_text(page_title):
    # Get the content from the Wikipedia page
    try:
        page_content = wikipedia.page(page_title)
        return page_content.content, None
    except wikipedia.exceptions.PageError:
        error_msg = f"The page '{page_title}' does not exist on Wikipedia."
    except wikipedia.exceptions.DisambiguationError as e:
        error_msg = f"Multiple pages match '{page_title}'. Options include: {e.options}"
    except wikipedia.exceptions.HTTPTimeoutError:
        error_msg = "The request to Wikipedia timed out."
    except Exception as e:
        error_msg = f"An unexpected error occurred: {e}"

    return None, error_msg



def get_chunk_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    return chunks


def get_vector_store(text_chunks):
    # For OpenAI Embeddings

    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    return vectorstore


def get_conversation_chain(vector_store):
    # OpenAI Model

    llm = ChatOpenAI()

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

    return conversation_chain


def handle_user_input(question):
    response = st.session_state.conversation({'question': question})
    st.session_state.chat_history = response['chat_history']
    chat_history = st.session_state.chat_history

    # Check if the length of chat_history is even
    if len(chat_history) % 2 == 0:
        # Iterate over chat_history by index in steps of 2, starting from the end
        for i in range(len(chat_history) - 2, -1, -2):
            # Display the question (index i) first
            st.write(user_template.replace("{{MSG}}", chat_history[i].content), unsafe_allow_html=True)
            # Then display the answer (index i+1)
            st.write(bot_template.replace("{{MSG}}", chat_history[i + 1].content), unsafe_allow_html=True)


def main():
    # load_dotenv()
    st.set_page_config(page_title='Chat with Wikipedia', page_icon=':books:')

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Check if 'input_text' is in the session state
    if "previous_topic" not in st.session_state:
        st.session_state.previous_topic = ""

    # storing summary in session state as it will disappear otherwise when questions are asked
    if "summary" not in st.session_state:
        st.session_state.summary = ""


    question = ""

    st.header('Chat with Wikipedia :books:')
    current_topic = st.text_input("Enter a topic you want to know more about:")

    wkp = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    # this loop happens only if the topic changes and not with every question
    if current_topic != st.session_state.previous_topic:
        print("creating vector store for: " + current_topic)
        # clearing the previous conversation

        st.session_state.chat_history = None
        st.session_state.chat_history = None
        st.session_state.question_text = ""

        st.session_state.previous_topic = current_topic

        if current_topic.lower() in ["farhat", "farhat fachisa", "pachisa"]:
            st.write("Farhat is the author of this app")
            st.session_state.summary = ""
        else:
            raw_text, error_msg = wiki_to_text(current_topic)
            if error_msg:
                st.write(error_msg)
                st.session_state.summary = ""
            else:
                st.session_state.summary = wkp.run(current_topic)
                text_chunks = get_chunk_text(raw_text)

                # Create Vector Store
                vector_store = get_vector_store(text_chunks)

                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)

    if st.session_state.summary != "":
        expander = st.expander("See summary")
        expander.write(st.session_state.summary)
        question = st.text_input("Ask Questions", key="question_text")

    if question:
        handle_user_input(question)

    st.write(css, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
