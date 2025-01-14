import streamlit as st
from streamlit_chat import message
from langchain_core.messages import AIMessage, HumanMessage
from backend.backend import get_text_chunks, create_conversation_chain, create_vectorstore, extract_text_from_pdf
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Sample FAQ Data
faq_data = {
    "What courses are available?": "You can find a list of courses on our university website.",
    "What are the admission requirements?": "Admission requirements vary by course. Check the admissions page.",
    "How can I contact the admissions office?": "You can email admissions@university.edu or call +123456789."
}

# Vectorize FAQ Questions
faq_questions = list(faq_data.keys())
vectorizer = TfidfVectorizer()
faq_embeddings = vectorizer.fit_transform(faq_questions)


# Training data for the classifier
questions = [
    "What courses are available?",
    "What is the fee structure?",
    "How can I apply?",
    "Tell me about research opportunities.",
    "What are the admission requirements?",
    "How can I contact the admissions office?"
]
labels = [1, 0, 0, 0, 1, 1]  # 1 = FAQ, 0 = Non-FAQ

# Train the classifier
question_embeddings = vectorizer.transform(questions)
classifier = LogisticRegression()
classifier.fit(question_embeddings, labels)








# def handle_user_question(user_question):
#     response = st.session_state.conversation({"input": user_question, "chat_history": []})


#     st.session_state.chat_history.extend(
#         [
#             HumanMessage(content=response['input']),
#             AIMessage(content=response['answer']),
#         ]
#     )


#     # print('st.session_state.chat_history', st.session_state.chat_history)

#     # Display the chat messages
#     for i, msg in enumerate(st.session_state.chat_history):
#         if i % 2 == 0:
#             message(msg.content, is_user=True, key=str(i) + '_user', avatar_style="initials", seed="Kavita")
#         else:
#             message(msg.content, is_user=False, key=str(i), avatar_style="initials", seed="AI",)



def handle_user_question(user_question):
    # Classify the user's question
    user_question_embedding = vectorizer.transform([user_question])
    is_faq = classifier.predict(user_question_embedding)[0]

    if is_faq:
        # FAQ handling
        similarities = cosine_similarity(user_question_embedding, faq_embeddings).flatten()


    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user', avatar_style="initials", seed="Kavita")
        else:
            message(msg.content, is_user=False, key=str(i), avatar_style="initials", seed="AI",)



def main():

    st.set_page_config(page_title='ChatBot', page_icon=':books:')

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header('Ask questions about Ulster University :books:')

    user_question = st.chat_input(placeholder='Ask me about ulster university')
    if user_question and user_question != "":
        with st.spinner('Thinking...'):
            handle_user_question(user_question)

    if st.session_state.chat_history:
        clear_history = st.button('Clear Chat History') # clear chat history button
        if clear_history:
            st.session_state.chat_history = []
            # st.session_state.conversation = None
            st.success('Chat history cleared')

    with st.sidebar:
        st.subheader('Your documents')
        pdf_docs = st.file_uploader('Upload a PDF and click process', accept_multiple_files=True)
        if st.button('Process'):
            with st.spinner('Processing document... This may take a moment.'):
            
                # get pdf text
                pdf_text = extract_text_from_pdf(pdf_docs) 

                # get the text chunks
                text_chunks = get_text_chunks(pdf_text)
                # st.write(text_chunks)

                # create vector store
                vectorstore = create_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = create_conversation_chain(vectorstore)


            st.success('Processed successfully')



if __name__ == '__main__':
    main()