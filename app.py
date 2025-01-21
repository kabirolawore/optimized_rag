import streamlit as st
from streamlit_chat import message
from langchain_core.messages import AIMessage, HumanMessage
from backend.backend import get_text_chunks, create_conversation_chain, create_vectorstore, extract_text_from_pdf
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd




# Load the Excel file into a DataFrame
df = pd.read_excel('ques_dup.xlsx')

# Extract features
questions = df['questions']
answers = df['answers']  # Ensure the answers column is used
lengths = df['lengths']
labels = df['labels']

# Vectorize questions
vectorizer = TfidfVectorizer()
question_embeddings = vectorizer.fit_transform(questions)

# Combine question embeddings with lengths
lengths = lengths.values.reshape(-1, 1)
combined_features = np.hstack((question_embeddings.toarray(), lengths))

# Train the classifier
classifier = LogisticRegression()
classifier.fit(combined_features, labels)

def handle_user_question(user_question):
    # Classify the user's question
    user_question_embedding = vectorizer.transform([user_question])
    user_question_length = np.array([[len(user_question)]])
    user_combined_features = np.hstack((user_question_embedding.toarray(), user_question_length))
    
    is_faq = classifier.predict(user_combined_features)[0]

    if is_faq:
        # FAQ handling
        similarities = cosine_similarity(user_question_embedding, question_embeddings).flatten()
        best_match_index = np.argmax(similarities)
        best_match_score = similarities[best_match_index]

        if best_match_score > 0.8:
            response_text = answers.iloc[best_match_index]  # Fetch the corresponding answer
            print(response_text)
        else:
            response = st.session_state.conversation({"input": user_question, "chat_history": []})
            response_text = response['answer']

    else:
        # Non-FAQ handling using RAG pipeline
        response = st.session_state.conversation({"input": user_question, "chat_history": []})
        response_text = response['answer']
        
    # Update chat history for Ques and Ans
    st.session_state.chat_history.extend(
        [
            HumanMessage(content=user_question),
            AIMessage(content=response_text),
        ]
    )

    print('st.session_state.chat_history', st.session_state.chat_history)

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