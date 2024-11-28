import streamlit as st
from llama_cpp import Llama

if 'llm' not in st.session_state:
    st.session_state.llm = Llama.from_pretrained(
        repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
        filename="Llama-3.2-3B-Instruct-Q8_0.gguf",  
        verbose=True,
        n_ctx=32768,
        n_threads=2,
        chat_format="chatml"
    )

# Define the function to get responses from the model
def respond(message, history):
    messages = []

    for user_message, assistant_message in history:
        if user_message:
            messages.append({"role": "user", "content": user_message})
        if assistant_message:
            messages.append({"role": "assistant", "content": assistant_message})

    messages.append({"role": "user", "content": message})

    response = ""
    # Stream the response from the model
    response_stream = st.session_state.llm.create_chat_completion(
        messages=messages,
        stream=True,
        max_tokens=512,  # Use a default value for simplicity
        temperature=0.7,  # Use a default value for simplicity
        top_p=0.95  # Use a default value for simplicity
    )

    # Collect the response chunks
    for chunk in response_stream:
        if len(chunk['choices'][0]["delta"]) != 0 and "content" in chunk['choices'][0]["delta"]:
            response += chunk['choices'][0]["delta"]["content"]

    return response  # Return the full response

# Streamlit UI
st.title("🧠 Chatbot using LLM Llama-3.2 !!!")
st.write("### LLM used is bartowski/Llama-3.2-3B-Instruct-GGUF ")

# User input field
user_message = st.text_area("Your Message:", "")

if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []

# Button to send the message
if st.button("Send"):
    if user_message:  # Check if user has entered a message
        # Get the response from the model
        response = respond(user_message, st.session_state['chat_history'])
        st.write(f"**🧠 Assistant**: {response}")
        
        # Add user message and model response to history
        st.session_state['chat_history'].append((user_message, response))

        # Clear the input field after sending
        user_message = ""  # Reset user_message to clear input

st.write("## Chat History")
for user_msg, assistant_msg in reversed(st.session_state['chat_history']):
    st.write(f"**🧑 User**: {user_msg}")
    st.write(f"**🧠 Assistant**: {assistant_msg}")
    st.write("---")
