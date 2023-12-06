import time

import streamlit as st
import torch

from app_utils import load_big_model
from generate import generate_inference
from utils import set_seed, device, load_tokenizer


def main():

    # Enable CUDA if available and load in tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(device)
    EMPTY_TOKENS = torch.full((1, 1), tokenizer.bos_token_id, dtype=torch.long).to(
        device
    )

    st.title("Scaling Transformers")
    st.subheader("UCLA DSU Project, Fall 2023")
    st.markdown("Daniel Mendelevitch  \n Terry Ming  \n Casey Tattersall  \n Sean Tjoa")

    st.header("What Are Transformers? 🚗🔄🤖")

    header_text = """A transformer is a specific type of neural network that uses a mechanism called self-attention to learn the context (and 
        thus meaning) of sequential data. Transformer-based models can be used in many different domains, such as processing language, predicting 
        the weather, or even generating images.  \n\n You might be familiar with ChatGPT, a Transformer-based model which cost over \$100 million to train.  \n In contrast, we spent \$40*.
        """
    st.markdown(header_text)

    st.header("Let's make some stories! 📖")

    # Input from user
    user_input = st.text_input(
        "Enter your prompt:",
        placeholder="Write a prompt to make a story of your own or leave it empty for a random story!",
    ).strip()

    if st.checkbox("Show Prompting Tips"):
        st.markdown(
            "Our model was trained on the TinyStories dataset, a collection of synthetic short stories generated by GPT-4. These stories only contain words and themes that a typical 3-4 year old would understand."
        )
        st.markdown(
            """
            - Use simple vocabulary - words and themes that would appear in a children's story
            - Avoid using idioms - for example, instead of "hit the gym", say "went to the gym"
            - Include plenty of descriptive adjectives
            - The model often struggles with names - using common names and only including a person's first name can help
            """
        )
    ## Default values for advanced settings
    user_seed = 27  # Remove for random demo
    generation_method = "top-k"
    specified_k = 5
    specified_nucleus = 0.5
    specified_temperature = 0.9
    max_tokens = 400

    if st.checkbox("Show Advanced Settings"):
        user_seed = st.number_input(
            "Randomness Seed:",
            value=None,
            step=1,
            placeholder="Use to replicate response",
            min_value=1,
        )
        generation_method = st.selectbox(
            "Method of Generation:",
            ("top-k", "multinomial", "temperature", "greedy", "nucleus"),
            index=0,
        ).strip()

        if generation_method == "top-k":
            specified_k = st.number_input("Value for k:", value=5, step=1)

        if generation_method == "nucleus":
            specified_nucleus = st.number_input(
                "Value for p:", value=0.5, step=0.05, min_value=0.0, max_value=1.0
            )

        if generation_method == "temperature":
            specified_temperature = st.number_input(
                "Value for temperature:",
                value=0.9,
                step=0.05,
                min_value=0.0,
                max_value=1.0,
            )

        max_tokens = st.slider("Max Tokens Generated:", 100, 500, 400)

    ## Settings Clean up
    if not user_seed:
        user_seed = 7

    # model_version = st.radio("Which model would you like to use?", ["smoll", "beeg"])
    # small_model = load_casey_model(tokenizer, device)
    model = load_big_model(tokenizer, device)

    if st.button("Write my story!"):
        placeholder = st.empty()
        # if model_version == 'smoll':
        #     model = load_casey_model(tokenizer, device)
        # elif model_version == 'beeg':
        #     model = load_big_model(tokenizer, device)
        # with placeholder.container():
        #     st.write("Model Loaded! Preparing to Generate...")

        with st.spinner(""):
            result = generate_inference(
                model,
                tokenizer,
                device,
                method=generation_method,
                k=specified_k,
                p_nucleus=specified_nucleus,
                temp=specified_temperature,
                max_new_tokens=max_tokens,
                cond=user_input,
                deterministic=user_seed,
            )

        streamed_input = ""
        for word in user_input.split(" "):
            streamed_input += word
            with placeholder.container():
                st.markdown(f"**{streamed_input}**")
            streamed_input += " "
            time.sleep(0.1)

        if user_input != "":  ##conditional
            result = result[len(user_input) + 3 :]
            streamed_result = f"**{streamed_input[:-1]}**"
            time.sleep(1)
        else:  ##unconditional
            streamed_result = ""

        for word in result.split(" "):
            streamed_result += word + " "
            with placeholder.container():
                st.write(streamed_result)
            time.sleep(0.1)
        if st.button("Clear Output"):
            placeholder = st.empty()


if __name__ == "__main__":
    main()