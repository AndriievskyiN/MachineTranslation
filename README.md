# Machine Translation with Transformer

This project involves training a Transformer model from scratch for a machine translation task, specifically translating English sentences to German. The Transformer architecture has shown remarkable results in natural language processing tasks, and this project showcases its capabilities in the translation domain.

![Website preview](https://github.com/AndriievskyiN/MachineTranslation/assets/92473539/522fce30-6ae5-460b-9736-a71bf6c7d19e)


## Overview

In this project, I trained a Transformer model on a relatively small dataset of English-to-German translations. The Transformer architecture, known for its self-attention mechanism, enables capturing complex linguistic patterns, making it an ideal choice for machine translation tasks.

The model was trained using PyTorch and the Hugging Face Transformers library. The training loop, tokenization, and evaluation were implemented from scratch. The trained model is used to power a simple web application that accepts English sentences as input and provides their German translations as output.

## Getting Started

To run the web application and test the trained model, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/AndriievskyiN/MachineTranslation.git
    ``````
   
   
2. Go into the downloaded directory
    ```bash
    cd transformer-translation
    ```
    
3. Download the required libraries
    ```bash
    pip install -r requirements.txt
    ```
    
4. Go into the dev directory
    ```bash
    cd dev
    ```

5. Run the FastAPI server
    ```bash
    uvicorn app:app --reload
    ```   

6. Open the ***index.html*** file to access the web application


# Notes
The training dataset used in this project is relatively small, which might limit the model's performance. Consider training on a larger dataset for improved translation quality.
The provided code is a simplified demonstration. In production scenarios, additional considerations like security and optimization are essential.
Feel free to experiment with the code, try different configurations, and contribute to the project's improvement!

# Acknowledgments
This project is inspired by the Transformer architecture introduced in the "Attention Is All You Need" paper by Vaswani et al.

Here is the link to the paper: https://arxiv.org/pdf/1706.03762.pdf
