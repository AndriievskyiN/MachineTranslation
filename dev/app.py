from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, AutoTokenizer

import sys
sys.path.append("..")
from model import Transformer
from utils import tokenize_and_mask_for_translation

app = FastAPI()

# Configure CORS settings
origins = [
    "http://127.0.0.1:5500",  # Add your frontend URL here
    "http://localhost:5500"   # Add your frontend URL here (if using localhost)
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with the actual list of allowed origins
    allow_methods=["*"],  # Update with the allowed HTTP methods
    allow_headers=["*"],  # Update with the allowed headers
)

# Load your trained model and tokenizer
DEVICE = torch.device("cuda" if torch.cuda.is_available else "cpu")
MAX_LENGTH = 22
EOS_TOKEN = 103
# Initialize the english tokenizer
tokenizer_en = BertTokenizer.from_pretrained("bert-base-cased")
# Initialize the german tokenizer
tokenizer_ge = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
vocab_size_en = tokenizer_en.vocab_size
vocab_size_ge = tokenizer_ge.vocab_size

model = Transformer(
    src_vocab_size=vocab_size_en, 
    trgt_vocab_size=vocab_size_ge,
    n_embed=512,
    max_length=MAX_LENGTH, 
    n_heads=8, 
    n_layers=8, 
    dropout_p=0.2,
    device=DEVICE
)

model.load_state_dict(torch.load("transformer.pt", map_location=torch.device('cpu')))

class Item(BaseModel):
    input_text: str

@app.get("/")
async def home():
    return {"message": "Welcome to the translation API"}


@app.post("/translate/")
async def translate(input_text: Item):
    try:
        input_text = input_text.input_text
        model.eval()
        
        src = tokenizer_en([input_text], padding="max_length", truncation=True, max_length=MAX_LENGTH)
        enc_input = torch.tensor(src["input_ids"])
        src_mask = torch.tensor(src["attention_mask"])
        
        # Create initial translation with the start-of-sentence token
        trg_tokens = torch.tensor(tokenizer_ge.convert_tokens_to_ids(["[CLS]"]), dtype=torch.int64)
        
        with torch.no_grad():
            for word in range(MAX_LENGTH):
                dec_input, dec_mask = tokenize_and_mask_for_translation(trg_tokens, tokenizer_ge, MAX_LENGTH, DEVICE).values()
                logits = model(enc_input, dec_input, src_mask, dec_mask)
                next_word_prob_distribution = logits[0][word]
                next_token = torch.argmax(next_word_prob_distribution).item()

                if next_token == EOS_TOKEN:
                    break
                
                # Append to trg_tokens
                trg_tokens = torch.cat((trg_tokens, torch.tensor([next_token])), dim=0)

        # Decode the token IDs to words using the target tokenizer
        translated_sentence = tokenizer_ge.decode(trg_tokens[1:])
        
        return {"translated_sentence": translated_sentence}
    
    except Exception as e:
        return {"error": str(e)}


