#BLEU Score: 0.2213885886251307 : ACCURACY
#BLEU Score: 0.7292571723872933

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Function to translate text
def translate_text(text, model, tokenizer):
    """
    Translate the given text using the specified model and tokenizer.

    Args:
        text (str): Text to translate.
        model: Pre-trained translation model.
        tokenizer: Pre-trained tokenizer.

    Returns:
        str: Translated text.
    """
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    
    # Get input_ids and attention_mask for the model
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Generate translation using the model
    outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=512, num_beams=5, early_stopping=True)

    # Decode the output tokens to text
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text, input_ids, attention_mask


# Function to calculate loss and BLEU score
def calculate_loss_and_bleu(model, input_ids, attention_mask, target_text, tokenizer):
    """
    Calculate loss and BLEU score for the translation model.

    Args:
        model: Pre-trained translation model.
        input_ids: Tokenized input text.
        attention_mask: Attention mask for the input.
        target_text: Ground truth translation for comparison.
        tokenizer: Pre-trained tokenizer.

    Returns:
        loss: The computed loss value.
        bleu: BLEU score of the model's translation.
    """
    # Tokenize the target text
    target_ids = tokenizer(target_text, return_tensors="pt", max_length=512, truncation=True, padding=True)["input_ids"]

    # Compute loss using the model's forward method
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to track gradients during inference
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
        loss = outputs.loss

    # Generate the translation
    generated_translation, _, _ = translate_text(target_text, model, tokenizer)
    
    # Tokenize the generated and target translations for BLEU score computation
    reference = [target_text.split()]  # Reference should be a list of tokenized sentences
    candidate = generated_translation.split()  # Tokenized generated translation

    # Compute BLEU score
    smoothing_function = SmoothingFunction().method4  # Apply smoothing to avoid 0 scores
    bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothing_function)
    
    return loss.item(), bleu_score


if __name__ == "__main__":
    # Load a pre-trained model and tokenizer for translation
    model_name = "Helsinki-NLP/opus-mt-en-tl"  # English to Tagalog
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Input text in English
    english_text = input("ENGLISH: ")

    # Specify a target translation for comparison (you should use a real translation in practice)
    target_translation = input("TARGET (Ground truth translation): ")  # Example: "Magandang umaga Ms. Anderson"

    # Translate to Tagalog
    tagalog_translation, input_ids, attention_mask = translate_text(english_text, model, tokenizer)

    # Display the original, generated, and target translations
    print("\nOriginal (English):", english_text)
    print("Generated Translation (Tagalog):", tagalog_translation)
    print("Target Translation (Tagalog):", target_translation)

    # Calculate loss and BLEU score
    loss, bleu_score = calculate_loss_and_bleu(model, input_ids, attention_mask, target_translation, tokenizer)
    
    print("\nLoss:", loss)
    print("BLEU Score:", bleu_score)
