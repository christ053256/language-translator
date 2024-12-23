from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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
    return translated_text

if __name__ == "__main__":
    # Load a pre-trained model and tokenizer for translation
    model_name = "Helsinki-NLP/opus-mt-en-tl"  # English to Tagalog
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Input text in English
    english_text = input("ENGLISH: ")

    # Translate to Tagalog
    tagalog_translation = translate_text(english_text, model, tokenizer)

    # Display the original and generated translations
    print("\nOriginal (English):", english_text)
    print("Generated Translation (Tagalog):", tagalog_translation)
