# libraries and modules
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# function to translate text
def translate_text(text, model, tokenizer):
    
    # tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    
    # get input_ids and attention_mask for the model
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # generate translation using the model
    outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=512, num_beams=5, early_stopping=True)

    #decode the output tokens to text
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text, input_ids, attention_mask


#function to calculate loss and BLEU score
def calculate_loss_and_bleu(model, input_ids, attention_mask, target_text, tokenizer):

    # tokenize the target text
    target_ids = tokenizer(target_text, return_tensors="pt", max_length=512, truncation=True, padding=True)["input_ids"]

    # compute loss using the model's forward method
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to track gradients during inference
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
        loss = outputs.loss

    # generate the translation
    generated_translation, _, _ = translate_text(target_text, model, tokenizer)
    
    # tokenize the generated and target translations for BLEU score computation
    reference = [target_text.split()]  # reference should be a list of tokenized sentences
    candidate = generated_translation.split()  # tokenized generated translation

    # Compute BLEU score
    smoothing_function = SmoothingFunction().method4  # Apply smoothing to avoid 0 scores
    bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothing_function)
    
    return loss.item(), bleu_score


if __name__ == "__main__":
    # load a pre-trained model and tokenizer for translation
    model_name = "Helsinki-NLP/opus-mt-en-tl"  # English to Tagalog
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # input text in English
    english_text = input("ENGLISH: ")

    # specify a target translation for comparison
    target_translation = input("TARGET (Ground truth translation): ")  # example: "Magandang umaga"

    # translate to Tagalog
    tagalog_translation, input_ids, attention_mask = translate_text(english_text, model, tokenizer)

    # display the original, generated, and target translations
    print("\nOriginal (English):", english_text)
    print("Target Translation (Tagalog):", target_translation)
    print('\n======================================================================================\n')
    print("Generated Translation (Tagalog):", tagalog_translation)

    # calculate loss and BLEU score
    loss, bleu_score = calculate_loss_and_bleu(model, input_ids, attention_mask, target_translation, tokenizer)
    
    print("\nLoss:", loss)
    print("BLEU Score:", bleu_score)
