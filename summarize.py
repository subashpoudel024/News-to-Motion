
def summarize(cleaned_content,model,tokenizer,max_length=150,min_length=30):
    print('Starting to summarize')
    inputs = tokenizer.encode("summarize: " + cleaned_content, return_tensors="pt", max_length=1024, truncation=True)
    print('tokenizer loaded')
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    print('summary generated')
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print('summary decoded')
    return summary


# Function to generate summary using BART
def generate_summary_bart(cleaned_content,bart_tokenizer,bart_model):
    inputs = bart_tokenizer(cleaned_content, max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = bart_model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True)
    summary_bart = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary_bart
    





