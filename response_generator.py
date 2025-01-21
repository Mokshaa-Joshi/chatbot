from transformers import pipeline

def generate_response(query, model_name="mistralai/Mixtral-8x7B-v0.1"):
    generator = pipeline('text-generation', model=model_name)
    response = generator(query, max_length=100, num_return_sequences=1)
    
    return response[0]['generated_text']

