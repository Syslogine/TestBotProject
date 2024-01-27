from transformers import BertTokenizer, BertForSequenceClassification
import torch
import requests

# Load the fine-tuned model and tokenizer
model = BertForSequenceClassification.from_pretrained("./my_finetuned_bert")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def fetch_from_stackexchange(query):
    url = "https://api.stackexchange.com/2.2/search/advanced"
    params = {
        "order": "desc",
        "sort": "relevance",
        "q": query,
        "site": "stackoverflow",
        "pagesize": 1  # Fetch only the top result
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['items']:
            top_result = data['items'][0]
            return top_result['title'], top_result['link']
        else:
            return "No results found.", ""
    else:
        return "Error in API request.", ""

def classify_intent(text):
    # Tokenize the text input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits to probabilities and get the predicted class
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    
    print(f"Predicted Intent: {predicted_class}")

    return predicted_class



def get_response(intent_id, user_query):
    if intent_id == 0:  # Assuming 0 represents a programming-related query
        title, link = fetch_from_stackexchange(user_query)
        if link:
            return f"I found something that might help: {title} {link}"
        else:
            return title
    elif intent_id == 1:
        return "This seems to be a general query. How can I assist you further?"

    else:
        return "I'm not sure how to answer that."


def main():
    print("Welcome to the Chatbot!")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        intent_id = classify_intent(user_input)  # This needs to be implemented
        response = get_response(intent_id, user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()