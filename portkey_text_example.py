# Install the Portkey AI Gateway SDK with pip
#   pip -i portkey-ai
# 
# For more information on the SDK see https://portkey.ai/docs/api-reference/sdk/python
# 
from portkey_ai import Portkey
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Before executing this code, define the API Key within an enironment variable in your OS
# Linux BASH example: export AI_SANDBOX_KEY=<key provided to you>

# Import API key from OS environment variables
AI_SANDBOX_KEY = os.getenv("AI_SANDBOX_KEY")

client = Portkey(api_key=AI_SANDBOX_KEY)

# Set the model deployment name that the prompt should be sent to
available_models = [
                    "gpt-5",
                    "o3-mini",
                    "gpt-4o-mini",
                    "gpt-4o", 
                    "gpt-4-turbo",
                    "gpt-35-turbo-16k", 
                    "Llama-3.3-70B-Instruct", 
                    "Meta-Llama-3-1-8B-Instruct", 
                    "mistral-small-2503",
                    "Mistral-Large-2411"
                ]

# This function will submit a simple text prompt to the chosen model
def text_prompt_example(model_to_be_used):
    # Establish a connection to your Azure OpenAI instance
    
    try:
        response = client.chat.completions.create(
        model=model_to_be_used, 
        temperature=0.5, # temperature = how creative/random the model is in generating response - 0 to 1 with 1 being most creative
        max_tokens=1000, # max_tokens = token limit on context to send to the model
        top_p=0.5, # top_p = diversity of generated text by the model considering probability attached to token - 0 to 1 - ex. top_p of 0.1 = only tokens within the top 10% probability are considered
        messages=[
        {"role": "system", "content": "You are a helpful junior physics professor."}, # describes model identity and purpose
        {"role": "user", "content": "Please explain quantum mechanics in 100 words or less."}, # user prompt
               ]
        )
        print("\n"+response.choices[0].message.content)

    except Exception as e:
        print(e.message)


#
# Execute the example functions
if __name__ == "__main__":

    # Test text prompts with all available models
    for i in range(len(available_models)):
    # Execute the text prompt example
        modelnum = i
        print("\nModel: " + available_models[modelnum])
        text_prompt_example(available_models[modelnum])

