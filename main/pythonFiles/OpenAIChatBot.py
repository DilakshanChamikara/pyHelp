import openai

# Set your OpenAI API key
openai.api_key = "your_openai_api_key"  # Replace with your API key


def chatbot():
    print("AI Chatbot: Hello! I'm your AI assistant. Type 'exit' to end the chat.")

    while True:
        # Get user input
        user_input = input("You: ")

        # Exit condition
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("AI Chatbot: Goodbye!")
            break

        try:
            # Call OpenAI GPT model
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Use "gpt-4" if available
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_input},
                ],
                max_tokens=150,
                temperature=0.7,
            )

            # Extract and print the AI's response
            ai_response = response['choices'][0]['message']['content']
            print(f"AI Chatbot: {ai_response}")

        except Exception as e:
            print(f"AI Chatbot: Oops! Something went wrong. ({e})")


# Run the chatbot
if __name__ == "__main__":
    chatbot()
