import gradio as gr
import os
from dotenv import load_dotenv
from openai import OpenAI
from scrapeData import scrape_single_blog_post, get_all_clean_blog_text, blog_urls

# Load environment variables from .env file
load_dotenv()

# In-memory storage for scraped data
scraped_data = {
    "full_text": None,
    "blog_contents": {}
}

# Use environment variable for API key (already in your .env file)
OPEN_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCdpc-iIftehlnCDQpK04j1WnTSV-1Npbc")
model = OpenAI(api_key=OPEN_API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

def run_scraper():
    """Run the scraper once to collect diabetes information"""
    full_text = get_all_clean_blog_text()
    
    blog_contents = {}
    for url in blog_urls:
        blog_contents[url] = scrape_single_blog_post(url)
    
    scraped_data["full_text"] = full_text
    scraped_data["blog_contents"] = blog_contents
    
    return "Data loaded successfully!"

def get_system_context():
    """Get all scraped content as context for the LLM"""
    if not scraped_data["full_text"]:
        run_scraper()
    
    system_context = "DIABETES INFORMATION DATABASE:\n\n"
    
    for url, content in scraped_data["blog_contents"].items():
        blog_name = url.split('/')[-1].replace('-', ' ').title()
        if "?" in blog_name:
            blog_name = blog_name.split('?')[0]

        truncated_content = content[:2000] + ("..." if len(content) > 2000 else "")
        system_context += f"SOURCE: {blog_name}\n{truncated_content}\n\n---\n\n"
    
    return system_context

def gemini_chat(message, history):
    if not scraped_data["full_text"]:
        run_scraper()

    system_content = f"""
    You are a helpful diabetes assistant called Gemini. 
    You give friendly, conversational responses to help people manage diabetes.
    You explain concepts in simple terms.
    Always remind users to consult healthcare professionals for medical advice.
    
    The following information comes from trusted medical sources about diabetes.
    Base your answers on this information:
    
    {get_system_context()}
    """
    
    try:
        response = model.chat.completions.create(
            model="gemini-1.5-flash",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": message}
            ],
            max_tokens=500,
        )
        
        bot_message = response.choices[0].message.content

        if not any(phrase in bot_message.lower() for phrase in ["consult", "healthcare provider", "doctor", "physician", "medical professional"]):
            bot_message += "\n\n*Remember: Always consult your healthcare provider for personalized medical advice.*"
            
        return bot_message
    
    except Exception as e:
        return f"I'm sorry, I couldn't process your request. Error: {str(e)}"

# Custom CSS for better appearance
css = """
.gradio-container {
    background-color: #1a1a1a; 
}
.container {
    max-width: 900px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding-top: 1.5rem !important;
}
.header-text {
    text-align: center;
    color: black;
}
.header-subtext {
    text-align: center;
    color: black;
    margin-bottom: 2rem;
}
.examples-container {
    background-color: #f1f8e9;
    border-radius: 8px;
    padding: 1rem;
    margin-top: 1.5rem;
    border: 1px solid #c5e1a5;
    height: 25vh
}
.examples-header {
    color: #33691e;
    font-size: 1.3em;
    margin-top: 0;
}
.footer {
    text-align: center;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #e0e0e0;
    color: #7f8c8d;
    font-size: 0.9em;
}
"""

with gr.Blocks(title="DiaCare AI", css=css) as demo:
    with gr.Column(elem_classes="container"):
        gr.Markdown("# DiaCare AI", elem_classes="header-text")
        gr.Markdown("Your personal diabetes management assistant powered by trusted medical sources", elem_classes="header-subtext")
        


        chatbot = gr.Chatbot(
            height=500,
            avatar_images=("https://i.imgur.com/m5oYSal.png", "https://i.imgur.com/8kdkkiD.png"),
            elem_id="chatbot"
        )
        
        with gr.Row():
            chat_input = gr.Textbox(
                placeholder="Ask me anything about diabetes...",
                show_label=False,
                container=False,
                scale=9,
                elem_id="chat-input"
            )
            chat_button = gr.Button("Send", variant="primary", scale=1)
        
        with gr.Column(elem_classes="examples-container"):
            gr.Markdown("### How can I help you today?", elem_classes="examples-header")
            
            with gr.Row():
                with gr.Column(scale=1):
                    q1_button = gr.Button("What is Type 2 diabetes?")
                    q2_button = gr.Button("Diet recommendations")
                
                with gr.Column(scale=1):
                    q3_button = gr.Button("Exercise benefits")
                    q4_button = gr.Button("Blood sugar management")
            
            gr.Markdown("""
            You can also ask about:
            - Symptoms of hypoglycemia
            - How often to check blood glucose 
            - Diabetes complications
            - Good breakfast options for diabetics
            """)
        
        gr.Markdown("""
        <div class="footer">
        DiaCare AI provides information from trusted medical sources but is not a replacement for professional medical advice.
        <br>© 2023 DiaCare AI - Helping you manage diabetes better
        </div>
        """, elem_classes="footer")

    def respond(message, chat_history):
        bot_message = gemini_chat(message, chat_history)
        chat_history.append((message, bot_message))
        return "", chat_history
    
    # Set up chat functionality
    chat_input.submit(respond, [chat_input, chatbot], [chat_input, chatbot])
    chat_button.click(respond, [chat_input, chatbot], [chat_input, chatbot])
    
    # Add quick question buttons functionality
    q1_button.click(lambda: ("What is Type 2 diabetes?", []), None, [chat_input, chatbot], queue=False).then(
        respond, [chat_input, chatbot], [chat_input, chatbot]
    )
    q2_button.click(lambda: ("What diet do you recommend for diabetes?", []), None, [chat_input, chatbot], queue=False).then(
        respond, [chat_input, chatbot], [chat_input, chatbot]
    )
    q3_button.click(lambda: ("How does exercise help with diabetes?", []), None, [chat_input, chatbot], queue=False).then(
        respond, [chat_input, chatbot], [chat_input, chatbot]
    )
    q4_button.click(lambda: ("Tips for managing blood sugar levels", []), None, [chat_input, chatbot], queue=False).then(
        respond, [chat_input, chatbot], [chat_input, chatbot]
    )

if __name__ == "__main__":
    run_scraper()
    demo.launch()