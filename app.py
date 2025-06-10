import gradio as gr
from openai import OpenAI
from scrapeData import scrape_single_blog_post, get_all_clean_blog_text, blog_urls


scraped_data = {
    "full_text": None,
    "blog_contents": {}
}


openai_api_key = "AIzaSyCdpc-iIftehlnCDQpK04j1WnTSV-1Npbc" 
model = OpenAI(api_key=openai_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

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


with gr.Blocks(title="Diabetes Chatbot Assistant") as demo:
    gr.Markdown("# Diabetes Chatbot Assistant")
    gr.Markdown("Chat with an AI assistant about diabetes management and health information.")
    
    with gr.Row():
        status = gr.Markdown("*Loading diabetes information...*")

    chatbot = gr.Chatbot(
    height=500,
    avatar_images=("https://i.imgur.com/m5oYSal.png", "https://i.imgur.com/8kdkkiD.png")
)
    
    with gr.Row():
        chat_input = gr.Textbox(
            placeholder="Ask me anything about diabetes...",
            show_label=False,
            container=False,
            scale=9
        )
        chat_button = gr.Button("Send", variant="primary", scale=1)
    
    gr.Markdown("""
    ### Example Questions:
    - What is the difference between Type 1 and Type 2 diabetes?
    - How can exercise help control blood sugar?
    - What symptoms should I watch for with hypoglycemia?
    - How often should I check my blood glucose?
    - What are good breakfast options for diabetics?
    """)
    

    def respond(message, chat_history):
        bot_message = gemini_chat(message, chat_history)
        chat_history.append((message, bot_message))
        return "", chat_history
    
    chat_input.submit(respond, [chat_input, chatbot], [chat_input, chatbot])
    chat_button.click(respond, [chat_input, chatbot], [chat_input, chatbot])
    
    demo.load(lambda: "Data loaded and ready to chat!", None, status)

if __name__ == "__main__":

    run_scraper()

    demo.launch()