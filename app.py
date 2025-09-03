# main.py
# Main entry point for HR AI Tools application
import os
import gradio as gr
from tools import main

if __name__ == "__main__":
    demo=main()
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))