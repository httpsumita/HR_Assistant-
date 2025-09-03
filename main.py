# main.py
# Main entry point for HR AI Tools application
import os
import gradio as gr
from tools import main

if __name__ == "__main__":
    demo=main()
    demo.launch()