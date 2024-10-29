# MinionGPT

MinionGPT is a research assistant powered by Llama language models, designed to help users extract information from scientific papers and provide clear, organized responses. It leverages Streamlit for the user interface and LangChain to handle large language model (LLM) responses. With MinionGPT, users can upload scientific PDFs, query them, and receive answers that clearly differentiate between information sourced directly from the papers and general knowledge.

## Features

- **Upload Multiple PDFs**: Supports uploading multiple PDFs for analysis.
- **Clear Citation and Knowledge Separation**: Responses are divided into "Information from Papers" (cited) and "General Knowledge" to maintain clarity.
- **Interactive Interface**: Uses Streamlit for a user-friendly chat interface.
- **Context-aware Responses**: Generates answers based on specific context within the uploaded documents and general information when needed.

## Setup Instructions

### Prerequisites

- Python 3.8 or later
- Required Python packages (see `requirements.txt` below)
- [Streamlit](https://streamlit.io/) for UI
- [LangChain](https://github.com/hwchase17/langchain) for LLM chain management
- [FAISS](https://faiss.ai/) for vector storage

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/MinionGPT.git
   cd MinionGPT
   ```

2. **Create and Activate a Virtual Environment** 

  ```bash
   python3 -m venv venv
   source venv/bin/activate
```
3.	**Install Requirements**

   ```bash
streamlit run agent.py
```
