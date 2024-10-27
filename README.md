# Ollama Computer Use (for Mac)

This project is a fork of [Anthropic Computer Use](https://github.com/anthropics/anthropic-quickstarts/blob/main/computer-use-demo/README.md), modified to run on macOS and use a locally run Ollama Nemotron instance instead of the Anthropic API.

> [!CAUTION]
> This comes with obvious risks. The AI agent can control everything on your Mac. Please be careful.
> The Nemotron model may have different safety measures compared to Claude, so exercise additional caution.

## Setup Instructions

1. Clone the repository and navigate to it:

```bash
git clone https://github.com/your-username/ollama-computer-use-mac
cd ollama-computer-use-mac
```

2. Create and activate a virtual environment:

```bash
python3.12 -m venv venv
source venv/bin/activate
```

3. Run the setup script:

```bash
chmod +x setup.sh
./setup.sh
```

4. Install Python requirements:

```bash
pip install -r requirements.txt
```

5. Install and run Ollama:

Follow the instructions at [Ollama's official website](https://ollama.ai/) to install Ollama on your Mac.

Once installed, run the Nemotron model:

```bash
ollama run nemotron-3-8b-chat
```

## Running the Demo

1. Set up your environment:

In a `.env` file, add:

```
API_PROVIDER=ollama
OLLAMA_API_URL=http://localhost:11434
WIDTH=800
HEIGHT=600
DISPLAY_NUM=1
```

Set the screen dimensions (recommended: stay within XGA/WXGA resolution).

2. Activate the virtual environment and set the necessary environment variables:

```bash
source activate.sh
```

3. Start the Streamlit app:

```bash
streamlit run app.py
```

The interface will be available at http://localhost:8501

## Screen Size Considerations

We recommend using one of these resolutions for optimal performance:

- XGA: 1024x768 (4:3)
- WXGA: 1280x800 (16:10)
- FWXGA: 1366x768 (~16:9)

Higher resolutions will be automatically scaled down to these targets to optimize model performance. You can set the resolution using environment variables:

```bash
export WIDTH=1024
export HEIGHT=768
streamlit run app.py
```

## Important Notes

- This version uses a locally run Ollama Nemotron instance instead of the Anthropic API. Make sure Ollama is running with the Nemotron model before starting the application.
- The system prompt and tool implementations are optimized for macOS. You may need to adjust them if you encounter any OS-specific issues.
- Always exercise caution when allowing an AI model to control your computer. Review and understand the actions it's taking.

> [!IMPORTANT]
> This is an experimental project and may contain bugs or unexpected behavior. Use at your own risk and always monitor the AI's actions closely.
