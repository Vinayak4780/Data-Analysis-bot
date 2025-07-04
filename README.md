# Data Analyst Agent

A powerful AI-powered data analyst agent that can process multiple file formats and provide intelligent data analysis, visualizations, and Q&A capabilities.

## Features

- **Multi-format Support**: Upload and analyze .doc, .txt, .xlsx, .csv, .pdf, and image files
- **Intelligent Analysis**: Uses meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 for advanced reasoning
- **Data Visualization**: Creates interactive charts and graphs using Plotly and Matplotlib
- **Q&A Interface**: Ask follow-up questions about your data
- **Streamlit UI**: User-friendly web interface

## Installation

1. Clone or download this project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Together.ai API key:
   - Create a `.env` file in the project directory
   - Add: `TOGETHER_API_KEY=your_api_key_here`

4. Install Tesseract OCR for image processing:
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - Add Tesseract to your PATH

## Usage

### Command Line Interface
```bash
python data_analyst_agent.py
```

### Streamlit Web Interface
```bash
streamlit run streamlit_app.py
```

## Supported File Types

- **Text Files**: .txt, .doc, .docx
- **Spreadsheets**: .xlsx, .csv
- **PDFs**: .pdf
- **Images**: .jpg, .jpeg, .png, .bmp (with OCR)

## Architecture

The agent consists of several key components:

1. **File Processor**: Handles different file formats and extracts content
2. **Data Analyzer**: Performs statistical analysis and insights generation
3. **Visualization Engine**: Creates charts and graphs
4. **LLM Interface**: Integrates with Together.ai for intelligent responses
5. **Memory System**: Maintains context across conversations
#
