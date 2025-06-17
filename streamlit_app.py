import streamlit as st
import os
import tempfile
import pandas as pd
from data_analyst_agent import DataAnalystAgent, FileType
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import json

# Configure Streamlit page
st.set_page_config(
    page_title="🤖 Data Analyst Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .insight-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-left: 4px solid #4CAF50;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-left: 4px solid #ffc107;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-left: 4px solid #28a745;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_agent():
    """Initialize the data analyst agent"""
    try:
        if st.session_state.agent is None:
            st.session_state.agent = DataAnalystAgent()
        return True
    except Exception as e:
        st.error(f"Failed to initialize agent: {e}")
        st.info("Please make sure you have set up your TOGETHER_API_KEY in the .env file")
        return False

def display_file_info(result):
    """Display file information and analysis results"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📋 Analysis Summary")
        st.markdown(f'<div class="success-box">{result.summary}</div>', unsafe_allow_html=True)
        
        st.markdown("### 🔍 Key Insights")
        for insight in result.insights:
            st.markdown(f'<div class="insight-box">• {insight}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 File Information")
        st.info(f"**File Type:** {result.file_type.value.upper()}")
        
        if result.file_type in [FileType.CSV, FileType.XLSX]:
            shape = result.metadata.get('shape', (0, 0))
            st.info(f"**Dimensions:** {shape[0]} rows × {shape[1]} columns")
            
            missing_values = result.metadata.get('missing_values', {})
            total_missing = sum(missing_values.values())
            if total_missing > 0:
                st.warning(f"**Missing Values:** {total_missing}")
            else:
                st.success("**No Missing Values** ✅")

def display_visualizations(result):
    """Display generated visualizations"""
    if result.visualizations:
        st.markdown("### 📈 Visualizations")
        
        for viz_path in result.visualizations:
            if viz_path.endswith('.html'):
                # Display Plotly HTML files
                try:
                    with open(viz_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=500)
                except Exception as e:
                    st.error(f"Error loading visualization: {e}")
            
            elif viz_path.endswith('.png'):
                # Display PNG files (like word clouds)
                try:
                    image = Image.open(viz_path)
                    st.image(image, caption=os.path.basename(viz_path), use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {e}")

def display_data_preview(result):
    """Display data preview for structured data"""
    if result.file_type in [FileType.CSV, FileType.XLSX]:
        st.markdown("### 👀 Data Preview")
        
        if isinstance(result.content, pd.DataFrame):
            # Show basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", result.content.shape[0])
            with col2:
                st.metric("Columns", result.content.shape[1])
            with col3:
                st.metric("Memory Usage", f"{result.content.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            # Show data types
            st.markdown("#### Data Types")
            dtype_df = pd.DataFrame({
                'Column': result.content.columns,
                'Type': result.content.dtypes.astype(str),
                'Non-Null Count': result.content.count(),
                'Null Count': result.content.isnull().sum()
            })
            st.dataframe(dtype_df, use_container_width=True)
            
            # Show first few rows
            st.markdown("#### First 10 Rows")
            st.dataframe(result.content.head(10), use_container_width=True)
            
            # Show statistical summary for numeric columns
            numeric_cols = result.content.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                st.markdown("#### Statistical Summary")
                st.dataframe(result.content[numeric_cols].describe(), use_container_width=True)

def chat_interface():
    """Chat interface for asking questions"""
    st.markdown("### 💬 Ask Questions About Your Data")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("#### Conversation History")
        for i, exchange in enumerate(st.session_state.chat_history):
            with st.expander(f"Q{i+1}: {exchange['question'][:50]}..."):
                st.markdown(f"**Question:** {exchange['question']}")
                st.markdown(f"**Answer:** {exchange['answer']}")
    
    # New question input
    question = st.text_input("Ask a question about your data:", placeholder="e.g., What are the main trends in this data?")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("Ask Question", type="primary")
    
    if ask_button and question.strip():
        if st.session_state.agent and st.session_state.analysis_result:
            with st.spinner("Thinking..."):
                try:
                    answer = st.session_state.agent.ask_question(question)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'question': question,
                        'answer': answer
                    })
                    
                    # Display the answer
                    st.markdown("#### Latest Answer")
                    st.markdown(f'<div class="success-box">{answer}</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
        else:
            st.warning("Please upload and process a file first!")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Data Analyst Agent</h1>
        <p>Upload your data files and get intelligent analysis, insights, and visualizations powered by AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 Configuration")
        
        # API Key check
        api_key = os.getenv('TOGETHER_API_KEY')
        if api_key:
            st.success("✅ Together.ai API Key detected")
        else:
            st.error("❌ Together.ai API Key not found")
            st.markdown("""
            Please set up your API key:
            1. Create a `.env` file
            2. Add: `TOGETHER_API_KEY=your_key_here`
            """)
        
        st.markdown("### 📁 Supported File Types")
        st.markdown("""
        - **Spreadsheets:** .csv, .xlsx
        - **Documents:** .txt, .docx, .pdf
        - **Images:** .jpg, .png, .bmp (OCR)
        """)
        
        st.markdown("### 🚀 Features")
        st.markdown("""
        - Intelligent data analysis
        - Automatic visualizations
        - Q&A with follow-up questions
        - Multi-format support
        - Statistical insights
        """)
    
    # Initialize agent
    if not initialize_agent():
        return
    
    # File upload section
    st.markdown("## 📤 Upload Your Data")
    
    uploaded_file = st.file_uploader(
        "Choose a file to analyze",
        type=['csv', 'xlsx', 'txt', 'docx', 'pdf', 'jpg', 'jpeg', 'png', 'bmp'],
        help="Upload any supported file format for analysis"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Process file button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 Analyze File", type="primary", use_container_width=True):
                with st.spinner("Processing file... This may take a moment."):
                    try:
                        # Process the file
                        result = st.session_state.agent.process_file(tmp_file_path)
                        st.session_state.analysis_result = result
                        
                        st.success("✅ File processed successfully!")
                        
                    except Exception as e:
                        st.error(f"Error processing file: {e}")
                        st.exception(e)
        
        # Clean up temporary file
        try:
            os.unlink(tmp_file_path)
        except:
            pass
    
    # Display results if available
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "👀 Data Preview", "📈 Visualizations", "💬 Chat"])
        
        with tab1:
            display_file_info(result)
        
        with tab2:
            display_data_preview(result)
        
        with tab3:
            display_visualizations(result)
        
        with tab4:
            chat_interface()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🤖 Powered by meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 via Together.ai</p>
        <p>Built with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
