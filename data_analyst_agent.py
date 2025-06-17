import os
import iO
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

# File processing imports
from docx import Document
import PyPDF2
from PIL import Image
import pytesseract
import openpyxl

# LLM and API imports
import together
from dotenv import load_dotenv

# Machine learning imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from wordcloud import WordCloud

warnings.filterwarnings('ignore')
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileType(Enum):
    CSV = "csv"
    XLSX = "xlsx"
    TXT = "txt"
    DOC = "doc"
    DOCX = "docx"
    PDF = "pdf"
    IMAGE = "image"

@dataclass
class AnalysisResult:
    file_type: FileType
    content: Any
    metadata: Dict[str, Any]
    summary: str
    insights: List[str]
    visualizations: List[str]

class FileProcessor:
    """Handles processing of different file types"""
    
    @staticmethod
    def detect_file_type(file_path: str) -> FileType:
        """Detect file type based on extension"""
        ext = os.path.splitext(file_path)[1].lower()
        
        type_mapping = {
            '.csv': FileType.CSV,
            '.xlsx': FileType.XLSX,
            '.xls': FileType.XLSX,
            '.txt': FileType.TXT,
            '.doc': FileType.DOC,
            '.docx': FileType.DOCX,
            '.pdf': FileType.PDF,
            '.jpg': FileType.IMAGE,
            '.jpeg': FileType.IMAGE,
            '.png': FileType.IMAGE,
            '.bmp': FileType.IMAGE,
        }
        
        return type_mapping.get(ext, FileType.TXT)
    
    @staticmethod
    def process_csv(file_path: str) -> pd.DataFrame:
        """Process CSV files"""
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    logger.info(f"Successfully read CSV with {encoding} encoding")
                    return df
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, try with error handling
            df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
            return df
            
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            raise
    
    @staticmethod
    def process_xlsx(file_path: str) -> pd.DataFrame:
        """Process Excel files"""
        try:
            df = pd.read_excel(file_path)
            return df
        except Exception as e:
            logger.error(f"Error processing Excel file: {e}")
            raise
    
    @staticmethod
    def process_text(file_path: str) -> str:
        """Process text files"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error processing text file: {e}")
            raise
    
    @staticmethod
    def process_docx(file_path: str) -> str:
        """Process DOCX files"""
        try:
            doc = Document(file_path)
            full_text = []
            for paragraph in doc.paragraphs:
                full_text.append(paragraph.text)
            return '\n'.join(full_text)
        except Exception as e:
            logger.error(f"Error processing DOCX file: {e}")
            raise
    
    @staticmethod
    def process_pdf(file_path: str) -> str:
        """Process PDF files"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error processing PDF file: {e}")
            raise
    
    @staticmethod
    def process_image(file_path: str) -> str:
        """Process image files using OCR"""
        try:
            image = Image.open(file_path)
            
            # Set tesseract path if specified in environment
            tesseract_path = os.getenv('TESSERACT_PATH')
            if tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            logger.error(f"Error processing image file: {e}")
            raise

class DataAnalyzer:
    """Performs statistical analysis and generates insights"""
    
    @staticmethod
    def analyze_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive analysis of DataFrame"""
        analysis = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
        }
        
        # Numerical analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis['numeric_summary'] = df[numeric_cols].describe().to_dict()
            analysis['correlations'] = df[numeric_cols].corr().to_dict()
        
        # Categorical analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            analysis['categorical_summary'] = {}
            for col in categorical_cols:
                analysis['categorical_summary'][col] = {
                    'unique_count': df[col].nunique(),
                    'top_values': df[col].value_counts().head(10).to_dict()
                }
        
        return analysis
    
    @staticmethod
    def analyze_text(text: str) -> Dict[str, Any]:
        """Analyze text content"""
        words = text.split()
        sentences = text.split('.')
        
        analysis = {
            'character_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_words_per_sentence': len(words) / max(len(sentences), 1),
            'most_common_words': {},
        }
        
        # Word frequency analysis
        word_freq = {}
        for word in words:
            word = word.lower().strip('.,!?";')
            if len(word) > 3:  # Filter out short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top 20 most common words
        analysis['most_common_words'] = dict(
            sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        )
        
        return analysis

class VisualizationEngine:
    """Creates various types of visualizations"""
    
    @staticmethod
    def create_correlation_heatmap(df: pd.DataFrame) -> str:
        """Create correlation heatmap for numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return None
        
        fig = px.imshow(
            df[numeric_cols].corr(),
            title="Correlation Heatmap",
            aspect="auto",
            color_continuous_scale="RdBu_r"
        )
        
        fig_path = "correlation_heatmap.html"
        fig.write_html(fig_path)
        return fig_path
    
    @staticmethod
    def create_distribution_plots(df: pd.DataFrame) -> List[str]:
        """Create distribution plots for numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return []
        
        plots = []
        for col in numeric_cols:
            fig = px.histogram(df, x=col, title=f"Distribution of {col}")
            fig_path = f"distribution_{col}.html"
            fig.write_html(fig_path)
            plots.append(fig_path)
        
        return plots
    
    @staticmethod
    def create_categorical_plots(df: pd.DataFrame) -> List[str]:
        """Create plots for categorical columns"""
        categorical_cols = df.select_dtypes(include=['object']).columns
        plots = []
        
        for col in categorical_cols:
            if df[col].nunique() <= 20:  # Only plot if not too many categories
                value_counts = df[col].value_counts()
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Count of {col}"
                )
                fig_path = f"categorical_{col}.html"
                fig.write_html(fig_path)
                plots.append(fig_path)
        
        return plots
    
    @staticmethod
    def create_wordcloud(text: str) -> str:
        """Create word cloud from text"""
        try:
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=100
            ).generate(text)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Word Cloud')
            
            fig_path = "wordcloud.png"
            plt.savefig(fig_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            return fig_path
        except Exception as e:
            logger.error(f"Error creating word cloud: {e}")
            return None

class LLMInterface:
    """Interface for interacting with Together.ai LLM"""
    
    def __init__(self):
        self.api_key = os.getenv('TOGETHER_API_KEY')
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY not found in environment variables")
        
        together.api_key = self.api_key
        self.model = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using the specified LLM"""
        try:
            full_prompt = f"""You are an expert data analyst AI assistant. Analyze the provided data and context, then provide insightful analysis and recommendations.

Context: {context}

User Query: {prompt}

Please provide a comprehensive analysis including:
1. Key insights from the data
2. Patterns or trends observed
3. Recommendations or next steps
4. Any interesting findings

Response:"""

            response = together.Complete.create(
                model=self.model,
                prompt=full_prompt,
                max_tokens=1000,
                temperature=0.7,
                top_p=0.8,
                top_k=50,
                repetition_penalty=1.1
            )
            
            return response['output']['choices'][0]['text'].strip()
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return f"Error generating response: {str(e)}"

class DataAnalystAgent:
    """Main agent class that orchestrates all components"""
    
    def __init__(self):
        self.file_processor = FileProcessor()
        self.data_analyzer = DataAnalyzer()
        self.viz_engine = VisualizationEngine()
        self.llm = LLMInterface()
        self.conversation_history = []
        self.current_data = None
        self.current_analysis = None
    
    def process_file(self, file_path: str) -> AnalysisResult:
        """Process a file and return comprehensive analysis"""
        try:
            file_type = self.file_processor.detect_file_type(file_path)
            logger.info(f"Processing file: {file_path}, Type: {file_type}")
            
            # Process file based on type
            if file_type == FileType.CSV:
                content = self.file_processor.process_csv(file_path)
                analysis = self.data_analyzer.analyze_dataframe(content)
                
                # Create visualizations
                viz_paths = []
                heatmap = self.viz_engine.create_correlation_heatmap(content)
                if heatmap:
                    viz_paths.append(heatmap)
                
                dist_plots = self.viz_engine.create_distribution_plots(content)
                viz_paths.extend(dist_plots)
                
                cat_plots = self.viz_engine.create_categorical_plots(content)
                viz_paths.extend(cat_plots)
                
            elif file_type == FileType.XLSX:
                content = self.file_processor.process_xlsx(file_path)
                analysis = self.data_analyzer.analyze_dataframe(content)
                
                # Create visualizations
                viz_paths = []
                heatmap = self.viz_engine.create_correlation_heatmap(content)
                if heatmap:
                    viz_paths.append(heatmap)
                
                dist_plots = self.viz_engine.create_distribution_plots(content)
                viz_paths.extend(dist_plots)
                
                cat_plots = self.viz_engine.create_categorical_plots(content)
                viz_paths.extend(cat_plots)
                
            elif file_type in [FileType.TXT, FileType.DOC, FileType.DOCX, FileType.PDF, FileType.IMAGE]:
                if file_type == FileType.TXT:
                    content = self.file_processor.process_text(file_path)
                elif file_type == FileType.DOCX:
                    content = self.file_processor.process_docx(file_path)
                elif file_type == FileType.PDF:
                    content = self.file_processor.process_pdf(file_path)
                elif file_type == FileType.IMAGE:
                    content = self.file_processor.process_image(file_path)
                
                analysis = self.data_analyzer.analyze_text(content)
                
                # Create word cloud
                viz_paths = []
                wordcloud_path = self.viz_engine.create_wordcloud(content)
                if wordcloud_path:
                    viz_paths.append(wordcloud_path)
            
            # Store current data for follow-up questions
            self.current_data = content
            self.current_analysis = analysis
            
            # Generate LLM insights
            context = f"File type: {file_type.value}\nAnalysis: {json.dumps(analysis, default=str)}"
            summary = self.llm.generate_response(
                "Provide a comprehensive summary and insights about this data.",
                context
            )
            
            insights = self._extract_insights(analysis, file_type)
            
            result = AnalysisResult(
                file_type=file_type,
                content=content,
                metadata=analysis,
                summary=summary,
                insights=insights,
                visualizations=viz_paths
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            raise
    
    def ask_question(self, question: str) -> str:
        """Ask a follow-up question about the current data"""
        if self.current_data is None:
            return "Please upload and process a file first before asking questions."
        
        # Prepare context from current analysis
        context = f"Current data analysis: {json.dumps(self.current_analysis, default=str)}"
        
        # Add conversation history for context
        if self.conversation_history:
            context += f"\nPrevious conversation: {self.conversation_history[-5:]}"  # Last 5 exchanges
        
        response = self.llm.generate_response(question, context)
        
        # Store in conversation history
        self.conversation_history.append({"question": question, "answer": response})
        
        return response
    
    def _extract_insights(self, analysis: Dict[str, Any], file_type: FileType) -> List[str]:
        """Extract key insights from analysis"""
        insights = []
        
        if file_type in [FileType.CSV, FileType.XLSX]:
            # Data insights
            shape = analysis.get('shape', (0, 0))
            insights.append(f"Dataset contains {shape[0]} rows and {shape[1]} columns")
            
            missing_values = analysis.get('missing_values', {})
            total_missing = sum(missing_values.values())
            if total_missing > 0:
                insights.append(f"Found {total_missing} missing values across the dataset")
            
            numeric_cols = len([k for k, v in analysis.get('dtypes', {}).items() 
                              if 'int' in str(v) or 'float' in str(v)])
            if numeric_cols > 0:
                insights.append(f"Dataset has {numeric_cols} numeric columns suitable for statistical analysis")
        
        elif file_type in [FileType.TXT, FileType.DOC, FileType.DOCX, FileType.PDF, FileType.IMAGE]:
            # Text insights
            word_count = analysis.get('word_count', 0)
            insights.append(f"Document contains {word_count} words")
            
            avg_words = analysis.get('avg_words_per_sentence', 0)
            insights.append(f"Average words per sentence: {avg_words:.1f}")
            
            common_words = analysis.get('most_common_words', {})
            if common_words:
                top_word = max(common_words, key=common_words.get)
                insights.append(f"Most frequent word: '{top_word}' ({common_words[top_word]} occurrences)")
        
        return insights

def main():
    """Main function for command-line interface"""
    print("🤖 Data Analyst Agent")
    print("=" * 50)
    
    agent = DataAnalystAgent()
    
    while True:
        print("\nOptions:")
        print("1. Process a new file")
        print("2. Ask a question about current data")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            file_path = input("Enter file path: ").strip()
            if os.path.exists(file_path):
                try:
                    print("Processing file...")
                    result = agent.process_file(file_path)
                    
                    print(f"\n📊 Analysis Results for {result.file_type.value.upper()} file:")
                    print("-" * 40)
                    print(f"Summary: {result.summary}")
                    
                    print("\n🔍 Key Insights:")
                    for insight in result.insights:
                        print(f"• {insight}")
                    
                    if result.visualizations:
                        print(f"\n📈 Generated {len(result.visualizations)} visualization(s)")
                        print("Visualization files:", result.visualizations)
                    
                except Exception as e:
                    print(f"Error processing file: {e}")
            else:
                print("File not found!")
        
        elif choice == '2':
            question = input("Ask your question: ").strip()
            if question:
                print("Thinking...")
                answer = agent.ask_question(question)
                print(f"\n🤖 Answer: {answer}")
            else:
                print("Please enter a question!")
        
        elif choice == '3':
            print("Goodbye! 👋")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
