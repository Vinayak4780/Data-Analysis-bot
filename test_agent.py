"""
Test script for the Data Analyst Agent
This script tests various functionalities of the agent with sample data
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add the project directory to the path
project_dir = Path(__file__).parent
sys.path.append(str(project_dir))

try:
    from data_analyst_agent import DataAnalystAgent, FileType
    print("✅ Successfully imported DataAnalystAgent")
except ImportError as e:
    print(f"❌ Failed to import DataAnalystAgent: {e}")
    sys.exit(1)

def test_csv_processing():
    """Test CSV file processing"""
    print("\n🧪 Testing CSV Processing...")
    
    # Create a simple test CSV
    test_data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
        'Age': [25, 30, 35, 28],
        'Salary': [50000, 60000, 70000, 55000],
        'Department': ['IT', 'Sales', 'IT', 'Marketing']
    }
    
    df = pd.DataFrame(test_data)
    test_file = project_dir / "test_data.csv"
    df.to_csv(test_file, index=False)
    
    try:
        agent = DataAnalystAgent()
        result = agent.process_file(str(test_file))
        
        print(f"  ✅ File type detected: {result.file_type}")
        print(f"  ✅ Generated {len(result.insights)} insights")
        print(f"  ✅ Created {len(result.visualizations)} visualizations")
        print(f"  ✅ Summary length: {len(result.summary)} characters")
        
        # Test Q&A
        answer = agent.ask_question("What is the average salary?")
        print(f"  ✅ Q&A working: {len(answer)} character response")
        
        # Clean up
        os.unlink(test_file)
        
        return True
        
    except Exception as e:
        print(f"  ❌ CSV processing failed: {e}")
        return False

def test_text_processing():
    """Test text file processing"""
    print("\n🧪 Testing Text Processing...")
    
    test_text = """
    This is a sample text document for testing purposes.
    It contains multiple sentences and various words.
    The data analyst agent should be able to analyze this text
    and provide meaningful insights about word frequency,
    sentence structure, and overall content analysis.
    """
    
    test_file = project_dir / "test_text.txt"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_text)
    
    try:
        agent = DataAnalystAgent()
        result = agent.process_file(str(test_file))
        
        print(f"  ✅ File type detected: {result.file_type}")
        print(f"  ✅ Generated {len(result.insights)} insights")
        print(f"  ✅ Word count analysis available")
        
        # Clean up
        os.unlink(test_file)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Text processing failed: {e}")
        return False

def test_agent_initialization():
    """Test agent initialization"""
    print("\n🧪 Testing Agent Initialization...")
    
    try:
        agent = DataAnalystAgent()
        print("  ✅ Agent initialized successfully")
        
        # Test components
        if hasattr(agent, 'file_processor'):
            print("  ✅ File processor available")
        if hasattr(agent, 'data_analyzer'):
            print("  ✅ Data analyzer available")
        if hasattr(agent, 'viz_engine'):
            print("  ✅ Visualization engine available")
        if hasattr(agent, 'llm'):
            print("  ✅ LLM interface available")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Agent initialization failed: {e}")
        print("  💡 Make sure TOGETHER_API_KEY is set in .env file")
        return False

def test_file_type_detection():
    """Test file type detection"""
    print("\n🧪 Testing File Type Detection...")
    
    from data_analyst_agent import FileProcessor
    
    test_cases = [
        ("test.csv", FileType.CSV),
        ("test.xlsx", FileType.XLSX),
        ("test.txt", FileType.TXT),
        ("test.docx", FileType.DOCX),
        ("test.pdf", FileType.PDF),
        ("test.jpg", FileType.IMAGE),
        ("test.png", FileType.IMAGE),
    ]
    
    processor = FileProcessor()
    all_passed = True
    
    for filename, expected_type in test_cases:
        detected_type = processor.detect_file_type(filename)
        if detected_type == expected_type:
            print(f"  ✅ {filename} -> {detected_type.value}")
        else:
            print(f"  ❌ {filename} -> Expected {expected_type.value}, got {detected_type.value}")
            all_passed = False
    
    return all_passed

def run_all_tests():
    """Run all tests"""
    print("🧪 Data Analyst Agent Test Suite")
    print("=" * 50)
    
    tests = [
        ("Agent Initialization", test_agent_initialization),
        ("File Type Detection", test_file_type_detection),
        ("CSV Processing", test_csv_processing),
        ("Text Processing", test_text_processing),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("🧪 Test Results Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The agent is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    run_all_tests()
