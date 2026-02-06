#!/usr/bin/env python3
"""
Test script for Gemini 2.5 Flash API integration
Tests the Gemini model response with various prompts
"""

import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime

def test_gemini_basic():
    """Test basic Gemini response"""
    print("\n" + "="*60)
    print("TEST 1: Basic Gemini Response")
    print("="*60)
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        prompt = "What is the capital of France? Answer in one sentence."
        
        print(f"\n📝 Prompt: {prompt}")
        print("\n⏳ Generating response...")
        
        response = model.generate_content(prompt)
        print(f"\n✅ Response: {response.text}")
        
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def test_gemini_with_config():
    """Test Gemini with generation config"""
    print("\n" + "="*60)
    print("TEST 2: Gemini with Generation Config")
    print("="*60)
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = "Explain what machine learning is in simple terms."
        
        print(f"\n📝 Prompt: {prompt}")
        print("\n⏳ Generating response with config (max 200 tokens, temp 0.9)...")
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=200,
                temperature=0.9,
            )
        )
        
        print(f"\n✅ Response: {response.text}")
        
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def test_gemini_complex_prompt():
    """Test Gemini with a complex reasoning prompt"""
    print("\n" + "="*60)
    print("TEST 3: Complex Reasoning Prompt")
    print("="*60)
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        prompt = """Solve this math problem step by step:
        
A store sells apples for $2 each and oranges for $3 each. 
If Sarah buys 5 apples and 3 oranges, how much does she spend in total?"""
        
        print(f"\n📝 Prompt: {prompt}")
        print("\n⏳ Generating response...")
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=300,
                temperature=0.7,
            )
        )
        
        print(f"\n✅ Response: {response.text}")
        
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def test_gemini_safety_settings():
    """Test Gemini with safety settings"""
    print("\n" + "="*60)
    print("TEST 4: Safety Settings Test")
    print("="*60)
    
    try:
        # Configure safety settings
        safety_settings = [
            {
                "category": "HARM_CATEGORY_DANGEROUS",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
        
        model = genai.GenerativeModel(
            'gemini-2.0-flash-exp',
            safety_settings=safety_settings
        )
        
        prompt = "Write a short poem about artificial intelligence."
        
        print(f"\n📝 Prompt: {prompt}")
        print("\n⏳ Generating response with safety settings...")
        
        response = model.generate_content(prompt)
        print(f"\n✅ Response: {response.text}")
        
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def test_model_info():
    """Display available model information"""
    print("\n" + "="*60)
    print("TEST 5: Model Information")
    print("="*60)
    
    try:
        print("\n📋 Available Gemini models:")
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                print(f"  • {model.name}")
                print(f"    - Description: {model.description}")
                print(f"    - Input token limit: {model.input_token_limit}")
                print(f"    - Output token limit: {model.output_token_limit}")
                print()
        
        return True
    except Exception as e:
        print(f"\n❌ Error listing models: {e}")
        return False


def main():
    """Main test runner"""
    print("\n" + "="*60)
    print("🧪 GEMINI 2.5 FLASH API TEST SUITE")
    print("="*60)
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load environment variables
    print("\n📂 Loading environment variables...")
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("\n❌ ERROR: GOOGLE_API_KEY not found in environment variables!")
        print("\n💡 Please add your Google API key to the .env file:")
        print("   GOOGLE_API_KEY=your_api_key_here")
        print("\n🔗 Get your API key from: https://makersuite.google.com/app/apikey")
        sys.exit(1)
    
    print(f"✅ API key found: {api_key[:10]}...{api_key[-5:]}")
    
    # Configure Gemini API
    print("\n🔧 Configuring Gemini API...")
    try:
        genai.configure(api_key=api_key)
        print("✅ Gemini API configured successfully")
    except Exception as e:
        print(f"❌ Failed to configure Gemini API: {e}")
        sys.exit(1)
    
    # Run tests
    results = []
    
    results.append(("Basic Response", test_gemini_basic()))
    results.append(("Generation Config", test_gemini_with_config()))
    results.append(("Complex Reasoning", test_gemini_complex_prompt()))
    results.append(("Safety Settings", test_gemini_safety_settings()))
    results.append(("Model Information", test_model_info()))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✨ All tests passed! Gemini integration is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
