#!/usr/bin/env python3
"""
Quick Gemini API Test
Simple script to quickly verify Gemini API is working
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ ERROR: GOOGLE_API_KEY not found!")
    print("Add GOOGLE_API_KEY to your .env file")
    exit(1)

print(f"✅ API Key found: {api_key[:10]}...")

# Configure Gemini
genai.configure(api_key=api_key)
print("✅ Gemini configured")

# Initialize model
model = genai.GenerativeModel('gemini-2.5-flash')
print("✅ Model initialized: gemini-2.0-flash-exp")

# Test prompt
prompt = "Say 'Hello, I am Gemini 2.5 Flash and I am working correctly!' in a friendly way."

print(f"\n📝 Testing with prompt: {prompt}")
print("\n⏳ Generating response...\n")

try:
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=200,
            temperature=0.9,
        )
    )
    
    print("="*60)
    print("🤖 GEMINI RESPONSE:")
    print("="*60)
    print(response.text)
    print("="*60)
    print("\n✅ SUCCESS! Gemini is working correctly! 🎉")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    exit(1)
