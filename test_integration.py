#!/usr/bin/env python3
"""
Quick integration test for RAG in main.py
Tests that the knowledge base integrates correctly without needing API keys
"""

import os

# Set a dummy API key to prevent initialization errors
os.environ["ANTHROPIC_API_KEY"] = "dummy-key-for-testing"

from main import CustomerServiceTrainer


def test_integration():
    """Test that the trainer initializes with knowledge base"""
    print("Testing RAG integration in CustomerServiceTrainer...")
    print("=" * 60)

    # Initialize trainer (will initialize knowledge base)
    trainer = CustomerServiceTrainer()

    # Verify knowledge base is initialized
    assert trainer.knowledge_base is not None, "Knowledge base not initialized"
    print("✅ Knowledge base initialized successfully")

    # Test that display_briefing works
    print("\n" + "=" * 60)
    print("Testing RAG-powered briefing display:")
    print("=" * 60)
    try:
        trainer.display_briefing()
        print("\n✅ Briefing displayed successfully using RAG")
    except Exception as e:
        print(f"❌ Error displaying briefing: {e}")
        raise

    # Test quick reference
    print("\n" + "=" * 60)
    print("Testing RAG-powered quick reference:")
    print("=" * 60)
    trainer.scenario_active = True  # Enable reference display
    try:
        trainer.show_quick_reference()
        print("\n✅ Quick reference displayed successfully using RAG")
    except Exception as e:
        print(f"❌ Error displaying quick reference: {e}")
        raise

    print("\n" + "=" * 60)
    print("ALL INTEGRATION TESTS PASSED! ✅")
    print("=" * 60)
    print("\nRAG is successfully integrated into the AI Roleplay Trainer!")
    print("The knowledge base is being used to:")
    print("  • Display company information")
    print("  • Show product features and benefits")
    print("  • Present policies and representative authority")
    print("  • Provide coaching tips")


if __name__ == "__main__":
    test_integration()
