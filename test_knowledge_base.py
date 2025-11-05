#!/usr/bin/env python3
"""
Tests for TrainingKnowledgeBase RAG retrieval quality
"""

from knowledge_base import TrainingKnowledgeBase


def test_retrieval_quality():
    """Test that RAG retrieval returns relevant results"""
    print("Initializing knowledge base...")
    kb = TrainingKnowledgeBase()
    print("✅ Knowledge base initialized\n")

    # Test 1: Can we find the right policy?
    print("Test 1: Policy retrieval")
    results = kb.retrieve_policies("can I give customer refund")
    assert len(results) > 0, "No policy results found"
    assert any(
        "refund" in r["content"].lower() for r in results
    ), "No refund policy found"
    print("✅ Policy retrieval works")
    print(f"   Found {len(results)} relevant policies")
    print(f"   Sample: {results[0]['content'][:100]}...\n")

    # Test 2: Does metadata filtering work?
    print("Test 2: Metadata filtering on policies")
    results = kb.retrieve_policies(
        "what can representatives do", filter_metadata={"requires_approval": "false"}
    )
    assert len(results) > 0, "No filtered results found"
    assert all(
        r["metadata"]["requires_approval"] == "false" for r in results
    ), "Filtering failed"
    print("✅ Metadata filtering works")
    print(f"   Found {len(results)} policies not requiring approval\n")

    # Test 3: Customer behavior quality
    print("Test 3: Customer behavior retrieval")
    behaviors = kb.retrieve_customer_behaviors("frustrated", "mild")
    assert len(behaviors) > 0, "No behaviors found"
    print("✅ Behavior retrieval works")
    print(f"   Found {len(behaviors)} behavior patterns")
    print(f"   Sample: {behaviors[0][:100]}...\n")

    # Test 4: Company knowledge retrieval
    print("Test 4: Company knowledge retrieval")
    facts = kb.retrieve_company_facts(
        "TechFlow Plus Enhancement benefits features"
    )
    assert len(facts) > 0, "No company facts found"
    print("✅ Company knowledge retrieval works")
    print(f"   Found {len(facts)} relevant facts")
    print(f"   Sample: {facts[0][:100]}...\n")

    # Test 5: Coaching tips retrieval
    print("Test 5: Coaching tips retrieval")
    tips = kb.retrieve_coaching_tips(
        "customer feels dismissed and getting angry", category="de_escalation"
    )
    assert len(tips) > 0, "No coaching tips found"
    print("✅ Coaching tips retrieval works")
    print(f"   Found {len(tips)} coaching tips")
    print(f"   Sample: {tips[0]['content'][:100]}...\n")

    # Test 6: Scenario briefing generation
    print("Test 6: Scenario briefing generation")
    briefing = kb.get_scenario_briefing()
    assert "company_info" in briefing, "Missing company_info"
    assert "product_features" in briefing, "Missing product_features"
    assert "policies" in briefing, "Missing policies"
    assert "coaching_tips" in briefing, "Missing coaching_tips"
    print("✅ Scenario briefing generation works")
    print(f"   Company info: {len(briefing['company_info'])} items")
    print(f"   Product features: {len(briefing['product_features'])} items")
    print(f"   Policies: {len(briefing['policies'])} items")
    print(f"   Coaching tips: {len(briefing['coaching_tips'])} items\n")

    print("=" * 60)
    print("ALL TESTS PASSED! ✅")
    print("=" * 60)

    # Show sample retrieved content
    print("\n📊 Sample Retrieved Policy:")
    print("-" * 60)
    print(results[0]["content"])
    print("\nMetadata:", results[0]["metadata"])


def test_specific_queries():
    """Test specific use cases"""
    print("\n" + "=" * 60)
    print("SPECIFIC USE CASE TESTS")
    print("=" * 60 + "\n")

    kb = TrainingKnowledgeBase()

    # Use case 1: What can tier1 rep do without approval?
    print("Use Case 1: What can tier1 rep do without approval?")
    policies = kb.retrieve_policies(
        "representative authority",
        filter_metadata={
            "$and": [
                {"authority_level": "tier1"},
                {"requires_approval": "false"}
            ]
        },
    )
    print(f"Found {len(policies)} policies:")
    for i, policy in enumerate(policies[:3], 1):
        print(f"{i}. {policy['metadata']['action']}: {policy['content'][:80]}...")
    print()

    # Use case 2: How do customers react when they feel heard?
    print("Use Case 2: How do customers react when they feel heard?")
    behaviors = kb.retrieve_customer_behaviors("frustrated_to_calm")
    print(f"Found {len(behaviors)} behaviors:")
    for behavior in behaviors[:2]:
        print(f"- {behavior[:100]}...")
    print()

    # Use case 3: What's included in TechFlow Plus?
    print("Use Case 3: What's included in TechFlow Plus Enhancement?")
    features = kb.retrieve_company_facts("TechFlow Plus Enhancement features", n_results=5)
    print(f"Found {len(features)} features:")
    for feature in features:
        print(f"- {feature}")
    print()

    # Use case 4: Best practices for empathy
    print("Use Case 4: Best practices for showing empathy")
    tips = kb.retrieve_coaching_tips("showing empathy to customer", category="empathy")
    print(f"Found {len(tips)} tips:")
    for tip in tips:
        print(f"- {tip['content'][:120]}...")
    print()


if __name__ == "__main__":
    test_retrieval_quality()
    test_specific_queries()
