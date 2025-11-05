#!/usr/bin/env python3
"""
Training Knowledge Base with RAG (Retrieval-Augmented Generation)
Uses ChromaDB to store and retrieve training knowledge across 4 collections
"""

import chromadb
from chromadb.api.types import EmbeddingFunction
from typing import List, Dict, Optional
import hashlib
import re


class SimpleEmbeddingFunction(EmbeddingFunction):
    """
    Simple custom embedding function that doesn't require model downloads.
    Uses a combination of character n-grams and word hashing for basic semantic similarity.
    """

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        embeddings = []
        for text in input:
            embeddings.append(self._embed_text(text))
        return embeddings

    def _embed_text(self, text: str, dim: int = 384) -> List[float]:
        """
        Create a simple embedding vector for text.
        Uses word and character n-gram features with hashing.
        """
        text = text.lower()
        words = re.findall(r'\w+', text)

        # Initialize embedding vector
        embedding = [0.0] * dim

        # Add word-level features
        for word in words:
            # Hash each word to multiple dimensions
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = hash_val % dim
            embedding[idx] += 1.0

            # Add character trigrams for partial matching
            for i in range(len(word) - 2):
                trigram = word[i:i+3]
                hash_val = int(hashlib.md5(trigram.encode()).hexdigest(), 16)
                idx = hash_val % dim
                embedding[idx] += 0.3

        # Normalize the embedding vector
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding


class TrainingKnowledgeBase:
    """
    Knowledge base for AI roleplay training with 4 specialized collections:
    1. Company Knowledge - Product/service facts
    2. Policies - What representatives can/cannot do
    3. Customer Behaviors - Realistic customer reaction patterns
    4. Coaching Tips - Best practices for training
    """

    def __init__(self):
        """Initialize ChromaDB client and create collections"""
        self.client = chromadb.Client()

        # Use simple custom embedding function (no external model downloads needed)
        self.embedding_function = SimpleEmbeddingFunction()

        # Create 4 collections for different query types
        self.company_knowledge = self.client.get_or_create_collection(
            "company_knowledge",
            embedding_function=self.embedding_function
        )
        self.policies = self.client.get_or_create_collection(
            "policies",
            embedding_function=self.embedding_function
        )
        self.customer_behaviors = self.client.get_or_create_collection(
            "customer_behaviors",
            embedding_function=self.embedding_function
        )
        self.coaching_tips = self.client.get_or_create_collection(
            "coaching_tips",
            embedding_function=self.embedding_function
        )

        # Load all knowledge into collections
        self._load_all_knowledge()

    def _load_all_knowledge(self):
        """Load all knowledge into collections"""
        # Clear existing data (for fresh loads)
        try:
            self.client.delete_collection("company_knowledge")
            self.client.delete_collection("policies")
            self.client.delete_collection("customer_behaviors")
            self.client.delete_collection("coaching_tips")
        except:
            pass

        # Recreate collections
        self.company_knowledge = self.client.create_collection(
            "company_knowledge",
            embedding_function=self.embedding_function
        )
        self.policies = self.client.create_collection(
            "policies",
            embedding_function=self.embedding_function
        )
        self.customer_behaviors = self.client.create_collection(
            "customer_behaviors",
            embedding_function=self.embedding_function
        )
        self.coaching_tips = self.client.create_collection(
            "coaching_tips",
            embedding_function=self.embedding_function
        )

        # Load data into each collection
        self._load_company_knowledge()
        self._load_policies()
        self._load_customer_behaviors()
        self._load_coaching_tips()

    def _load_company_knowledge(self):
        """
        Product/service information - small factual chunks
        Chunking strategy: One document per atomic fact
        """
        knowledge = [
            # Product details - each benefit is separate for precise retrieval
            {
                "content": "TechFlow Plus Enhancement includes priority 24/7 customer support with dedicated phone line",
                "metadata": {
                    "type": "product_feature",
                    "product": "TechFlow Plus Enhancement",
                    "category": "support",
                    "cost_related": "false",
                },
            },
            {
                "content": "TechFlow Plus Enhancement includes free premium channels: HBO, Showtime, and all Sports packages",
                "metadata": {
                    "type": "product_feature",
                    "product": "TechFlow Plus Enhancement",
                    "category": "entertainment",
                    "cost_related": "false",
                },
            },
            {
                "content": "TechFlow Plus Enhancement provides 50% internet speed boost on top of base plan speed",
                "metadata": {
                    "type": "product_feature",
                    "product": "TechFlow Plus Enhancement",
                    "category": "performance",
                    "cost_related": "false",
                },
            },
            {
                "content": "TechFlow Plus Enhancement eliminates early termination fees if customer wants to cancel service",
                "metadata": {
                    "type": "product_feature",
                    "product": "TechFlow Plus Enhancement",
                    "category": "flexibility",
                    "cost_related": "true",
                },
            },
            {
                "content": "TechFlow Plus Enhancement includes free tech support visits, which normally cost $75 each visit",
                "metadata": {
                    "type": "product_feature",
                    "product": "TechFlow Plus Enhancement",
                    "category": "support",
                    "cost_related": "true",
                },
            },
            # Pricing information
            {
                "content": "TechFlow Plus Enhancement costs $45/month. If purchased separately, these features would cost $89/month, making it a 49% discount",
                "metadata": {
                    "type": "pricing",
                    "product": "TechFlow Plus Enhancement",
                    "category": "cost",
                    "cost_related": "true",
                },
            },
            # Service catalog
            {
                "content": "TechFlow Communications offers high-speed fiber internet with plans ranging from 100Mbps to 1Gbps",
                "metadata": {
                    "type": "service_offering",
                    "service": "internet",
                    "category": "product_catalog",
                    "cost_related": "false",
                },
            },
            {
                "content": "TechFlow Communications provides unlimited local and long-distance calling with all phone service plans",
                "metadata": {
                    "type": "service_offering",
                    "service": "phone",
                    "category": "product_catalog",
                    "cost_related": "false",
                },
            },
            {
                "content": "TechFlow Communications TV service includes 200+ channels including premium networks",
                "metadata": {
                    "type": "service_offering",
                    "service": "tv",
                    "category": "product_catalog",
                    "cost_related": "false",
                },
            },
            {
                "content": "TechFlow Communications offers discounted bundle packages combining 2-3 services (internet, phone, TV)",
                "metadata": {
                    "type": "service_offering",
                    "service": "bundles",
                    "category": "product_catalog",
                    "cost_related": "true",
                },
            },
            # Company background
            {
                "content": "TechFlow Communications was founded in 2018 and serves over 50,000 customers across the metropolitan area",
                "metadata": {
                    "type": "company_background",
                    "category": "about",
                    "cost_related": "false",
                },
            },
            {
                "content": "TechFlow Communications prides itself on reliable service and customer satisfaction as core company values",
                "metadata": {
                    "type": "company_background",
                    "category": "values",
                    "cost_related": "false",
                },
            },
        ]

        for i, item in enumerate(knowledge):
            self.company_knowledge.add(
                documents=[item["content"]],
                metadatas=[item["metadata"]],
                ids=[f"company_{i}"],
            )

    def _load_policies(self):
        """
        Company policies - what reps can/cannot do
        Chunking strategy: One document per policy rule with context
        """
        policies = [
            # Fee removal policies
            {
                "content": "Representatives can remove the TechFlow Plus Enhancement fee. Customer must provide 30-day written notice. The fee will be removed starting the next billing cycle after the notice period.",
                "metadata": {
                    "type": "fee_policy",
                    "action": "removal",
                    "authority_level": "tier1",
                    "requires_approval": "false",
                    "timeframe": "30_days",
                },
            },
            {
                "content": "Representatives can issue a full refund for the current month's TechFlow Plus Enhancement fee if the customer requests removal within 15 days of being billed.",
                "metadata": {
                    "type": "refund_policy",
                    "action": "refund",
                    "authority_level": "tier1",
                    "requires_approval": "false",
                    "timeframe": "15_days",
                    "conditions": "within_15_days_of_billing",
                },
            },
            # Retention offers
            {
                "content": "Representatives can offer a 50% discount on the TechFlow Plus Enhancement fee for up to 3 months as a customer retention incentive. This does not require supervisor approval.",
                "metadata": {
                    "type": "retention_policy",
                    "action": "discount",
                    "authority_level": "tier1",
                    "requires_approval": "false",
                    "max_duration": "3_months",
                    "discount_amount": "50_percent",
                },
            },
            {
                "content": "Representatives can waive up to $100 in fees per customer per year without supervisor approval as a goodwill gesture.",
                "metadata": {
                    "type": "waiver_policy",
                    "action": "waive_fees",
                    "authority_level": "tier1",
                    "requires_approval": "false",
                    "max_amount": "$100",
                    "frequency": "annual",
                },
            },
            # Escalation policies
            {
                "content": "If a customer threatens to cancel their entire service (not just the enhancement), representatives must escalate to a retention specialist. Do not process service cancellations without supervisor involvement.",
                "metadata": {
                    "type": "escalation_policy",
                    "action": "escalate",
                    "authority_level": "tier1",
                    "requires_approval": "true",
                    "trigger": "cancellation_threat",
                },
            },
            {
                "content": "If a customer becomes verbally abusive or uses profanity repeatedly after one warning, representatives should escalate to a supervisor and may end the call if abuse continues.",
                "metadata": {
                    "type": "escalation_policy",
                    "action": "escalate",
                    "authority_level": "tier1",
                    "requires_approval": "true",
                    "trigger": "abusive_behavior",
                },
            },
            # Legal/disclosure policies
            {
                "content": "The TechFlow Plus Enhancement fee is disclosed in Section 12.3 of the standard service agreement. It automatically applies after the initial 2-year contract period expires unless customer opts out.",
                "metadata": {
                    "type": "legal_disclosure",
                    "action": "inform",
                    "authority_level": "tier1",
                    "requires_approval": "false",
                    "contract_section": "12.3",
                },
            },
        ]

        for i, policy in enumerate(policies):
            self.policies.add(
                documents=[policy["content"]],
                metadatas=[policy["metadata"]],
                ids=[f"policy_{i}"],
            )

    def _load_customer_behaviors(self):
        """
        Realistic customer behavior patterns for AI simulation
        Chunking strategy: One document per behavior pattern with context
        """
        behaviors = [
            # Frustration patterns
            {
                "content": "When customers first learn about an unexpected fee, they typically express confusion and mild frustration. They often say things like 'I never agreed to this' or 'I don't remember signing up for that.' Tone is questioning rather than angry.",
                "metadata": {
                    "emotion": "frustrated",
                    "intensity": "mild",
                    "stage": "initial_discovery",
                    "typical_phrases": "never agreed, don't remember, why wasn't I told",
                    "scenario_type": "billing_dispute",
                },
            },
            {
                "content": "If a customer feels dismissed or not heard, their frustration escalates. They may repeat their concern more forcefully, interrupt explanations, or say things like 'You're not listening to me' or 'This is unacceptable.'",
                "metadata": {
                    "emotion": "frustrated",
                    "intensity": "escalating",
                    "stage": "feeling_dismissed",
                    "typical_phrases": "not listening, unacceptable, this is ridiculous",
                    "scenario_type": "billing_dispute",
                },
            },
            # De-escalation patterns
            {
                "content": "When representatives acknowledge the customer's frustration with empathy (e.g., 'I understand why you're upset'), customers often soften their tone and become more collaborative. They shift from 'why did you do this' to 'what can we do about this.'",
                "metadata": {
                    "emotion": "frustrated_to_calm",
                    "intensity": "de-escalating",
                    "stage": "feeling_heard",
                    "trigger": "empathy_shown",
                    "scenario_type": "billing_dispute",
                },
            },
            {
                "content": "When offered a concrete solution or compromise (refund, discount, removal), most customers accept if it addresses their core concern. They may express relief with phrases like 'That works for me' or 'I appreciate you working with me.'",
                "metadata": {
                    "emotion": "satisfied",
                    "intensity": "calm",
                    "stage": "resolution",
                    "trigger": "acceptable_solution",
                    "scenario_type": "billing_dispute",
                },
            },
            # Value recognition patterns
            {
                "content": "When representatives clearly explain the value of a service ($89 worth for $45), some customers shift from wanting removal to wanting to keep it. They ask clarifying questions like 'So I'm actually saving money?' or 'What exactly do I get?'",
                "metadata": {
                    "emotion": "curious",
                    "intensity": "neutral",
                    "stage": "evaluating_value",
                    "trigger": "value_explained",
                    "scenario_type": "billing_dispute",
                },
            },
            # Loyalty customer patterns
            {
                "content": "Long-term customers (2+ years) often mention their loyalty when disputing charges. They say things like 'I've been a customer for X years' or 'I always pay my bills on time.' This is usually a plea for special consideration, not a cancellation threat.",
                "metadata": {
                    "emotion": "frustrated",
                    "intensity": "mild",
                    "stage": "seeking_recognition",
                    "customer_type": "long_term",
                    "typical_phrases": "been a customer for, always paid on time",
                    "scenario_type": "billing_dispute",
                },
            },
            # Cancellation threat patterns
            {
                "content": "When customers threaten cancellation, it's often because they feel they have no other option. They say 'I guess I'll have to cancel' or 'Maybe I should switch providers.' This is usually a last resort, not their preferred outcome. Most want the issue resolved so they can stay.",
                "metadata": {
                    "emotion": "frustrated",
                    "intensity": "high",
                    "stage": "considering_cancellation",
                    "trigger": "no_solution_offered",
                    "typical_phrases": "have to cancel, switch providers, take my business elsewhere",
                    "scenario_type": "billing_dispute",
                },
            },
        ]

        for i, behavior in enumerate(behaviors):
            self.customer_behaviors.add(
                documents=[behavior["content"]],
                metadatas=[behavior["metadata"]],
                ids=[f"behavior_{i}"],
            )

    def _load_coaching_tips(self):
        """
        Customer service best practices for coaching
        Chunking strategy: One document per coaching principle with examples
        """
        tips = [
            # Empathy techniques
            {
                "content": "Start responses by acknowledging the customer's emotion before explaining anything. Say 'I understand this is frustrating' or 'I can see why this would be concerning' before launching into explanations. This shows you're listening.",
                "metadata": {
                    "category": "empathy",
                    "skill": "emotional_acknowledgment",
                    "when_to_use": "customer_shows_emotion",
                    "example_phrases": "I understand, I can see why, That makes sense",
                    "applies_to_scenarios": "billing_dispute, service_issue, general",
                },
            },
            {
                "content": "Use active listening techniques: paraphrase what the customer said back to them. For example: 'So if I understand correctly, you're seeing a $45 charge you didn't expect.' This confirms you heard them correctly.",
                "metadata": {
                    "category": "active_listening",
                    "skill": "paraphrasing",
                    "when_to_use": "after_customer_explains",
                    "example_phrases": "So if I understand, What I'm hearing is, Let me make sure I have this right",
                    "applies_to_scenarios": "billing_dispute, service_issue, general",
                },
            },
            # Problem-solving approach
            {
                "content": "Don't just explain policies - offer solutions immediately. Instead of 'The fee is in your contract,' say 'I can help you remove that fee or I can explain what you're getting for it. Which would you prefer?' Give the customer agency.",
                "metadata": {
                    "category": "problem_solving",
                    "skill": "offering_options",
                    "when_to_use": "customer_disputes_charge",
                    "example_phrases": "I can help you, Here are your options, What would work best for you",
                    "applies_to_scenarios": "billing_dispute, general",
                },
            },
            {
                "content": "Explain value before discussing removal. Help customers make informed decisions by clearly stating what they're getting before offering to remove it. Many will choose to keep the service once they understand the benefits.",
                "metadata": {
                    "category": "problem_solving",
                    "skill": "value_communication",
                    "when_to_use": "customer_wants_to_remove_service",
                    "applies_to_scenarios": "billing_dispute, retention",
                },
            },
            # Communication clarity
            {
                "content": "Avoid jargon and corporate language. Instead of 'per your service agreement Section 12.3,' say 'When you signed up, there was a note about this in the fine print.' Make it conversational and human.",
                "metadata": {
                    "category": "communication",
                    "skill": "plain_language",
                    "when_to_use": "explaining_policies",
                    "avoid_phrases": "per your agreement, as stated in, pursuant to",
                    "prefer_phrases": "when you signed up, here's what happened, let me explain",
                    "applies_to_scenarios": "general",
                },
            },
            {
                "content": "Use 'we' language to show partnership. Say 'Let's figure this out together' or 'We can fix this' instead of 'You need to' or 'I can't.' This creates collaboration rather than confrontation.",
                "metadata": {
                    "category": "communication",
                    "skill": "collaborative_language",
                    "when_to_use": "problem_solving",
                    "example_phrases": "Let's, We can, Together we'll",
                    "avoid_phrases": "You need to, You have to, I can't",
                    "applies_to_scenarios": "general",
                },
            },
            # De-escalation techniques
            {
                "content": "When a customer is escalating, slow down your speech and lower your voice slightly. Match their concern level with calmness, not with urgency. This helps regulate their emotional state.",
                "metadata": {
                    "category": "de_escalation",
                    "skill": "tone_management",
                    "when_to_use": "customer_escalating",
                    "applies_to_scenarios": "billing_dispute, service_issue, general",
                },
            },
            {
                "content": "If a customer feels unheard, stop explaining and ask a question instead. 'What would feel like a fair resolution to you?' or 'What outcome are you hoping for?' This gives them voice and often reveals a simple solution.",
                "metadata": {
                    "category": "de_escalation",
                    "skill": "seeking_customer_input",
                    "when_to_use": "customer_feels_dismissed",
                    "example_phrases": "What would work for you, What are you hoping for, How can I make this right",
                    "applies_to_scenarios": "billing_dispute, service_issue, general",
                },
            },
            # Common mistakes to avoid
            {
                "content": "Never say 'There's nothing I can do' - this ends the conversation. Instead say 'Here's what I CAN do' and list options. Even if limited, showing you tried helps maintain goodwill.",
                "metadata": {
                    "category": "mistakes_to_avoid",
                    "skill": "positive_framing",
                    "when_to_use": "limited_options_available",
                    "avoid_phrases": "nothing I can do, that's impossible, no",
                    "prefer_phrases": "here's what I CAN do, let me see what's possible, I'll try",
                    "applies_to_scenarios": "general",
                },
            },
            {
                "content": "Don't immediately defend the company policy when a customer complains. First validate their frustration, THEN explain the reasoning. 'I completely understand why this is frustrating. Here's why this happened...'",
                "metadata": {
                    "category": "mistakes_to_avoid",
                    "skill": "validation_before_explanation",
                    "when_to_use": "customer_complains_about_policy",
                    "applies_to_scenarios": "billing_dispute, service_issue, general",
                },
            },
        ]

        for i, tip in enumerate(tips):
            self.coaching_tips.add(
                documents=[tip["content"]],
                metadatas=[tip["metadata"]],
                ids=[f"coaching_{i}"],
            )

    # Retrieval Interface Methods

    def retrieve_company_facts(
        self, query: str, n_results: int = 3
    ) -> List[str]:
        """
        For reference questions: 'What does Plus Enhancement include?'

        Args:
            query: The question or search query
            n_results: Number of results to return

        Returns:
            List of relevant company facts
        """
        results = self.company_knowledge.query(
            query_texts=[query], n_results=n_results
        )
        return results["documents"][0] if results["documents"] else []

    def retrieve_policies(
        self, query: str, filter_metadata: Dict = None, n_results: int = 3
    ) -> List[Dict]:
        """
        For validation: 'Can I offer a refund?'
        Returns both content and metadata

        Args:
            query: The policy question or search query
            filter_metadata: Optional metadata filters (e.g., {"authority_level": "tier1"})
            n_results: Number of results to return

        Returns:
            List of dicts with 'content' and 'metadata' keys
        """
        kwargs = {"query_texts": [query], "n_results": n_results}
        if filter_metadata:
            kwargs["where"] = filter_metadata

        results = self.policies.query(**kwargs)

        if not results["documents"]:
            return []

        # Return structured results with metadata
        return [
            {"content": doc, "metadata": meta}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]

    def retrieve_customer_behaviors(
        self, emotion: str, intensity: str = None
    ) -> List[str]:
        """
        For customer simulation: 'How does frustrated customer act?'

        Args:
            emotion: The emotion to search for (frustrated, satisfied, curious, etc.)
            intensity: Optional intensity filter (mild, escalating, high, etc.)

        Returns:
            List of relevant behavior patterns
        """
        # Build where filter using ChromaDB's $and operator for multiple conditions
        if intensity:
            where_filter = {
                "$and": [
                    {"emotion": emotion},
                    {"intensity": intensity}
                ]
            }
        else:
            where_filter = {"emotion": emotion}

        results = self.customer_behaviors.query(
            query_texts=[f"customer behavior {emotion}"],
            n_results=5,
            where=where_filter,
        )
        return results["documents"][0] if results["documents"] else []

    def retrieve_coaching_tips(
        self, situation: str, category: str = None, n_results: int = 3
    ) -> List[Dict]:
        """
        For coaching: 'How to handle escalating customer?'

        Args:
            situation: The coaching situation or question
            category: Optional category filter (empathy, de_escalation, etc.)

        Returns:
            List of dicts with 'content' and 'metadata' keys
        """
        kwargs = {"query_texts": [situation], "n_results": n_results}
        if category:
            kwargs["where"] = {"category": category}

        results = self.coaching_tips.query(**kwargs)

        if not results["documents"]:
            return []

        return [
            {"content": doc, "metadata": meta}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]

    def get_scenario_briefing(self) -> Dict[str, List[str]]:
        """
        Generate comprehensive scenario briefing using RAG retrieval

        Returns:
            Dict with keys: company_info, product_features, policies, tips
        """
        briefing = {
            "company_info": self.retrieve_company_facts(
                "TechFlow Communications company background services", n_results=5
            ),
            "product_features": self.retrieve_company_facts(
                "TechFlow Plus Enhancement benefits features pricing", n_results=6
            ),
            "policies": self.retrieve_policies(
                "representative authority removal refund retention", n_results=5
            ),
            "coaching_tips": self.retrieve_coaching_tips(
                "customer service best practices empathy communication", n_results=4
            ),
        }
        return briefing
