#!/usr/bin/env python3
"""
Customer Service Training CLI Prototype
A simple CLI tool to practice customer service skills with AI roleplay
"""

import os
import argparse
from llm_providers import create_provider
from knowledge_base import TrainingKnowledgeBase


class CustomerServiceTrainer:
    def __init__(self, provider=None):
        self.conversation_history = []
        self.scenario_active = False
        self.coaching_enabled = False  # Coach starts disabled

        # Initialize LLM provider
        self.llm_provider = create_provider(provider)

        # Initialize RAG knowledge base
        print("Initializing knowledge base...")
        self.knowledge_base = TrainingKnowledgeBase()
        print("Knowledge base ready!")

        # Scenario setup with comprehensive briefing
        self.scenario = {
            "title": "Billing Dispute - Service Enhancement Fee",
            "company_briefing": {
                "company_name": "TechFlow Communications",
                "your_role": "Customer Service Representative - Tier 1 Support",
                "company_overview": """
                    TechFlow Communications provides internet, phone, and TV bundle services to residential customers.
                    We pride ourselves on reliable service and customer satisfaction.
                    Founded in 2018, we serve over 50,000 customers across the metropolitan area.
                """,
                "services": {
                    "internet": "High-speed fiber internet (100Mbps - 1Gbps plans)",
                    "phone": "Unlimited local and long-distance calling",
                    "tv": "200+ channels including premium networks",
                    "bundles": "Discounted packages combining 2-3 services",
                },
                "service_enhancement_package": {
                    "name": "TechFlow Plus Enhancement",
                    "cost": "$45/month",
                    "when_applied": "Automatically after initial 2-year contract expires",
                    "disclosure": "Mentioned in original contract fine print (Section 12.3)",
                    "benefits": [
                        "Priority customer support (24/7 dedicated line)",
                        "Free premium channels (HBO, Showtime, Sports packages)",
                        "Internet speed boost (+50% faster)",
                        "Free tech support visits (normally $75 each)",
                        "No early termination fees if you want to cancel service",
                    ],
                    "value": "Regular price would be $89/month for these features separately",
                },
                "policies": {
                    "fee_removal": "Enhancement can be removed with 30-day written notice",
                    "refunds": "Can refund current month if removed within 15 days of billing",
                    "escalation": "Escalate to supervisor if customer requests cancellation of entire service",
                    "retention_offers": "Can offer 50% discount on enhancement fee for 3 months as retention",
                },
            },
            "customer_background": {
                "name": "Sarah Chen",
                "account_details": {
                    "customer_since": "March 2023 (2 years)",
                    "services": "Internet + Phone bundle",
                    "payment_history": "Always pays on time",
                    "previous_contacts": "Called once in 2023 about internet outage - resolved quickly",
                },
                "current_situation": {
                    "issue": "Unexpected $45 Service Enhancement Fee on latest bill",
                    "customer_knowledge": "Doesn't remember agreeing to this",
                    "desired_outcome": "Understand the charge and potentially remove it",
                    "mood": "Mildly frustrated but reasonable",
                },
            },
            "success_criteria": {
                "ideal_outcome": "Customer understands the value, keeps the enhancement",
                "acceptable_outcome": "Customer removes enhancement but stays satisfied",
                "escalation_needed": "If customer threatens to cancel entire service",
            },
            "customer_prompt": """
                You are Sarah Chen, a customer calling about a billing issue. You have these characteristics:

                PERSONALITY & TONE:
                - Mildly frustrated but reasonable
                - Direct communicator who wants clear answers
                - Willing to work with customer service if treated respectfully
                - Gets more frustrated if you feel dismissed or not heard

                YOUR SITUATION:
                - You've been a loyal customer for 2 years
                - You just received your bill with an unexpected $45 "Service Enhancement Fee"
                - You never signed up for any service enhancement
                - You want to understand what this charge is and get it removed if it's a mistake

                CONVERSATION RULES:
                - Stay in character as Sarah Chen throughout the entire conversation
                - Only discuss this billing issue - if the human asks about anything else (travel, other topics, personal advice), respond with something like "I'm sorry, but I'm calling about my billing issue. Can we please focus on resolving this charge?"
                - Be persistent about getting answers but remain civil
                - Show appreciation when the representative is helpful
                - Your goal is to understand the charge and get it resolved

                IMPORTANT: You are ONLY Sarah Chen calling about a billing issue. Do not break character or discuss anything unrelated to customer service.

                Start the conversation by explaining your billing concern.
            """,
        }

    def make_api_call(self, messages, max_tokens=1000):
        """Make a call to the configured LLM provider"""
        return self.llm_provider.make_call(messages, max_tokens)

    def display_briefing(self):
        """Display comprehensive scenario briefing using RAG retrieval"""
        briefing = self.scenario["company_briefing"]
        customer = self.scenario["customer_background"]

        print("=" * 80)
        print(f"SCENARIO BRIEFING: {self.scenario['title'].upper()}")
        print("=" * 80)

        # Company Overview
        print(f"\n🏢 COMPANY: {briefing['company_name']}")
        print(f"YOUR ROLE: {briefing['your_role']}")

        # Retrieve company info from knowledge base
        company_info = self.knowledge_base.retrieve_company_facts(
            "TechFlow Communications company background", n_results=2
        )
        for info in company_info:
            print(f"  • {info}")

        # Services Overview - Retrieved from KB
        print(f"\n📋 OUR SERVICES:")
        services_info = self.knowledge_base.retrieve_company_facts(
            "TechFlow Communications services internet phone TV bundles", n_results=4
        )
        for service in services_info:
            print(f"  • {service}")

        # The Specific Issue - Retrieved from KB
        print(f"\n⚠️  THE SERVICE ENHANCEMENT FEE:")
        enhancement_info = self.knowledge_base.retrieve_company_facts(
            "TechFlow Plus Enhancement pricing cost value", n_results=1
        )
        if enhancement_info:
            print(f"  • {enhancement_info[0]}")

        # Enhancement Benefits - Retrieved from KB
        print(f"\n✨ ENHANCEMENT BENEFITS:")
        benefits = self.knowledge_base.retrieve_company_facts(
            "TechFlow Plus Enhancement features benefits support channels speed", n_results=5
        )
        for benefit in benefits:
            print(f"  • {benefit}")

        # Your Policies & Powers - Retrieved from KB
        print(f"\n📋 YOUR POLICIES & AUTHORITY:")
        policies = self.knowledge_base.retrieve_policies(
            "representative authority removal refund retention escalation", n_results=5
        )
        for policy in policies:
            # Extract the key action from metadata
            action = policy['metadata'].get('action', 'policy')
            print(f"  • {action.title()}: {policy['content'][:100]}...")

        # Customer Details
        print(f"\n👤 CUSTOMER: {customer['name']}")
        account = customer["account_details"]
        situation = customer["current_situation"]

        print(f"  • Account: Customer since {account['customer_since']}")
        print(f"  • Services: {account['services']}")
        print(
            f"  • History: {account['payment_history']}, {account['previous_contacts']}"
        )
        print(f"  • Issue: {situation['issue']}")
        print(f"  • Goal: {situation['desired_outcome']}")
        print(f"  • Mood: {situation['mood']}")

        # Success Metrics
        print(f"\n🎯 SUCCESS OUTCOMES:")
        success = self.scenario["success_criteria"]
        print(f"  • Ideal: {success['ideal_outcome']}")
        print(f"  • Acceptable: {success['acceptable_outcome']}")
        print(f"  • Escalate if: {success['escalation_needed']}")

        # Coaching Tips - Retrieved from KB
        print(f"\n💡 COACHING TIPS:")
        tips = self.knowledge_base.retrieve_coaching_tips(
            "customer service best practices empathy communication", n_results=3
        )
        for tip in tips[:3]:
            print(f"  • {tip['content'][:100]}...")

        print("\n" + "=" * 80)
        print("Take time to review this information before starting the roleplay.")
        print("You can reference this briefing during the conversation if needed.")
        print("=" * 80)

    def start_scenario(self):
        """Initialize the customer service scenario with briefing"""
        self.display_briefing()

        print(f"\nPress Enter when ready to start the roleplay...")
        input()

        # Initialize conversation with customer - enhanced prompt with business context
        enhanced_customer_prompt = f"""
            You are Sarah Chen, calling TechFlow Communications about a billing issue. Context:

            PERSONALITY & TONE:
            - Mildly frustrated but reasonable
            - Direct communicator who wants clear answers
            - Willing to work with customer service if treated respectfully
            - Gets more frustrated if you feel dismissed or not heard

            YOUR SITUATION:
            - You've been a customer since March 2023 (2 years)
            - You have Internet + Phone bundle service
            - You always pay your bills on time
            - You just received your bill with an unexpected $45 "TechFlow Plus Enhancement" fee
            - You don't remember signing up for any enhancement
            - You want to understand what this charge is and get it removed if it's a mistake

            CONVERSATION RULES:
            - Stay in character as Sarah Chen throughout the entire conversation
            - Only discuss this billing issue - if asked about anything else, redirect back to billing
            - Be persistent about getting answers but remain civil
            - Show appreciation when the representative is helpful
            - Your goal is to understand the charge and get it resolved

            IMPORTANT: You are ONLY Sarah Chen calling about the billing issue. Stay in character.

            Start by explaining your billing concern.
        """

        initial_message = [{"role": "user", "content": enhanced_customer_prompt}]
        customer_response = self.make_api_call(initial_message)

        print("\n" + "=" * 60)
        print("CUSTOMER CALLING...")
        print("=" * 60)
        print(f"Customer: {customer_response}")

        # Store the conversation
        self.conversation_history = [
            {"role": "user", "content": enhanced_customer_prompt},
            {"role": "assistant", "content": customer_response},
        ]

        self.scenario_active = True
        return customer_response

    def analyze_conversation_for_coaching(self):
        """Analyze recent conversation to provide coaching hints"""
        if not self.conversation_history or len(self.conversation_history) < 3:
            return None

        # Get the last few exchanges for context
        recent_conversation = self.conversation_history[-4:]  # Last 2 exchanges

        coaching_prompt = f"""
            Analyze this customer service conversation and provide a brief coaching hint for the representative:

            RECENT CONVERSATION:
            {self.format_recent_conversation(recent_conversation)}

            CONTEXT: This is a billing dispute where the customer (Sarah Chen) is upset about an unexpected $45 charge.

            Provide ONE specific coaching hint in 1-2 sentences that would help the representative improve their next response. Focus on:
            - Empathy and acknowledgment
            - Active listening 
            - Addressing customer's actual concerns
            - Professional problem-solving approach

            Format: Just the coaching advice, no extra text. Keep it concise and actionable.
        """

        coaching_messages = [{"role": "user", "content": coaching_prompt}]
        coaching_hint = self.make_api_call(coaching_messages, max_tokens=200)

        return coaching_hint

    def format_recent_conversation(self, recent_messages):
        """Format recent conversation for coaching analysis"""
        formatted = []
        is_customer = True  # Start with customer (skip system prompt)

        for msg in recent_messages:
            if msg["role"] == "assistant":
                formatted.append(f"Customer: {msg['content']}")
            elif "Continue playing Sarah Chen" in msg["content"]:
                # Extract the actual representative response from the prompt
                content = msg["content"]
                if 'just said: "' in content:
                    rep_response = content.split('just said: "')[1].split('"')[0]
                    formatted.append(f"Representative: {rep_response}")

        return "\n\n".join(formatted[-4:])  # Last 2 exchanges

    def handle_user_response(self, user_input):
        """Process user's customer service response"""
        if not self.scenario_active:
            return "Please start a scenario first."

        # Add user response to conversation
        self.conversation_history.append(
            {
                "role": "user",
                "content": f"""
                    Continue playing Sarah Chen, the customer with the billing issue. 
                    The customer service representative just said: "{user_input}"

                    Remember:
                    - Stay in character as Sarah Chen
                    - Focus only on the billing dispute
                    - If they ask about anything unrelated, redirect back to your billing issue
                    - React appropriately to their response (appreciative if helpful, more frustrated if dismissed)
                """,
            }
        )

        # Get AI customer response
        customer_response = self.make_api_call(self.conversation_history)

        # Add AI response to history
        self.conversation_history.append(
            {"role": "assistant", "content": customer_response}
        )

        return customer_response

    def show_quick_reference(self):
        """Show condensed reference during active scenario using RAG"""
        if not self.scenario_active:
            print("No active scenario. Start a scenario first.")
            return

        briefing = self.scenario["company_briefing"]
        print("\n" + "=" * 50)
        print("QUICK REFERENCE")
        print("=" * 50)
        print(f"Company: {briefing['company_name']}")

        # Get enhancement info from KB
        enhancement_info = self.knowledge_base.retrieve_company_facts(
            "TechFlow Plus Enhancement cost pricing", n_results=1
        )
        if enhancement_info:
            print(f"Enhancement: {enhancement_info[0]}")

        # Get key policies from KB
        print("\nKey Policies:")
        policies = self.knowledge_base.retrieve_policies(
            "removal refund retention", n_results=3
        )
        for policy in policies:
            action = policy['metadata'].get('action', 'policy')
            print(f"  • {action.title()}: {policy['content'][:80]}...")

        print("=" * 50)

    def toggle_coaching(self):
        """Toggle coaching on/off"""
        self.coaching_enabled = not self.coaching_enabled
        status = "ENABLED" if self.coaching_enabled else "DISABLED"
        print(f"\n🎯 Coaching is now {status}")
        if self.coaching_enabled:
            print("You'll receive coaching hints after customer responses.")
        else:
            print("Coaching hints are turned off. Use 'coach' to re-enable.")

    def show_coaching_hint(self):
        """Display coaching hint if enabled and scenario is active"""
        if not self.coaching_enabled or not self.scenario_active:
            return

        print("\n🎯 Getting coaching hint...")
        coaching_hint = self.analyze_conversation_for_coaching()

        if coaching_hint:
            print(f"💡 COACH HINT: {coaching_hint.strip()}")
        else:
            print("💡 COACH: Keep the conversation going!")
        print()

    def end_scenario_with_feedback(self):
        """Generate feedback on the user's performance"""
        if not self.conversation_history:
            return "No conversation to analyze."

        # Create feedback prompt
        feedback_prompt = f"""
            Analyze this customer service conversation and provide constructive feedback:

            CONVERSATION:
            {self.format_conversation_for_review()}

            Please provide feedback on the customer service representative's performance:

            1. STRENGTHS: What did they do well?
            2. AREAS FOR IMPROVEMENT: What could they have done better?
            3. SPECIFIC SUGGESTIONS: Concrete advice for handling similar situations
            4. OVERALL RATING: Rate their performance from 1-5 with brief explanation

            Focus on:
            - Empathy and active listening
            - Problem-solving approach
            - Communication clarity
            - De-escalation techniques
            - Professional tone

            Keep feedback constructive and specific.
        """

        feedback_messages = [{"role": "user", "content": feedback_prompt}]
        feedback = self.make_api_call(feedback_messages, max_tokens=1500)

        return feedback

    def format_conversation_for_review(self):
        """Format conversation history for feedback analysis"""
        formatted = []
        is_customer = True  # First message is always customer

        for i, msg in enumerate(self.conversation_history[1:], 1):  # Skip system prompt
            if is_customer:
                formatted.append(f"Customer: {msg['content']}")
            else:
                formatted.append(f"Representative: {msg['content']}")
            is_customer = not is_customer  # Alternate between customer and rep

        return "\n\n".join(formatted)

    def run(self):
        """Main CLI loop with enhanced commands"""
        provider_name = type(self.llm_provider).__name__.replace("Provider", "")
        print(f"Welcome to Customer Service Training! (Using {provider_name} API)")
        print(
            "Commands: 'start' (scenario), 'end' (feedback), 'ref' (reference), 'coach' (toggle coaching), 'quit'"
        )

        while True:
            if self.scenario_active:
                coach_status = "🎯ON" if self.coaching_enabled else "OFF"
                command = (
                    input(f"\n[In Call - Coach:{coach_status}] > ").strip().lower()
                )
            else:
                command = input("\n> ").strip().lower()

            if command == "quit":
                print("Thanks for training! Goodbye.")
                break

            elif command == "start":
                if self.scenario_active:
                    print(
                        "Scenario already active. Type 'end' to finish current scenario first."
                    )
                    continue
                self.start_scenario()

            elif command == "end":
                if not self.scenario_active:
                    print("No active scenario to end.")
                    continue

                print("\nGenerating feedback on your performance...")
                feedback = self.end_scenario_with_feedback()
                print("\n" + "=" * 60)
                print("PERFORMANCE FEEDBACK")
                print("=" * 60)
                print(feedback)

                # Reset for next scenario
                self.conversation_history = []
                self.scenario_active = False
                print(
                    "\nScenario complete! Type 'start' to try again or 'quit' to exit."
                )

            elif command == "ref" or command == "reference":
                self.show_quick_reference()

            elif command == "coach":
                self.toggle_coaching()

            elif command.startswith("help"):
                print("\nCommands:")
                print("- start: Begin customer service roleplay")
                print("- end: Finish scenario and get feedback")
                print("- ref: Show quick reference during calls")
                print("- coach: Toggle coaching hints on/off")
                print("- quit: Exit the program")
                print("- During roleplay: Type your customer service responses")

            elif self.scenario_active:
                # User is responding to customer during active scenario
                print("\nProcessing your response...")
                customer_response = self.handle_user_response(command)
                print(f"\nCustomer: {customer_response}")

                # Show coaching hint if enabled
                self.show_coaching_hint()

                print("\nYour response (or 'ref'/'coach' for help):")

            else:
                print("Unknown command. Type 'help' for available commands.")


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description="AI-Powered Customer Service Training")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        help="LLM provider to use (default: anthropic, or set LLM_PROVIDER env var)",
    )

    args = parser.parse_args()

    try:
        trainer = CustomerServiceTrainer(provider=args.provider)
        trainer.run()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("Available providers: openai, anthropic")
        print("Set LLM_PROVIDER environment variable or use --provider flag")
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
