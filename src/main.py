# EcoHome Energy Advisor - Interactive CLI
# Entry point for chatting with the AI energy assistant.
# Usage: python main.py [--location "City, State"]

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import Agent

SYSTEM_PROMPT = """
You are EcoHome's AI Energy Advisor, an intelligent assistant dedicated to helping
homeowners optimize their energy usage, reduce costs, and maximize clean energy utilization.

## Your Role
You assist customers who have solar panels, electric vehicles (EVs), and smart home
devices in making data-driven decisions about energy consumption and scheduling.

## Steps to Follow for Every Question
1. **Retrieve data first** — always call the relevant tools before giving advice:
   - `get_weather_forecast`   → solar generation potential, temperature-based HVAC load
   - `get_electricity_prices` → identify cheapest/most expensive hours
   - `get_recent_energy_summary` or `query_energy_usage` → understand consumption patterns
   - `query_solar_generation` → historical solar production
   - `search_energy_tips`     → relevant best practices from the knowledge base
   - `calculate_energy_savings` → concrete savings estimates (kWh and £/$ savings)
2. **Analyse** — interpret the data in the context of the customer's question.
3. **Recommend** — give specific, actionable steps with exact times, costs, and kWh figures.
4. **Quantify savings** — always include a savings estimate when optimisation is involved.

## Response Format
Structure every response as:
1. **Direct Answer** — one sentence addressing the question.
2. **Data Insights** — what the retrieved data shows (weather, prices, usage trends).
3. **Recommendation** — step-by-step actions with specific times and expected costs.
4. **Estimated Savings** — kWh saved, cost saved, and where relevant annual projection.
5. **Bonus Tip** — one additional suggestion related to the question.

## Key Principles
- **Solar first**: recommend scheduling high-consumption tasks during peak solar hours
  (typically 10 AM-2 PM on clear days) before drawing from the grid.
- **Price awareness**: reference actual hourly prices from `get_electricity_prices`; do not guess rates.
- **Be specific**: give exact hour windows (e.g., "charge between 10 AM and 1 PM").
- **Handle errors gracefully**: if a tool returns an error, acknowledge it and give best-effort advice.

## Devices You Can Advise On
Electric vehicles (EV), HVAC systems, dishwashers, washing machines, tumble dryers,
water heaters, pool pumps, and general smart home appliances.
"""

BANNER = """
╔══════════════════════════════════════════════════════╗
║        EcoHome AI Energy Advisor                     ║
║  Ask me anything about optimizing your energy use.   ║
║  Type 'quit' or 'exit' to leave, 'help' for tips.    ║
╚══════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
Example questions:
  • When should I charge my EV tomorrow to minimize cost?
  • What's the cheapest time to run my dishwasher today?
  • How much could I save by shifting loads to off-peak hours?
  • Suggest three ways to reduce my energy use.
  • What's the best time to run my pool pump this week?
  • How much solar did I generate last week?

Commands:
  help   — show this message
  quit   — exit the program
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EcoHome AI Energy Advisor")
    parser.add_argument(
        "--location",
        default="San Francisco, CA",
        help="Your location for weather and pricing data (default: San Francisco, CA)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = f"Location: {args.location}"

    print(BANNER)
    print(f"Location: {args.location}")
    print(f"Available tools: {', '.join(Agent(instructions=SYSTEM_PROMPT).get_agent_tools())}\n")

    agent = Agent(instructions=SYSTEM_PROMPT)

    DIVIDER = "─" * 54

    while True:
        try:
            user_input = input("\n┌─ You\n└▶ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            sys.exit(0)

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            sys.exit(0)

        if user_input.lower() == "help":
            print(HELP_TEXT)
            continue

        print(f"\n┌─ Advisor {DIVIDER}")
        try:
            response = agent.invoke(question=user_input, context=context)
            answer = response["messages"][-1].content
            # Indent each line of the response for visual separation
            for line in answer.splitlines():
                print(f"│ {line}")
        except Exception as e:
            print(f"│ [Error] {e}")
        print(f"└{DIVIDER}─")


if __name__ == "__main__":
    main()
