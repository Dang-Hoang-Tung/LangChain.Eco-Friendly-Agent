# EcoHome Energy Advisor - Agent Run & Evaluation
#
# Runs the Energy Advisor agent with real-world scenarios and evaluates:
#   - Accuracy      : Correct information and calculations
#   - Relevance     : Responses address the user's question
#   - Completeness  : Comprehensive answers with actionable advice
#   - Tool Usage    : Appropriate use of available tools
#   - Reasoning     : Clear explanation of recommendations

# =============================================================================
# 1. Import and Initialize
# =============================================================================
import json
import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import Agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

ECOHOME_SYSTEM_PROMPT = """
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
- **Solar first**: always recommend scheduling high-consumption tasks during peak solar hours
  (typically 10 AM-2 PM on clear days) before drawing from the grid.
- **Price awareness**: reference actual hourly prices from `get_electricity_prices` when
  recommending off-peak scheduling; do not guess rates.
- **Be specific**: give exact hour windows (e.g., "charge between 10 AM and 1 PM"),
  not vague advice like "charge during the day".
- **Handle errors gracefully**: if a tool returns an error, acknowledge it and provide
  the best advice possible using remaining data.

## Devices You Can Advise On
Electric vehicles (EV), HVAC systems (heating/cooling), dishwashers, washing machines,
tumble dryers, water heaters, pool pumps, and general smart home appliances.

## Example Questions You Handle
- "When should I charge my electric car tomorrow to minimize cost and maximize solar power?"
- "What temperature should I set my thermostat on Wednesday afternoon if electricity prices spike?"
- "Suggest three ways I can reduce energy use based on my usage history."
- "How much can I save by running my dishwasher during off-peak hours?"
- "What's the best time to run my pool pump this week based on the weather forecast?"
"""

ecohome_agent = Agent(instructions=ECOHOME_SYSTEM_PROMPT)

# Quick smoke test
response = ecohome_agent.invoke(
    question="When should I charge my electric car tomorrow to minimize cost and maximize solar power?",
    context="Location: San Francisco, CA",
)
print(response["messages"][-1].content)

print("\nTOOLS:")
for msg in response["messages"]:
    obj = msg.model_dump()
    if obj.get("tool_call_id"):
        print("-", msg.name)

# =============================================================================
# 2. Define Test Cases (minimum 10)
# =============================================================================
# Covers: EV charging, thermostat/HVAC, appliance scheduling,
#         solar maximization, cost savings, multi-device planning,
#         and historical analysis.
test_cases = [
    # --- EV Charging ---
    {
        "id": "ev_charging_1",
        "question": "When should I charge my electric car tomorrow to minimize cost and maximize solar power?",
        "expected_tools": ["get_weather_forecast", "get_electricity_prices"],
        "expected_response": (
            "Specific hour window for charging, reference to solar generation forecast, "
            "cheapest price period identified, cost estimate for the session."
        ),
    },
    {
        "id": "ev_charging_2",
        "question": "How much money would I save if I charged my EV off-peak instead of during peak hours every day?",
        "expected_tools": ["get_electricity_prices", "calculate_energy_savings"],
        "expected_response": (
            "Daily and annual savings amount in GBP, comparison of peak vs off-peak rate, "
            "kWh per charge session used in the calculation."
        ),
    },

    # --- Thermostat / HVAC ---
    {
        "id": "thermostat_1",
        "question": "What temperature should I set my thermostat to on Wednesday afternoon when electricity prices spike?",
        "expected_tools": ["get_electricity_prices", "get_weather_forecast", "search_energy_tips"],
        "expected_response": (
            "Recommended thermostat setpoint during peak hours, outdoor temperature context, "
            "tip to pre-cool/pre-heat before the price spike."
        ),
    },
    {
        "id": "thermostat_2",
        "question": "How much does my HVAC system cost me per day and what can I do to reduce it?",
        "expected_tools": ["query_energy_usage", "get_electricity_prices", "calculate_energy_savings", "search_energy_tips"],
        "expected_response": (
            "Daily HVAC cost in USD/GBP from usage history, at least two actionable reduction strategies, "
            "savings estimate for each strategy."
        ),
    },

    # --- Appliance Scheduling ---
    {
        "id": "appliance_dishwasher",
        "question": "What is the cheapest time to run my dishwasher today?",
        "expected_tools": ["get_electricity_prices"],
        "expected_response": (
            "Specific hour(s) with lowest electricity rate, cost estimate for running the dishwasher "
            "at that time vs peak time."
        ),
    },
    {
        "id": "appliance_washing_machine",
        "question": "How much can I save by switching my washing machine to cold water and running it during off-peak hours?",
        "expected_tools": ["get_electricity_prices", "calculate_energy_savings", "search_energy_tips"],
        "expected_response": (
            "Savings from cold water wash (kWh and cost), additional savings from off-peak scheduling, "
            "combined annual saving projection."
        ),
    },

    # --- Solar Power Maximization ---
    {
        "id": "solar_pool_pump",
        "question": "What is the best time to run my pool pump this week based on the weather forecast?",
        "expected_tools": ["get_weather_forecast", "get_electricity_prices"],
        "expected_response": (
            "Best day and hour window for pool pump operation, reference to solar irradiance forecast, "
            "electricity cost saving vs running at other times."
        ),
    },
    {
        "id": "solar_generation_review",
        "question": "How much solar energy did I generate last week and how does weather affect my production?",
        "expected_tools": ["query_solar_generation", "get_weather_forecast"],
        "expected_response": (
            "Total kWh generated last week, daily breakdown or average, correlation between weather "
            "conditions and generation levels, forecast for the coming days."
        ),
    },

    # --- Cost Savings Calculations ---
    {
        "id": "cost_savings_top3",
        "question": "Suggest three ways I can reduce my energy use based on my usage history.",
        "expected_tools": ["get_recent_energy_summary", "query_energy_usage", "search_energy_tips", "calculate_energy_savings"],
        "expected_response": (
            "Three specific recommendations grounded in actual usage data, each with an estimated "
            "kWh saving and cost saving per month."
        ),
    },
    {
        "id": "cost_savings_shift_loads",
        "question": "How much could I save monthly if I shifted all my high-consumption appliances to off-peak hours?",
        "expected_tools": ["query_energy_usage", "get_electricity_prices", "calculate_energy_savings"],
        "expected_response": (
            "Current peak-hour consumption in kWh, off-peak rate vs peak rate, monthly and annual "
            "savings figure, list of which appliances to shift."
        ),
    },

    # --- Multi-Device Optimization ---
    {
        "id": "multi_device_day_plan",
        "question": "Give me a full day schedule for tomorrow that optimizes my EV charging, dishwasher, and HVAC to minimize electricity cost.",
        "expected_tools": ["get_weather_forecast", "get_electricity_prices", "search_energy_tips"],
        "expected_response": (
            "Hour-by-hour or block schedule for tomorrow covering EV, HVAC, and dishwasher, "
            "with the cheapest/solar-optimal windows highlighted and estimated total daily cost."
        ),
    },

    # --- Historical Analysis ---
    {
        "id": "historical_highest_consumer",
        "question": "Which of my devices consumes the most energy and what would be the impact of optimizing it?",
        "expected_tools": ["get_recent_energy_summary", "query_energy_usage", "calculate_energy_savings"],
        "expected_response": (
            "Top energy-consuming device identified from usage data, current daily/weekly kWh, "
            "concrete optimisation suggestion with savings estimate."
        ),
    },
]

if len(test_cases) < 10:
    raise ValueError("You MUST have at least 10 test cases")

print(f"\nDefined {len(test_cases)} test cases:")
for tc in test_cases:
    print(f"  [{tc['id']}]  tools: {tc['expected_tools']}")

# =============================================================================
# 3. Run Agent Tests
# =============================================================================
CONTEXT = "Location: San Francisco, CA"

print("\n=== Running Agent Tests ===")
test_results = []

for i, test_case in enumerate(test_cases):
    print(f"\nTest {i+1}: {test_case['id']}")
    print(f"Question: {test_case['question']}")
    print("-" * 50)

    try:
        response = ecohome_agent.invoke(question=test_case["question"], context=CONTEXT)
        result = {
            "test_id":           test_case["id"],
            "question":          test_case["question"],
            "response":          response,
            "expected_tools":    test_case["expected_tools"],
            "expected_response": test_case["expected_response"],
            "timestamp":         datetime.now().isoformat(),
        }
    except Exception as e:
        print(f"Error: {e}")
        result = {
            "test_id":           test_case["id"],
            "question":          test_case["question"],
            "response":          f"Error: {str(e)}",
            "expected_tools":    test_case["expected_tools"],
            "expected_response": test_case["expected_response"],
            "timestamp":         datetime.now().isoformat(),
            "error":             str(e),
        }
    test_results.append(result)

print(f"\nCompleted {len(test_results)} tests")

# =============================================================================
# 4. Evaluate Responses
# =============================================================================

# Shared LLM used by all evaluation functions (temp=0 for deterministic scoring).
_eval_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("OPENAI_API_KEY"),  # type: ignore
)


def evaluate_response(question: str, final_response: str, expected_response: str) -> dict:
    """
    LLM-as-judge evaluation of a single agent response.

    Scores four metrics on a 0-10 scale:
    - ACCURACY     : Facts, data references, and calculations correct?
    - RELEVANCE    : Does the response directly address the question?
    - COMPLETENESS : Covers all aspects in expected_response?
    - USEFULNESS   : Specific and actionable enough to act on?

    Returns per-metric scores + feedback and an overall average score.
    """
    prompt = f"""You are an expert evaluator assessing an AI energy advisor's response.

QUESTION: {question}

EXPECTED RESPONSE CRITERIA: {expected_response}

ACTUAL RESPONSE:
{final_response}

Score each metric from 0 to 10 and provide a one-sentence feedback for each.
Return ONLY a valid JSON object with exactly this structure — no markdown, no extra text:
{{
  "accuracy":     {{"score": <int 0-10>, "feedback": "<one sentence>"}},
  "relevance":    {{"score": <int 0-10>, "feedback": "<one sentence>"}},
  "completeness": {{"score": <int 0-10>, "feedback": "<one sentence>"}},
  "usefulness":   {{"score": <int 0-10>, "feedback": "<one sentence>"}}
}}

Scoring guide:
  ACCURACY     — penalise invented numbers, incorrect rates, or fabricated tool outputs.
  RELEVANCE    — penalise off-topic detours or failure to answer the actual question.
  COMPLETENESS — penalise missing elements listed in the expected criteria.
  USEFULNESS   — penalise vague advice; reward specific times, costs, and kWh figures."""

    try:
        result = _eval_llm.invoke(prompt)
        metrics = json.loads(result.content)
        overall = round(sum(v["score"] for v in metrics.values()) / len(metrics), 1)
        return {"metrics": metrics, "overall_score": overall}
    except Exception as e:
        empty = {"score": 0, "feedback": f"Evaluation error: {e}"}
        return {
            "metrics": {
                "accuracy":     empty,
                "relevance":    empty,
                "completeness": empty,
                "usefulness":   empty,
            },
            "overall_score": 0,
            "error": str(e),
        }


def evaluate_tool_usage(messages_list: list, expected_tools: list) -> dict:
    """
    Evaluate tool usage from an agent's message list.

    Metrics (each 0-10):
    - TOOL APPROPRIATENESS : Only sensible tools called? (−2 per unexpected tool)
    - TOOL COMPLETENESS    : All expected tools called? (matched/expected × 10)

    Returns tools_used, matched_tools, per-metric scores + feedback, overall average.
    """
    tools_used = [
        msg.model_dump()["name"]
        for msg in messages_list
        if msg.model_dump().get("tool_call_id") and msg.model_dump().get("name")
    ]

    used_set     = set(tools_used)
    expected_set = set(expected_tools)
    matched      = used_set & expected_set
    unexpected   = used_set - expected_set

    # Tool Completeness
    completeness_score = round((len(matched) / len(expected_set)) * 10, 1) if expected_set else 10.0
    missing = expected_set - used_set
    completeness_feedback = (
        f"Called {len(matched)}/{len(expected_set)} expected tool(s)."
        + (f" Missing: {sorted(missing)}." if missing else " All expected tools were used.")
    )

    # Tool Appropriateness — soft penalty of −2 per unexpected tool, floor at 0.
    # Extra tools are sometimes fine (agent may fetch proactive context).
    appropriateness_score = round(max(0.0, 10.0 - len(unexpected) * 2), 1)
    appropriateness_feedback = (
        "All tools called were appropriate for the question."
        if not unexpected
        else f"Potentially unnecessary tool(s) called: {sorted(unexpected)}."
    )

    return {
        "tools_used":    sorted(tools_used),
        "expected_tools": sorted(expected_tools),
        "matched_tools": sorted(matched),
        "tool_appropriateness": {"score": appropriateness_score, "feedback": appropriateness_feedback},
        "tool_completeness":    {"score": completeness_score,    "feedback": completeness_feedback},
        "overall_score": round((completeness_score + appropriateness_score) / 2, 1),
    }


def generate_evaluation_report() -> dict:
    """
    Run evaluate_response + evaluate_tool_usage for every entry in test_results,
    aggregate scores, identify strengths/weaknesses, and return a structured report.
    """
    if not test_results:
        raise ValueError("test_results is empty — run the agent tests first.")

    print(f"Generating evaluation report for {len(test_results)} test(s)…")
    evaluations = []

    for i, result in enumerate(test_results):
        print(f"  [{i+1}/{len(test_results)}] {result['test_id']}")

        if isinstance(result["response"], dict):
            messages  = result["response"].get("messages", [])
            final_txt = messages[-1].content if messages else ""
        else:
            messages  = []
            final_txt = str(result["response"])

        resp_eval = evaluate_response(
            question=result["question"],
            final_response=final_txt,
            expected_response=result["expected_response"],
        )
        tool_eval = evaluate_tool_usage(
            messages_list=messages,
            expected_tools=result["expected_tools"],
        )
        combined = round((resp_eval["overall_score"] + tool_eval["overall_score"]) / 2, 1)

        evaluations.append({
            "test_id":             result["test_id"],
            "question":            result["question"],
            "final_response":      final_txt,
            "response_evaluation": resp_eval,
            "tool_evaluation":     tool_eval,
            "combined_score":      combined,
        })

    # Aggregate response metrics
    resp_metric_names = ["accuracy", "relevance", "completeness", "usefulness"]
    resp_averages = {
        m: round(
            sum(e["response_evaluation"]["metrics"][m]["score"] for e in evaluations) / len(evaluations), 1
        )
        for m in resp_metric_names
    }

    # Aggregate tool metrics
    tool_averages = {
        "tool_appropriateness": round(
            sum(e["tool_evaluation"]["tool_appropriateness"]["score"] for e in evaluations) / len(evaluations), 1
        ),
        "tool_completeness": round(
            sum(e["tool_evaluation"]["tool_completeness"]["score"] for e in evaluations) / len(evaluations), 1
        ),
    }

    overall_score = round(sum(e["combined_score"] for e in evaluations) / len(evaluations), 1)
    all_scores = {**resp_averages, **tool_averages}

    strengths  = [m for m, s in all_scores.items() if s >= 7.0]
    weaknesses = [m for m, s in all_scores.items() if s <  6.0]

    # Build improvement recommendations based on low-scoring metrics
    recommendations = []
    if resp_averages["accuracy"] < 7:
        recommendations.append(
            "Ground responses in tool data: prompt the agent to cite specific "
            "numbers from tool outputs before drawing conclusions."
        )
    if resp_averages["completeness"] < 7:
        recommendations.append(
            "Enforce structured output: add explicit sections (Data Insights, "
            "Recommendation, Savings) to the system prompt."
        )
    if resp_averages["usefulness"] < 7:
        recommendations.append(
            "Add few-shot examples showing exact time windows and cost figures "
            "in the system prompt."
        )
    if tool_averages["tool_completeness"] < 7:
        recommendations.append(
            "Strengthen the system prompt to specify which tools are mandatory "
            "for each question category."
        )
    if tool_averages["tool_appropriateness"] < 7:
        recommendations.append(
            "Review system prompt to discourage redundant tool calls; "
            "consider adding a tool-selection reasoning step."
        )
    if not recommendations:
        recommendations.append(
            "Agent is performing well across all metrics. "
            "Consider expanding the test suite with adversarial or ambiguous questions."
        )

    return {
        "generated_at":             datetime.now().isoformat(),
        "total_tests":              len(evaluations),
        "overall_score":            overall_score,
        "response_metric_averages": resp_averages,
        "tool_metric_averages":     tool_averages,
        "strengths":                strengths,
        "weaknesses":               weaknesses,
        "recommendations":          recommendations,
        "test_evaluations":         evaluations,
    }


def display_evaluation_report(report: dict) -> None:
    """Pretty-print a report produced by generate_evaluation_report()."""
    LINE    = "=" * 62
    SUBLINE = "-" * 62

    print(LINE)
    print("   ECOHOME ENERGY ADVISOR — EVALUATION REPORT")
    print(f"   Generated : {report['generated_at']}")
    print(LINE)
    print(f"\n  {'OVERALL SCORE':<38} {report['overall_score']:>5.1f} / 10")
    print(f"  {'Tests run':<38} {report['total_tests']:>5}")

    print(f"\n{SUBLINE}")
    print("  RESPONSE QUALITY METRICS")
    print(SUBLINE)
    for metric, score in report["response_metric_averages"].items():
        bar   = "█" * int(score) + "░" * (10 - int(score))
        label = metric.upper()
        print(f"  {label:<16} {bar}  {score:>4.1f} / 10")

    print(f"\n{SUBLINE}")
    print("  TOOL USAGE METRICS")
    print(SUBLINE)
    for metric, score in report["tool_metric_averages"].items():
        bar   = "█" * int(score) + "░" * (10 - int(score))
        label = metric.replace("_", " ").upper()
        print(f"  {label:<22} {bar}  {score:>4.1f} / 10")

    print(f"\n{SUBLINE}")
    print("  STRENGTHS")
    print(SUBLINE)
    for s in report["strengths"] or ["None identified"]:
        print(f"  + {s.replace('_', ' ').title()}")

    print(f"\n{SUBLINE}")
    print("  WEAKNESSES")
    print(SUBLINE)
    for w in report["weaknesses"] or ["None identified"]:
        print(f"  - {w.replace('_', ' ').title()}")

    print(f"\n{SUBLINE}")
    print("  RECOMMENDATIONS")
    print(SUBLINE)
    for i, rec in enumerate(report["recommendations"], 1):
        # Word-wrap at ~56 chars
        words, line, lines = rec.split(), "", []
        for word in words:
            if len(line) + len(word) + 1 > 56:
                lines.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        if line:
            lines.append(line)
        print(f"  {i}. {lines[0]}")
        for extra in lines[1:]:
            print(f"     {extra}")

    print(f"\n{SUBLINE}")
    print("  PER-TEST RESULTS")
    print(SUBLINE)
    print(f"  {'Test ID':<28} {'Resp':>5}  {'Tools':>5}  {'Score':>5}  {'Pass'}")
    print(f"  {'-'*28} {'-'*5}  {'-'*5}  {'-'*5}  {'-'*4}")
    for ev in report["test_evaluations"]:
        r = ev["response_evaluation"]["overall_score"]
        t = ev["tool_evaluation"]["overall_score"]
        c = ev["combined_score"]
        status = "PASS" if c >= 6.0 else "FAIL"
        print(f"  {ev['test_id']:<28} {r:>5.1f}  {t:>5.1f}  {c:>5.1f}  {status}")

    print(f"\n{LINE}\n")


# =============================================================================
# Run evaluation
# =============================================================================
report = generate_evaluation_report()
display_evaluation_report(report)
