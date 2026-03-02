# Overview

An AI-powered energy optimization agent that helps customers reduce electricity costs and environmental impact. The Energy Advisor agent provides personalized recommendations about when to run solar panels, electric vehicles, and smart thermostats to minimize costs and carbon footprint.

## Key Features

- **Weather Integration**: Uses weather forecasts to predict solar generation
- **Dynamic Pricing**: Considers time-of-day electricity prices for cost optimization
- **Historical Analysis**: Queries past energy usage patterns for personalized advice
- **RAG Pipeline**: Retrieves relevant energy-saving tips and best practices
- **Multi-device Optimization**: Handles EVs, HVAC, appliances, and solar systems
- **Cost Calculations**: Provides specific savings estimates and ROI analysis

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -e .
```

### 2. Set Environment Variables

Create a `.env` file with your API keys:

```bash
OPENAI_API_KEY=<your-key>
OPENAI_BASE_URL=<optional-url>
```

## Agent Capabilities

### Tools Available

- **Weather Forecast**: Get hourly weather predictions and solar irradiance
- **Electricity Pricing**: Access time-of-day pricing data
- **Energy Usage Query**: Retrieve historical consumption data
- **Solar Generation Query**: Get past solar production data
- **Energy Tips Search**: Find relevant energy-saving recommendations
- **Savings Calculator**: Compute potential cost savings

### Example Questions

The Energy Advisor can answer questions like:

- "When should I charge my electric car tomorrow to minimize cost and maximize solar power?"
- "What temperature should I set my thermostat on Wednesday afternoon if electricity prices spike?"
- "Suggest three ways I can reduce energy use based on my usage history."
- "How much can I save by running my dishwasher during off-peak hours?"

## Database Schema

### Energy Usage Table
- `timestamp`: When the energy was consumed
- `consumption_kwh`: Amount of energy used
- `device_type`: Type of device (EV, HVAC, appliance)
- `device_name`: Specific device name
- `cost_usd`: Cost at time of usage

### Solar Generation Table
- `timestamp`: When the energy was generated
- `generation_kwh`: Amount of solar energy produced
- `weather_condition`: Weather during generation
- `temperature_c`: Temperature at time of generation
- `solar_irradiance`: Solar irradiance level

## Key Technologies

- **LangChain**: Agent framework and tool integration
- **LangGraph**: Agent orchestration and workflow
- **ChromaDB**: Vector database for document retrieval
- **SQLAlchemy**: Database ORM and management
- **OpenAI**: LLM and embeddings
- **SQLite**: Local database storage

## Evaluation Criteria

The agent is evaluated on:

- **Accuracy**: Correct information and calculations
- **Relevance**: Responses address the user's question
- **Completeness**: Comprehensive answers with actionable advice
- **Tool Usage**: Appropriate use of available tools
- **Reasoning**: Clear explanation of recommendations

## Getting Started

1. Clone this repository
2. Install the required dependencies
3. Set up your environment variables
4. Run the notebooks in sequence
5. Test the agent with your own questions

## License

This project is part of the Udacity curriculum.
