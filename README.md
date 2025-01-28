# AI Planning Assistant

A Flask-based API that generates detailed plans using Google's Gemini 2.0 AI model. Supports various types of planning queries from diet plans to financial goals, automatically detecting and structuring responses based on time periods.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure API key:
- Add your Google API key to the `.env` file

## Usage

1. Start the server:
```bash
python milestone_generator.py
```

2. The API provides two endpoints:
- GET `/sample`: Returns example queries and input format
- POST `/generate`: Generates plans based on natural language queries

3. Example curl commands:
```bash
# Get sample queries
curl http://localhost:5000/sample

# Generate a plan
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "Build a week protein rich diet plan"}'
```

## Input Format

The API expects a simple JSON input with a query:
```json
{
  "query": "Your planning request here"
}
```

Example queries:
- "Build a week protein rich diet plan"
- "How can I achieve financial goals in a year?"
- "Create a month-long workout routine"
- "Plan my daily study schedule for IELTS"

## Time Period Detection

The system automatically detects the time period from your query:
- Year: Creates 12-month plans with weekly breakdowns
- Month: Creates 4-week plans with daily highlights
- Week: Creates 7-day plans with detailed activities
- Day: Creates daily schedules with morning/afternoon/evening sections

## Output Format

The API returns a JSON response containing:
- Structured timeline based on detected period
- Specific actionable items
- Measurable goals and metrics
- Required resources
- Tips and recommendations 