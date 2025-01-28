import google.generativeai as genai
import json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API with key from environment variable
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')

def generate_milestone_plan(user_query):
    # Determine timeframe from query
    timeframe = "weekly"  # default
    timeframe_map = {
        'week': 'week',
        'month': 'month',
        'year': 'year'
    }
    for key in timeframe_map:
        if key in user_query.lower():
            timeframe = timeframe_map[key]
            break

    # Generate appropriate date ranges
    base_date = datetime.now()
    entries_template = []
    period_info = {
        'week': {
            'period': 'day',
            'format': '%A',  # Monday, Tuesday, etc.
            'range': 7,
            'delta': timedelta(days=1)
        },
        'month': {
            'period': 'week',
            'format': 'Week %V',  # Week number
            'range': 4,
            'delta': timedelta(weeks=1)
        },
        'year': {
            'period': 'month',
            'format': '%B',  # January, February, etc.
            'range': 12,
            'delta': relativedelta(months=1)
        }
    }

    info = period_info[timeframe]
    for i in range(info['range']):
        entry_date = base_date + (info['delta'] * i)
        entries_template.append({
            "period": info['period'],
            "periodName": entry_date.strftime(info['format']) if info['period'] != 'week' 
                         else f"Week {i + 1}",
            "date": entry_date.strftime("%Y-%m-%d"),
            "title": "task title",
            "description": "detailed description"
        })

    # Generate prompt for Gemini
    prompt = f"""Generate a detailed plan in JSON format for: "{user_query}".
    Use this structure:
    {{
        "period": "{timeframe}",
        "title": "plan title",
        "entries": {json.dumps(entries_template, indent=2)}
    }}
    
    Current date: {base_date.strftime('%Y-%m-%d')}

    Follow these period formatting rules strictly:
    1. Use ONLY the period type specified in the template above
    2. Keep all periodName values exactly as provided in the template
    3. Maintain the exact dates provided in the template
    4. Include all activities for each period in the description field
    5. DO NOT create additional entries or change the period structure
    
    Based on the type of plan requested, include these specific details in the description field:
    - For fitness/workout plans: Include exercises, sets, reps, and rest periods
    - For diet plans: Include meals, portions, calories, and nutritional info
    - For study plans: Include topics, learning objectives, and resources
    - For financial plans: Include specific amounts, strategies, and goals
    - For project plans: Include tasks, deadlines, and deliverables
    - For habit-building plans: Include specific actions, triggers, and tracking methods
    
    Keep descriptions detailed but concise. Format numbers consistently."""
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Extract JSON content between ```json and ``` markers
        if '```json' in response_text:
            json_content = response_text.split('```json')[1].split('```')[0].strip()
        else:
            json_content = response_text
        
        # Parse and validate the JSON
        parsed_json = json.loads(json_content)
        
        # Validate required fields and structure
        required_fields = ['period', 'title', 'entries']
        if not all(field in parsed_json for field in required_fields):
            raise ValueError("Missing required fields in response")
        
        # Validate period consistency
        if parsed_json['period'] != timeframe:
            raise ValueError(f"Incorrect period type. Expected {timeframe}")
        
        # Validate entries structure matches template
        if len(parsed_json['entries']) != len(entries_template):
            raise ValueError("Incorrect number of entries")
            
        for i, entry in enumerate(parsed_json['entries']):
            template_entry = entries_template[i]
            if entry['period'] != template_entry['period']:
                raise ValueError(f"Incorrect period type in entry {i}")
            if entry['date'] != template_entry['date']:
                raise ValueError(f"Incorrect date in entry {i}")
            if entry['periodName'] != template_entry['periodName']:
                raise ValueError(f"Incorrect periodName in entry {i}")
            
        return parsed_json
    except json.JSONDecodeError as e:
        return {
            "error": f"Invalid JSON response: {str(e)}",
            "raw_response": response.text if 'response' in locals() else "No response received"
        }
    except ValueError as e:
        return {
            "error": f"Validation error: {str(e)}",
            "raw_response": response.text if 'response' in locals() else "No response received"
        }
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

# Example usage
if __name__ == "__main__":
    # Test different types of plans
    test_queries = [
        "create a month plan to loose belly fat ",
        "create a monthly financial savings plan with $1000 budget",
        "design a weekly workout routine for beginners",
        "make a weekly habit building plan for early morning routine",
        "plan a yearly career development roadmap for software engineer"
    ]
    
    for query in test_queries:
        print(f"\n=== Testing: {query} ===")
        plan = generate_milestone_plan(query)
        print(json.dumps(plan, indent=4, ensure_ascii=False))