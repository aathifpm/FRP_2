import os
import json
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.corpus import stopwords
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from datetime import datetime, timedelta

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')

app = Flask(__name__)

def analyze_query(query):
    """Analyze the query using NLTK to extract key information."""
    # Tokenize the query
    tokens = word_tokenize(query)
    # Get POS tags
    pos_tags = pos_tag(tokens)
    # Get named entities
    named_entities = ne_chunk(pos_tags)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w.lower() not in stop_words]
    
    # Extract key components
    time_words = []
    action_words = []
    subjects = []
    
    for token, pos in pos_tags:
        if pos.startswith('VB'):  # Verbs
            action_words.append(token)
        elif pos.startswith('NN'):  # Nouns
            subjects.append(token)
        elif token.lower() in ['day', 'week', 'month', 'year']:
            time_words.append(token)
    
    return {
        'tokens': tokens,
        'pos_tags': pos_tags,
        'named_entities': named_entities,
        'filtered_tokens': filtered_tokens,
        'time_words': time_words,
        'action_words': action_words,
        'subjects': subjects
    }

def clean_query(query):
    """Clean and normalize the input query using NLTK."""
    # Tokenize and normalize
    tokens = word_tokenize(query.lower())
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    # Join back into a string
    cleaned = ' '.join(tokens)
    # Remove special characters but keep basic punctuation
    cleaned = re.sub(r'[^\w\s.,!?-]', '', cleaned)
    return cleaned

def extract_duration(query):
    """Extract duration information from the query."""
    query = query.lower()
    
    # Common duration patterns
    duration_patterns = {
        'year': r'(\d+)[\s-]*(year|yr|annual|yearly)',
        'month': r'(\d+)[\s-]*(month|monthly)',
        'week': r'(\d+)[\s-]*(week|wk|weekly)',
        'day': r'(\d+)[\s-]*(day|daily)'
    }
    
    for period, pattern in duration_patterns.items():
        match = re.search(pattern, query)
        if match:
            number = int(match.group(1))
            return period, number
    
    return None, None

def extract_time_context(query, pos_tags):
    """Extract detailed time context from the query using NLTK analysis."""
    # Common time-related words and their mappings
    time_mappings = {
        'daily': 'day',
        'tomorrow': 'day',
        'today': 'day',
        'weekly': 'week',
        'monthly': 'month',
        'yearly': 'year',
        'annual': 'year'
    }
    
    # Time unit patterns
    time_units = {
        'day': ['day', 'days', 'daily', 'tomorrow', 'today'],
        'week': ['week', 'weeks', 'weekly', '7-day', 'seven-day'],
        'month': ['month', 'months', 'monthly', '30-day', 'thirty-day'],
        'year': ['year', 'years', 'yearly', 'annual', 'annually']
    }
    
    # Extract time-related phrases using POS tags
    time_phrases = []
    query_lower = query.lower()
    
    # Look for numeric time patterns first (e.g., "7 day", "30 day", etc.)
    numeric_patterns = [
        (r'(\d+)[\s-]*(day|week|month|year)s?', lambda m: (m.group(2), int(m.group(1)))),
        (r'(seven|thirty)[\s-]*(day|week|month)s?', lambda m: (m.group(2), 7 if m.group(1) == 'seven' else 30))
    ]
    
    for pattern, handler in numeric_patterns:
        match = re.search(pattern, query_lower)
        if match:
            unit, number = handler(match)
            if unit == 'day' and number == 7:
                return 'week'
            if unit == 'day' and number == 30:
                return 'month'
            return unit
    
    # Extract time-related words and their context
    for i, (word, tag) in enumerate(pos_tags):
        word_lower = word.lower()
        
        # Direct mapping check
        if word_lower in time_mappings:
            return time_mappings[word_lower]
        
        # Check for time units
        for unit, variants in time_units.items():
            if word_lower in variants:
                # Look for modifiers before the time word
                if i > 0:
                    prev_word = pos_tags[i-1][0].lower()
                    if prev_word.isdigit():
                        if int(prev_word) == 7 and unit == 'day':
                            return 'week'
                        if int(prev_word) == 30 and unit == 'day':
                            return 'month'
                return unit
    
    # Check for specific phrases
    phrases = [
        ('next week', 'week'),
        ('this month', 'month'),
        ('this year', 'year'),
        ('daily routine', 'day'),
        ('weekly plan', 'week'),
        ('monthly schedule', 'month'),
        ('yearly goals', 'year')
    ]
    
    for phrase, period in phrases:
        if phrase in query_lower:
            return period
    
    # Default based on query context
    if any(word in query_lower for word in ['routine', 'today', 'tomorrow', 'schedule']):
        return 'day'
    if any(word in query_lower for word in ['weekend', 'weekly']):
        return 'week'
    
    # Default to week as it's the most common planning period
    return 'week'

def detect_time_period(query):
    """Detect the time period from the query using improved NLTK analysis."""
    # Tokenize and get POS tags
    tokens = word_tokenize(query)
    pos_tags = pos_tag(tokens)
    
    # Extract time context
    time_period = extract_time_context(query, pos_tags)
    
    # Additional validation
    query_lower = query.lower()
    
    # Check for compound time words
    if any(pattern in query_lower for pattern in ['week long', 'week-long', 'weekly']):
        time_period = 'week'
    elif any(pattern in query_lower for pattern in ['month long', 'month-long', 'monthly']):
        time_period = 'month'
    elif any(pattern in query_lower for pattern in ['year long', 'year-long', 'yearly', 'annual']):
        time_period = 'year'
    elif any(pattern in query_lower for pattern in ['daily', 'day long', 'day-long', "tomorrow's", "today's"]):
        time_period = 'day'
    
    # Override for explicit patterns
    if '7 day' in query_lower or 'seven day' in query_lower or '7-day' in query_lower:
        time_period = 'week'
    elif '30 day' in query_lower or 'thirty day' in query_lower or '30-day' in query_lower:
        time_period = 'month'
    elif '365 day' in query_lower or 'year long' in query_lower:
        time_period = 'year'
    
    # Default to week for workout/diet/study plans if no specific time is mentioned
    if time_period == 'day':
        if any(word in query_lower for word in ['workout', 'diet', 'meal', 'study', 'learn']):
            time_period = 'week'
    
    return time_period

def structure_output(plan_data):
    """Structure the output using NLTK for better organization."""
    # Analyze descriptions and notes
    for key, value in plan_data.items():
        if isinstance(value, str) and key in ['description', 'notes']:
            # Split into sentences
            sentences = sent_tokenize(value)
            # Keep only the most informative sentences (first and last)
            if len(sentences) > 2:
                plan_data[key] = f"{sentences[0]} {sentences[-1]}"
        
        # Process nested structures
        elif isinstance(value, dict):
            plan_data[key] = structure_output(value)
        elif isinstance(value, list):
            # Process list items
            processed_items = []
            for item in value:
                if isinstance(item, (dict, list)):
                    processed_items.append(structure_output(item))
                elif isinstance(item, str):
                    # Remove redundant or less informative items
                    if len(processed_items) < 4:  # Keep max 4 items
                        processed_items.append(item)
            plan_data[key] = processed_items
    
    return plan_data

def format_output(plan, time_period):
    """Format the plan into the standardized output structure."""
    output = {
        "period": time_period,
        "title": plan.get('title', ''),
        "entries": []
    }
    
    if time_period == 'week':
        for day in plan.get('days', []):
            entry = {
                "period": "day",
                "periodName": day.get('day', ''),
                "date": "",  # Can be filled if needed
                "title": day.get('focus_area', ''),
                "description": {
                    "schedule": day.get('schedule', {}),
                    "resources": day.get('resources_needed', []),
                    "metrics": day.get('metrics_to_track', [])
                }
            }
            output['entries'].append(entry)
    
    elif time_period == 'month':
        for week in plan.get('weeks', []):
            entry = {
                "period": "week",
                "periodName": week.get('week', ''),
                "date": "",  # Can be filled if needed
                "title": week.get('focus_area', ''),
                "description": {
                    "goals": week.get('goals', []),
                    "activities": week.get('key_activities', []),
                    "resources": week.get('resources_needed', [])
                }
            }
            output['entries'].append(entry)
    
    elif time_period == 'year':
        for month in plan.get('months', []):
            entry = {
                "period": "month",
                "periodName": month.get('month', ''),
                "date": "",  # Can be filled if needed
                "title": month.get('focus_area', ''),
                "description": {
                    "goals": month.get('goals', []),
                    "milestones": month.get('milestones', []),
                    "resources": month.get('resources_needed', [])
                }
            }
            output['entries'].append(entry)
    
    return output

def get_date_for_entry(time_period, period_name):
    """Generate appropriate date for an entry based on time period."""
    today = datetime.now()
    
    if time_period == 'week':
        # Map day names to weekday numbers (0 = Monday)
        day_map = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        target_day = day_map.get(period_name, 0)
        current_day = today.weekday()
        days_ahead = target_day - current_day
        
        if days_ahead <= 0:  # Target day has passed this week
            days_ahead += 7  # Move to next week
            
        target_date = today + timedelta(days=days_ahead)
        return target_date.strftime("%Y-%m-%d")
        
    elif time_period == 'month':
        # For monthly plans, set dates to start of each week
        week_num = int(period_name.split()[1])  # Extract week number
        days_ahead = (week_num - 1) * 7  # Each week starts 7 days after the previous
        target_date = today + timedelta(days=days_ahead)
        return f"{target_date.strftime('%Y-%m-%d')} to {(target_date + timedelta(days=6)).strftime('%Y-%m-%d')}"
        
    elif time_period == 'year':
        # For yearly plans, set dates to start of each month
        month_names = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        target_month = month_names.get(period_name, 1)
        target_year = today.year if target_month >= today.month else today.year + 1
        start_date = datetime(target_year, target_month, 1)
        
        # Get the last day of the month
        if target_month == 12:
            end_date = datetime(target_year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(target_year, target_month + 1, 1) - timedelta(days=1)
            
        return f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    
    # For daily plans or fallback
    return today.strftime("%Y-%m-%d")

def generate_time_content(query, time_period, analysis):
    """Generate time-specific content using NLTK analysis."""
    # Create base structure for the plan
    base_structure = {
        "period": time_period,
        "title": query.strip().title(),
        "entries": []
    }

    # Extract duration from query
    query_lower = query.lower()
    duration = 0
    for num in ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
        if num in query_lower:
            duration = int(num) if num.isdigit() else {
                'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
            }[num]
            break

    # Determine content type and initial values
    content_type = "general"
    initial_values = {}
    if any(word in query_lower for word in ["workout", "fitness", "exercise", "training", "gym"]):
        content_type = "workout"
        initial_values = {
            "calories_target": "2500",
            "workout_duration": "45-60",
            "intensity_level": "moderate"
        }
    elif any(word in query_lower for word in ["study", "learn", "practice", "education", "course", "ielts", "coding"]):
        content_type = "study"
        initial_values = {
            "daily_hours": "4",
            "practice_sessions": "3",
            "review_frequency": "weekly"
        }
    elif any(word in query_lower for word in ["diet", "meal", "food", "nutrition", "protein", "vegetarian"]):
        content_type = "meal"
        initial_values = {
            "daily_calories": "2000",
            "protein_target": "150",
            "carbs_target": "250",
            "fats_target": "70"
        }
    elif any(word in query_lower for word in ["budget", "savings", "financial", "money", "finance"]):
        content_type = "finance"
        initial_values = {
            "monthly_savings": "20%",
            "emergency_fund": "6 months",
            "investment_ratio": "30%"
        }

    # Generate entries based on time period with progressive changes
    if time_period == 'week':
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for i, day in enumerate(days):
            entry = {
                "period": "day",
                "periodName": day,
                "date": get_date_for_entry(time_period, day),
                "title": f"{day}'s Focus",
                "description": create_progressive_content(content_type, i + 1, len(days), initial_values)
            }
            base_structure["entries"].append(entry)
            
    elif time_period == 'month':
        num_weeks = duration if duration > 0 else 4
        for week_num in range(1, num_weeks + 1):
            week_name = f"Week {week_num}"
            entry = {
                "period": "week",
                "periodName": week_name,
                "date": get_date_for_entry(time_period, week_name),
                "title": f"{week_name} Focus",
                "description": create_progressive_content(content_type, week_num, num_weeks, initial_values)
            }
            base_structure["entries"].append(entry)
            
    elif time_period == 'year':
        months = ['January', 'February', 'March', 'April', 'May', 'June', 
                 'July', 'August', 'September', 'October', 'November', 'December']
        num_months = duration if duration > 0 else 12
        months = months[:num_months]  # Limit months if duration specified
        
        for i, month in enumerate(months):
            entry = {
                "period": "month",
                "periodName": month,
                "date": get_date_for_entry(time_period, month),
                "title": f"{month} Focus",
                "description": create_progressive_content(content_type, i + 1, len(months), initial_values, month)
            }
            base_structure["entries"].append(entry)
            
    else:  # day
        entry = {
            "period": "day",
            "periodName": "Today",
            "date": get_date_for_entry(time_period, "Today"),
            "title": "Today's Focus",
            "description": create_progressive_content(content_type, 1, 1, initial_values)
        }
        base_structure["entries"].append(entry)

    return base_structure

def is_default_content(description, content_type):
    """Check if content contains only default values."""
    if content_type == "meal":
        macros = description.get('macros', {})
        return all(v == "0g" or v == "0" for v in macros.values())
    elif content_type == "finance":
        goals = description.get('goals', {})
        return all(v == "0" for v in goals.values())
    return False

def create_progressive_content(content_type, current_period, total_periods, initial_values, period_name=''):
    """Create content that progresses over time."""
    base = create_content_template(content_type)
    
    # Calculate progression factor (0.0 to 1.0)
    progress = current_period / total_periods
    
    if content_type == "study":
        # Define IELTS study components
        study_components = {
            "reading": [
                "Skimming and scanning practice",
                "Reading for detail",
                "Time management strategies",
                "Multiple choice questions",
                "True/False/Not Given",
                "Matching headings"
            ],
            "writing": [
                "Task 1 - Data interpretation",
                "Task 1 - Process description",
                "Task 2 - Essay structure",
                "Task 2 - Argument development",
                "Grammar and vocabulary",
                "Coherence and cohesion"
            ],
            "listening": [
                "Note completion",
                "Multiple choice",
                "Map/Plan completion",
                "Form filling",
                "Sentence completion",
                "Summary completion"
            ],
            "speaking": [
                "Part 1 - Personal questions",
                "Part 2 - Long turn",
                "Part 3 - Discussion",
                "Pronunciation practice",
                "Fluency development",
                "Vocabulary building"
            ]
        }

        # Set daily focus areas based on the day of the week
        if period_name in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            day_focus = {
                'Monday': ('reading', 'writing'),
                'Tuesday': ('listening', 'speaking'),
                'Wednesday': ('writing', 'reading'),
                'Thursday': ('speaking', 'listening'),
                'Friday': ('reading', 'listening'),
                'Saturday': ('writing', 'speaking'),
                'Sunday': ('review', 'practice_test')
            }
            focus_areas = day_focus[period_name]
        else:
            focus_areas = ('all_skills', 'review')

        # Create schedule based on focus areas
        morning_schedule = []
        afternoon_schedule = []
        evening_schedule = []
        
        if focus_areas[0] == 'review':
            # Sunday review schedule
            morning_schedule = [
                "Review previous week's materials - 1 hour",
                "Practice test (Reading) - 1 hour",
                "Practice test (Writing Task 1) - 1 hour"
            ]
            afternoon_schedule = [
                "Practice test (Listening) - 1 hour",
                "Practice test (Writing Task 2) - 1 hour",
                "Review and mark practice tests - 1 hour"
            ]
            evening_schedule = [
                "Speaking practice with study partner - 30 mins",
                "Plan next week's study goals - 30 mins",
                "Review weak areas identified in practice tests - 1 hour"
            ]
        else:
            # Regular day schedule
            if focus_areas[0] in study_components:
                morning_schedule = [
                    f"{focus_areas[0].title()} skill: {component} - 45 mins"
                    for component in study_components[focus_areas[0]][:2]
                ]
                morning_schedule.append("Vocabulary building - 30 mins")
            
            if focus_areas[1] in study_components:
                afternoon_schedule = [
                    f"{focus_areas[1].title()} skill: {component} - 45 mins"
                    for component in study_components[focus_areas[1]][:2]
                ]
                afternoon_schedule.append("Grammar practice - 30 mins")
            
            evening_schedule = [
                "Review day's learning - 30 mins",
                "Practice exercises - 45 mins",
                "Prepare for tomorrow's topics - 15 mins"
            ]

        # Update base content
        base.update({
            "schedule": {
                "morning": morning_schedule,
                "afternoon": afternoon_schedule,
                "evening": evening_schedule
            },
            "topics": [
                f"Focus 1: {focus_areas[0].replace('_', ' ').title()}",
                f"Focus 2: {focus_areas[1].replace('_', ' ').title()}",
                "Vocabulary and Grammar Development"
            ],
            "resources": [
                "Cambridge IELTS Practice Tests",
                "IELTS Official Guide",
                "Online practice platform",
                "Study timer/stopwatch",
                "Note-taking materials"
            ],
            "goals": [
                f"Complete {focus_areas[0].title()} exercises with {int(70 + (progress * 20))}% accuracy",
                f"Practice {focus_areas[1].title()} for at least 2 hours",
                "Learn and use 10 new vocabulary words",
                "Complete all planned practice sessions"
            ]
        })
        
    elif content_type == "workout":
        # Define exercise categories and progression
        base_exercises = {
            "warmup": [
                "Dynamic stretching - 5 mins",
                "Light jogging in place - 5 mins",
                "Arm circles and leg swings - 2 mins"
            ],
            "cardio": [
                "Jumping jacks - 1 min",
                "High knees - 30 secs",
                "Mountain climbers - 30 secs",
                "Burpees - 30 secs"
            ],
            "strength": [
                "Push-ups - 10 reps",
                "Bodyweight squats - 15 reps",
                "Lunges - 10 each leg",
                "Plank hold - 30 secs"
            ],
            "cooldown": [
                "Static stretching - 5 mins",
                "Deep breathing - 2 mins",
                "Light walking - 3 mins"
            ]
        }

        # Adjust intensity based on progression
        intensity_levels = ["Light", "Light-Moderate", "Moderate", "Moderate-High", "High"]
        current_intensity = intensity_levels[min(int(progress * 5), 4)]
        
        # Adjust exercise difficulty based on progress
        if progress > 0.3:
            base_exercises["strength"] = [
                "Push-ups - 15 reps",
                "Jump squats - 20 reps",
                "Walking lunges - 15 each leg",
                "Plank hold - 45 secs"
            ]
        if progress > 0.6:
            base_exercises["strength"] = [
                "Diamond push-ups - 12 reps",
                "Pistol squats - 8 each leg",
                "Jump lunges - 12 each leg",
                "Plank with shoulder taps - 45 secs"
            ]

        # Set default focus
        focus = "General Fitness"

        # Create weekly/monthly focus areas
        if period_name:
            if period_name.startswith('Week'):
                week_num = int(period_name.split()[1])
                focus_areas = {
                    1: "Form and Technique",
                    2: "Building Endurance",
                    3: "Strength Development",
                    4: "High-Intensity Training"
                }
                focus = focus_areas.get(week_num, "Maintenance")
            else:  # Monthly
                month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                           'July', 'August', 'September', 'October', 'November', 'December']
                try:
                    month_num = month_names.index(period_name)
                    if month_num < 3:
                        focus = "Building Foundation"
                    elif month_num < 6:
                        focus = "Strength and Endurance"
                    elif month_num < 9:
                        focus = "Power and Performance"
                    else:
                        focus = "Peak Performance"
                except ValueError:
                    focus = "General Fitness"

        # Combine exercises based on focus
        exercises = []
        exercises.extend(base_exercises["warmup"])
        exercises.extend(base_exercises["cardio"])
        exercises.extend(base_exercises["strength"])
        exercises.extend(base_exercises["cooldown"])

        # Update base content
        base.update({
            "exercises": exercises,
            "duration": f"{45 + int(progress * 15)} minutes",
            "intensity": current_intensity,
            "equipment": [
                "Exercise mat",
                "Water bottle",
                "Timer/stopwatch",
                "Resistance bands" if progress > 0.3 else None,
                "Dumbbells (optional)" if progress > 0.6 else None
            ],
            "notes": [
                f"Focus area: {focus}",
                f"Target heart rate: {110 + int(progress * 40)}-{130 + int(progress * 40)} BPM",
                "Rest 30-60 seconds between exercises",
                "Maintain proper form throughout",
                f"Progress to next level when current exercises feel comfortable"
            ]
        })

        # Clean up None values from equipment list
        base["equipment"] = [item for item in base["equipment"] if item is not None]
        
    elif content_type == "meal":
        # Adjust macros based on progression
        protein = int(initial_values['protein_target']) + int(10 * progress)
        carbs = int(initial_values['carbs_target']) - int(20 * progress)
        fats = int(initial_values['fats_target'])
        calories = int(initial_values['daily_calories']) + int(100 * progress)
        
        base['macros'].update({
            "protein_target": f"{protein}g",
            "carbs_target": f"{carbs}g",
            "fats_target": f"{fats}g",
            "total_calories": str(calories)
        })
        
        # Set tracking targets
        base['tracking'].update({
            "weight": f"Track weekly: Target {progress:.1f}% progress",
            "protein_intake": f"{protein}g daily",
            "water_intake": "2-3 liters daily"
        })
        
        # Add nutrient guidelines
        base['nutrients'].update({
            "protein": [
                f"Target {protein}g protein daily",
                "Space protein intake throughout the day",
                f"Minimum {int(protein/4)}g protein per main meal"
            ],
            "carbs": [
                f"Target {carbs}g complex carbs daily",
                "Focus on whole grains and vegetables",
                "Limit refined sugars and processed carbs"
            ],
            "fats": [
                f"Target {fats}g healthy fats daily",
                "Include sources of omega-3 fatty acids",
                "Limit saturated fats to 20g daily"
            ]
        })
        
        # Add monthly tasks
        base['tasks'] = [
            f"Week 1: Meal prep and grocery planning",
            f"Week 2: Review and adjust portions based on progress",
            f"Week 3: Try new recipes with seasonal ingredients",
            f"Week 4: Monthly progress assessment and adjustments"
        ]
        
        # Add contextual tips
        if period_name:
            seasonal_tips = {
                'winter': ["Boost vitamin D intake", "Include warming foods", "Focus on immune support"],
                'spring': ["Incorporate fresh greens", "Lighter cooking methods", "Seasonal produce focus"],
                'summer': ["Stay hydrated", "Light, cooling meals", "Grill and fresh prep"],
                'fall': ["Boost fiber intake", "Hearty, warming dishes", "Immune system support"]
            }
            season = get_season(period_name)
            base['tips'] = seasonal_tips.get(season, ["Stay consistent with portions", "Track macros daily", "Prep meals in advance"])
        
        # Adjust meals based on season if period_name is a month
        if period_name:
            seasonal_foods = get_seasonal_foods(period_name)
            base['meals'] = create_varied_meal_plan(seasonal_foods, protein, period_name)
            
    elif content_type == "finance":
        # Progressive financial targets
        monthly_savings = float(initial_values['monthly_savings'].rstrip('%'))
        savings_target = monthly_savings + (5 * progress)
        investment_ratio = float(initial_values['investment_ratio'].rstrip('%'))
        investment_target = investment_ratio + (10 * progress)
        
        base['goals'].update({
            "savings_target": f"{savings_target:.1f}%",
            "investment_allocation": f"{investment_target:.1f}%",
            "expense_reduction": f"{5 + (5 * progress):.1f}%"
        })
        
        # Update tracking metrics
        base['tracking'].update({
            "current_balance": "Update weekly",
            "savings_progress": f"Target: {savings_target:.1f}% of income",
        })
        
        # Customize tasks based on period
        base['tasks'] = create_financial_tasks(current_period, period_name)
        
    return base

def get_season(month):
    """Determine season based on month."""
    seasons = {
        'winter': ['December', 'January', 'February'],
        'spring': ['March', 'April', 'May'],
        'summer': ['June', 'July', 'August'],
        'fall': ['September', 'October', 'November']
    }
    for season, months in seasons.items():
        if month in months:
            return season
    return 'summer'  # default

def get_seasonal_foods(month):
    """Get seasonal foods for each month."""
    seasonal_map = {
        'January': ['root vegetables', 'citrus fruits', 'winter greens'],
        'February': ['winter squash', 'potatoes', 'citrus fruits'],
        'March': ['spring greens', 'asparagus', 'early berries'],
        'April': ['peas', 'asparagus', 'spring onions'],
        'May': ['strawberries', 'new potatoes', 'spring vegetables'],
        'June': ['summer berries', 'leafy greens', 'early tomatoes'],
        'July': ['tomatoes', 'summer squash', 'stone fruits'],
        'August': ['corn', 'tomatoes', 'melons'],
        'September': ['apples', 'pears', 'fall squash'],
        'October': ['pumpkins', 'apples', 'root vegetables'],
        'November': ['winter squash', 'root vegetables', 'cranberries'],
        'December': ['winter citrus', 'root vegetables', 'winter greens']
    }
    return seasonal_map.get(month, ['vegetables', 'fruits', 'proteins'])

def create_varied_meal_plan(seasonal_foods, protein_target, month):
    """Create a varied meal plan incorporating seasonal foods."""
    breakfast_options = [
        f"Oatmeal with {seasonal_foods[0]} and almonds ({int(protein_target/6)}g protein)",
        f"Greek yogurt parfait with {seasonal_foods[1]} ({int(protein_target/6)}g protein)",
        f"Protein smoothie bowl with {seasonal_foods[1]} ({int(protein_target/6)}g protein)",
        f"Quinoa breakfast bowl with {seasonal_foods[0]} ({int(protein_target/6)}g protein)"
    ]
    
    lunch_options = [
        f"Quinoa bowl with {seasonal_foods[1]} and tofu ({int(protein_target/4)}g protein)",
        f"Lean protein salad with {seasonal_foods[0]} ({int(protein_target/4)}g protein)",
        f"Whole grain wrap with {seasonal_foods[2]} ({int(protein_target/4)}g protein)",
        f"Buddha bowl with {seasonal_foods[1]} ({int(protein_target/4)}g protein)"
    ]
    
    dinner_options = [
        f"Grilled fish with {seasonal_foods[2]} ({int(protein_target/3)}g protein)",
        f"Lean protein stir-fry with {seasonal_foods[0]} ({int(protein_target/3)}g protein)",
        f"Baked protein with {seasonal_foods[1]} ({int(protein_target/3)}g protein)",
        f"One-pan roast with {seasonal_foods[2]} ({int(protein_target/3)}g protein)"
    ]
    
    # Adjust cooking methods by season
    season = get_season(month)
    if season == 'summer':
        dinner_options = [option.replace('Baked', 'Grilled').replace('roast', 'grilled') for option in dinner_options]
    elif season == 'winter':
        dinner_options = [option.replace('Grilled', 'Roasted').replace('stir-fry', 'baked') for option in dinner_options]
    
    snack_options = [
        f"Fresh {seasonal_foods[1]} with protein yogurt dip",
        f"Trail mix with nuts and dried {seasonal_foods[1]}",
        f"Protein smoothie with {seasonal_foods[0]}",
        "Hummus with vegetable crudités"
    ]
    
    return {
        "breakfast": breakfast_options[:3],  # Limit to 3 options
        "lunch": lunch_options[:3],
        "dinner": dinner_options[:3],
        "snacks": snack_options[:3]
    }

def create_financial_tasks(period_num, period_name=''):
    """Create specific financial tasks based on the period."""
    common_tasks = [
        "Review and categorize expenses",
        "Update budget tracking spreadsheet",
        "Check progress on savings goals"
    ]
    
    if period_name:  # Monthly tasks
        month_specific = {
            'January': ["Set annual financial goals", "Review previous year's performance"],
            'April': ["Prepare tax documents", "Review Q1 performance"],
            'July': ["Mid-year financial review", "Adjust investment strategy"],
            'October': ["Q4 planning", "Holiday budget preparation"]
        }
        return month_specific.get(period_name, common_tasks)
    
    return common_tasks

def create_content_template(content_type):
    """Create a content template based on the type."""
    if content_type == "workout":
        return {
            "exercises": [],
            "duration": "45-60 minutes",
            "intensity": "Moderate",
            "equipment": [],
            "notes": []
        }
    elif content_type == "study":
        return {
            "topics": [],
            "schedule": {
                "morning": [],
                "afternoon": [],
                "evening": []
            },
            "resources": [],
            "goals": []
        }
    elif content_type == "meal":
        return {
            "meals": {
                "breakfast": [],
                "lunch": [],
                "dinner": [],
                "snacks": []
            },
            "nutrients": {
                "protein": [],
                "carbs": [],
                "fats": []
            },
            "macros": {
                "protein_target": "0g",
                "carbs_target": "0g",
                "fats_target": "0g",
                "total_calories": "0"
            },
            "tasks": [],
            "tips": [],
            "tracking": {
                "weight": "",
                "protein_intake": "",
                "water_intake": ""
            }
        }
    elif content_type == "finance":
        return {
            "budget": {
                "income": [],
                "expenses": [],
                "savings": []
            },
            "goals": {
                "savings_target": "0",
                "expense_reduction": "0",
                "investment_allocation": "0"
            },
            "tracking": {
                "current_balance": "",
                "savings_progress": "",
                "expense_categories": []
            },
            "tasks": [],
            "tips": []
        }
    else:
        return {
            "tasks": [],
            "schedule": {
                "morning": [],
                "afternoon": [],
                "evening": []
            },
            "goals": [],
            "notes": []
        }

def create_prompt(query, time_period, content_type, base_structure):
    """Create a prompt for the model to generate content."""
    
    # Add content-specific requirements
    content_requirements = ""
    if content_type == "meal":
        content_requirements = """
        For meal/diet plans:
        1. Each entry must have UNIQUE and SPECIFIC:
           - Meal options with portions and calories
           - Macro targets (protein/carbs/fats in grams)
           - Total calorie targets
           - Specific tasks and tips
        2. For each meal provide:
           - Exact ingredients and portions
           - Protein content per meal
           - Cooking instructions or prep notes
        3. Progressive changes:
           - Gradually increase/adjust portions
           - Vary meal choices each week/month
           - Adapt to seasonal ingredients
        4. Tracking metrics:
           - Set specific weight targets
           - Daily protein intake goals
           - Water intake requirements
        5. Make content contextual:
           - Monday-Friday: Quick prep meals
           - Weekends: More elaborate meals
           - Monthly: Seasonal ingredients
           - Consider holidays and events
        """
    elif content_type == "workout":
        content_requirements = """
        For workout plans:
        1. Include specific exercises with sets and reps
        2. Provide duration and intensity
        3. List required equipment
        4. Add form tips and safety notes
        5. Include progressive overload
        """
    elif content_type == "study":
        content_requirements = """
        For study plans:
        1. Break down topics into manageable chunks
        2. Include specific resources and materials
        3. Add review and practice sessions
        4. Track progress with assessments
        """
    
    prompt = f"""
    Generate a detailed {time_period} plan for: "{query}"
    Content type: {content_type}
    
    Follow this EXACT format:
    {{
        "period": "{time_period}",
        "title": "Title of the plan",
        "entries": [
            {{
                "period": "day/week/month",
                "periodName": "Name of the period",
                "date": "today's date: {datetime.now().strftime("%Y-%m-%d")}",
                "title": "Title for this entry",
                "description": {{
                    // Content specific to the type
                    // MUST BE UNIQUE for each entry
                    // NO DEFAULT VALUES
                }}
            }}
        ]
    }}
    
    Base structure to follow:
    {json.dumps(base_structure, indent=2)}
    
    {content_requirements}
    
    IMPORTANT REQUIREMENTS:
    1. Keep EXACT format - no additional fields
    2. Fill all arrays with UNIQUE, SPECIFIC items
    3. NO DEFAULT VALUES or placeholder text
    4. Make content unique for each entry
    5. Keep content focused and actionable
    6. Return ONLY valid JSON
    7. Ensure no empty arrays in description
    8. Make content progressive and build over time
    9. Include specific, measurable goals
    10. Adapt content based on the time period (day/week/month)
    11. For meal plans: Include specific portions, calories, and macros
    12. For tracking fields: Always include specific target values
    """
    
    return prompt

def generate_plan_chunk(prompt, attempt=1):
    """Generate a single chunk of the plan with error handling."""
    try:
        temperature = 0.3 if attempt == 1 else 0.4
        
        # Create a more structured prompt
        json_prompt = f"""
        You are a specialized plan generator. Create a detailed plan following these rules:

        INPUT QUERY:
        {prompt}

        RESPONSE FORMAT:
        {{
            "title": "Clear, specific title",
            "description": "Brief, focused description",
            "plan_type": "Type of plan (day/week/month/year)",
            "content": {{
                // Specific content based on the plan type
                // Use the structure provided in the input
            }},
            "metrics": [
                // 3-4 specific, measurable metrics
            ],
            "resources": [
                // 3-4 specific required resources
            ],
            "tips": [
                // 3-4 actionable recommendations
            ]
        }}

        REQUIREMENTS:
        1. Use ONLY double quotes for ALL strings
        2. Include ALL required sections
        3. Make content specific and actionable
        4. Keep lists to 3-4 items maximum
        5. Use proper JSON format
        6. No comments or extra text
        """
        
        response = model.generate_content(json_prompt, 
            generation_config={
                "temperature": 0.9,
                "top_p": 0.95,
                "top_k": 50,
                "max_output_tokens": 8192,
                "candidate_count": 1
            })
        
        # Clean and parse the response
        text = response.text.strip()
        text = text.replace('```json', '').replace('```', '').strip()
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # Ensure proper JSON structure
        if not text.startswith('{'): text = '{' + text
        if not text.endswith('}'): text = text + '}'
        
        # Fix JSON issues
        text = aggressive_json_fix(text)
        
        try:
            plan = json.loads(text)
            if not validate_plan_content(plan):
                raise ValueError("Incomplete plan content")
            return trim_content(plan)
        except (json.JSONDecodeError, ValueError) as e:
            if attempt < 2:
                return generate_plan_chunk(prompt, attempt + 1)
            else:
                # Create a more specific fallback structure
                return create_specific_fallback(prompt)
    except Exception as e:
        if attempt < 2:
            return generate_plan_chunk(prompt, attempt + 1)
        else:
            return create_specific_fallback(prompt)

def aggressive_json_fix(text):
    """Aggressively fix common JSON formatting issues."""
    # Remove any non-JSON text
    text = re.sub(r'^[^{]*{', '{', text)
    text = re.sub(r'}[^}]*$', '}', text)
    
    # Fix quote issues
    text = text.replace("'", '"')  # Replace single quotes with double quotes
    text = re.sub(r'(?<!\\)"`|`"', '"', text)  # Remove backticks
    
    # Fix property names
    text = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
    
    # Fix trailing commas
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Fix missing quotes around string values
    text = re.sub(r':\s*([a-zA-Z][a-zA-Z0-9_\s-]*[a-zA-Z0-9])([,}\]])', r':"\1"\2', text)
    
    # Fix common structural issues
    text = re.sub(r',\s*,', ',', text)  # Remove duplicate commas
    text = re.sub(r'{\s*,', '{', text)  # Remove leading commas in objects
    text = re.sub(r'\[\s*,', '[', text)  # Remove leading commas in arrays
    
    return text

def create_specific_fallback(prompt):
    """Create a more specific fallback structure based on the prompt content."""
    # Try to extract key information from the prompt
    prompt_lower = prompt.lower()
    
    # Determine the type of plan
    plan_type = "general"
    if any(word in prompt_lower for word in ["diet", "meal", "food"]):
        plan_type = "meal"
    elif any(word in prompt_lower for word in ["workout", "exercise", "fitness"]):
        plan_type = "workout"
    elif any(word in prompt_lower for word in ["study", "learn", "education"]):
        plan_type = "study"
    
    # Create specific content based on plan type
    if plan_type == "meal":
        content = {
            "meals": {
                "breakfast": ["Healthy breakfast option 1", "Healthy breakfast option 2"],
                "lunch": ["Balanced lunch option 1", "Balanced lunch option 2"],
                "dinner": ["Nutritious dinner option 1", "Nutritious dinner option 2"],
                "snacks": ["Healthy snack options"]
            }
        }
    elif plan_type == "workout":
        content = {
            "exercises": {
                "warmup": ["Basic stretches", "Light cardio"],
                "main": ["Beginner-friendly exercises"],
                "cooldown": ["Cool-down stretches"]
            }
        }
    elif plan_type == "study":
        content = {
            "schedule": {
                "morning": ["Study session 1"],
                "afternoon": ["Study session 2"],
                "evening": ["Review and practice"]
            }
        }
    else:
        content = {
            "schedule": {
                "morning": ["Morning activities"],
                "afternoon": ["Afternoon activities"],
                "evening": ["Evening activities"]
            }
        }
    
    return {
        "title": prompt.strip(),
        "description": f"Structured {plan_type} plan with progressive content",
        "plan_type": plan_type,
        "content": content,
        "metrics_to_track": [
            f"Daily {plan_type} progress",
            "Goal completion rate",
            "Consistency level"
        ],
        "resources_needed": [
            "Essential tools and materials",
            "Progress tracking method",
            "Support resources"
        ],
        "tips_and_recommendations": [
            "Start with basics",
            "Maintain consistency",
            "Track progress regularly"
        ]
    }

def trim_content(obj):
    """Ensure conciseness in the generated content."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, list):
                # Keep only the first 3-4 items in lists
                obj[key] = value[:4]
            elif isinstance(value, (dict, list)):
                trim_content(value)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                trim_content(item)
    return obj

def validate_plan_content(plan):
    """Validate that the plan has no empty arrays or missing content."""
    def check_empty(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (list, dict)):
                    if not check_empty(value):
                        return False
                elif key != "notes" and isinstance(value, str) and not value.strip():
                    return False
        elif isinstance(obj, list):
            if not obj:  # Empty array
                return False
            for item in obj:
                if isinstance(item, (dict, list)):
                    if not check_empty(item):
                        return False
        return True
    
    return check_empty(plan)

def validate_and_fill_content(entry, content_type):
    """Validate and fill any empty arrays in the entry's description."""
    if not entry.get('description'):
        entry['description'] = {}
    
    description = entry['description']
        
    # Default content templates based on type
    default_content = {
        'exercises': [
            "Warm-up exercises (10 minutes)",
            "Main workout routine",
            "Cool-down stretches"
        ],
        'equipment': [
            "Exercise mat",
            "Water bottle",
            "Comfortable clothes"
        ],
        'notes': [
            "Start with proper form",
            "Listen to your body",
            "Stay hydrated"
        ],
        'morning': [
            "Start with light activity",
            "Focus on main goals",
            "Track progress"
        ],
        'afternoon': [
            "Continue with planned activities",
            "Review progress",
            "Adjust as needed"
        ],
        'evening': [
            "Complete remaining tasks",
            "Review day's progress",
            "Plan for tomorrow"
        ],
        'goals': [
            "Complete planned activities",
            "Maintain consistent effort",
            "Track progress"
        ],
        'tasks': [
            "Priority task 1",
            "Secondary task",
            "Follow-up activities"
        ],
        'meals': {
            'breakfast': [
                "High-protein breakfast option",
                "Healthy carbs and fruits"
            ],
            'lunch': [
                "Balanced protein and vegetables",
                "Complex carbohydrates"
            ],
            'dinner': [
                "Lean protein option",
                "Vegetables and whole grains"
            ],
            'snacks': [
                "Protein-rich snack",
                "Healthy fruits or nuts"
            ]
        },
        'nutrients': {
            'protein': [
                "Track daily protein intake",
                "Aim for balanced protein distribution"
            ],
            'carbs': [
                "Focus on complex carbohydrates",
                "Monitor carb intake"
            ],
            'fats': [
                "Include healthy fats",
                "Balance fat consumption"
            ]
        },
        'tips': [
            "Stay hydrated throughout the day",
            "Prepare meals in advance",
            "Track your nutrition"
        ],
        'budget': {
            'income': [
                "Regular income sources",
                "Additional income opportunities"
            ],
            'expenses': [
                "Essential expenses",
                "Non-essential spending"
            ],
            'savings': [
                "Emergency fund allocation",
                "Long-term savings goals"
            ]
        },
        'expense_categories': [
            "Housing and utilities",
            "Food and groceries",
            "Transportation",
            "Healthcare"
        ]
    }
    
    # Fill empty arrays based on content type
    if content_type == "workout" or content_type == "fitness":
        if not description.get('exercises') or len(description['exercises']) == 0:
            description['exercises'] = default_content['exercises']
        if not description.get('equipment') or len(description['equipment']) == 0:
            description['equipment'] = default_content['equipment']
        if not description.get('notes') or len(description['notes']) == 0:
            description['notes'] = default_content['notes']
        
    elif content_type == "study" or content_type == "learn":
        if 'schedule' in description:
            schedule = description['schedule']
            if not schedule.get('morning') or len(schedule['morning']) == 0:
                schedule['morning'] = default_content['morning']
            if not schedule.get('afternoon') or len(schedule['afternoon']) == 0:
                schedule['afternoon'] = default_content['afternoon']
            if not schedule.get('evening') or len(schedule['evening']) == 0:
                schedule['evening'] = default_content['evening']
        if not description.get('goals') or len(description['goals']) == 0:
            description['goals'] = default_content['goals']
            
    elif content_type == "diet" or content_type == "meal":
        if 'meals' not in description:
            description['meals'] = {}
        meals = description['meals']
        if not meals.get('breakfast') or len(meals['breakfast']) == 0:
            meals['breakfast'] = default_content['meals']['breakfast']
        if not meals.get('lunch') or len(meals['lunch']) == 0:
            meals['lunch'] = default_content['meals']['lunch']
        if not meals.get('dinner') or len(meals['dinner']) == 0:
            meals['dinner'] = default_content['meals']['dinner']
        if not meals.get('snacks') or len(meals['snacks']) == 0:
            meals['snacks'] = default_content['meals']['snacks']
            
        if 'nutrients' not in description:
            description['nutrients'] = {}
        nutrients = description['nutrients']
        if not nutrients.get('protein') or len(nutrients['protein']) == 0:
            nutrients['protein'] = default_content['nutrients']['protein']
        if not nutrients.get('carbs') or len(nutrients['carbs']) == 0:
            nutrients['carbs'] = default_content['nutrients']['carbs']
        if not nutrients.get('fats') or len(nutrients['fats']) == 0:
            nutrients['fats'] = default_content['nutrients']['fats']
            
        if not description.get('tips') or len(description['tips']) == 0:
            description['tips'] = default_content['tips']
            
    elif content_type == "finance":
        if 'budget' not in description:
            description['budget'] = {}
        budget = description['budget']
        if not budget.get('income') or len(budget['income']) == 0:
            budget['income'] = default_content['budget']['income']
        if not budget.get('expenses') or len(budget['expenses']) == 0:
            budget['expenses'] = default_content['budget']['expenses']
        if not budget.get('savings') or len(budget['savings']) == 0:
            budget['savings'] = default_content['budget']['savings']
            
        if 'tracking' not in description:
            description['tracking'] = {}
        tracking = description['tracking']
        if not tracking.get('expense_categories') or len(tracking['expense_categories']) == 0:
            tracking['expense_categories'] = default_content['expense_categories']
            
        if not description.get('tasks') or len(description['tasks']) == 0:
            description['tasks'] = default_content['tasks']
        if not description.get('tips') or len(description['tips']) == 0:
            description['tips'] = default_content['tips']
        
    else:  # general tasks
        if 'schedule' in description:
            schedule = description['schedule']
            if not schedule.get('morning') or len(schedule['morning']) == 0:
                schedule['morning'] = default_content['morning']
            if not schedule.get('afternoon') or len(schedule['afternoon']) == 0:
                schedule['afternoon'] = default_content['afternoon']
            if not schedule.get('evening') or len(schedule['evening']) == 0:
                schedule['evening'] = default_content['evening']
        if not description.get('tasks') or len(description['tasks']) == 0:
            description['tasks'] = default_content['tasks']
    
    return entry

def generate_plan(query):
    """Generate a structured plan based on the input query."""
    # Analyze and clean the query
    analysis = analyze_query(query)
    cleaned_query = clean_query(query)
    time_period = detect_time_period(query)
    
    # Generate time-specific content directly
    plan = generate_time_content(cleaned_query, time_period, analysis)
    
    return plan

@app.route('/generate', methods=['POST'])
def generate():
    """API endpoint to generate plans."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    if 'query' not in data:
        return jsonify({"error": "Query is required"}), 400
    
    plan = generate_plan(data['query'])
    return jsonify(plan)

@app.route('/sample', methods=['GET'])
def get_sample():
    """Return sample queries and their expected output format."""
    return jsonify({
        "example_queries": [
            "Build a 7-day protein rich diet plan for muscle gain",
            "Create a month-long workout routine for weight loss",
            "Plan a year-long financial savings strategy"
        ],
        "output_format": {
            "period": "weekly/monthly/yearly",
            "title": "Title of the goal",
            "entries": [{
                "period": "day/week/month",
                "periodName": "Day name or Month name",
                "date": "Date of the entry",
                "title": "Title of the entry",
                "description": "Object or Array with details"
            }]
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)