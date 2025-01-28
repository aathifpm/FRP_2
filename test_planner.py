import requests
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

def test_plan_generation(query):
    """Test plan generation with a specific query."""
    url = "http://localhost:5000/generate"
    payload = {"query": query}
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=30)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            return {
                "query": query,
                "success": True,
                "time_taken": round(end_time - start_time, 2),
                "result": result
            }
        else:
            return {
                "query": query,
                "success": False,
                "time_taken": round(end_time - start_time, 2),
                "error": f"Status code: {response.status_code}"
            }
    except requests.exceptions.Timeout:
        return {
            "query": query,
            "success": False,
            "error": "Request timed out after 30 seconds"
        }
    except Exception as e:
        return {
            "query": query,
            "success": False,
            "error": str(e)
        }

def print_test_result(result):
    """Print test result in a clean format."""
    print("\n" + "="*80)
    print(f"Query: {result['query']}")
    print(f"Time taken: {result.get('time_taken', 'N/A')} seconds")
    print("-"*80)
    
    if result['success']:
        plan = result['result']
        print(f"Period: {plan.get('period', 'N/A')}")
        print(f"Title: {plan.get('title', 'N/A')}")
        print(f"Number of entries: {len(plan.get('entries', []))}")
        
        print("\nComplete Plan Details:")
        print(json.dumps(plan, indent=2))
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    print("="*80 + "\n")

def main():
    # Test different time periods and query types
    test_queries = [
        # Diet Plans

        
        # Fitness Goals
        "Design a week-long home workout routine", 
        "Create a month-long fitness plan",
        "Plan a year of fitness goals and milestones",
        
        # Study Plans
        "Make a daily IELTS study schedule",
        "Plan a week of coding practice",
        "Design a year-long learning roadmap",
        
        # Financial Plans
        "Create a weekly budget plan",
        "Plan monthly savings strategy",
        "Make a yearly financial planning strategy",
        
        # Quick Plans
        "Plan tomorrow's workout",
        "Create a daily study plan",
        
        # Long-term Plans
        "Create a 12-month career development plan",
        "Plan a year of personal growth goals",
        "Design a yearly business strategy"
    ]
    
    print("Starting AI Planner Tests...")
    print(f"Testing {len(test_queries)} queries")
    print("Note: Each test has a 30-second timeout")
    print("\nFull output will be shown for each test")
    
    start_time = time.time()
    results = []
    
    # Test queries sequentially for clearer output
    for query in test_queries:
        result = test_plan_generation(query)
        results.append(result)
        print_test_result(result)
        print("Press Enter to continue to next test...")
        input()
    
    end_time = time.time()
    
    # Print summary
    print("\nTest Summary:")
    print(f"Total time: {round(end_time - start_time, 2)} seconds")
    print(f"Successful tests: {sum(1 for r in results if r['success'])}/{len(test_queries)}")
    print(f"Average time per test: {round((end_time - start_time)/len(test_queries), 2)} seconds")
    
    # Print timing statistics
    successful_times = [r['time_taken'] for r in results if r['success'] and 'time_taken' in r]
    if successful_times:
        print(f"Fastest response: {min(successful_times)} seconds")
        print(f"Slowest response: {max(successful_times)} seconds")
        print(f"Average response: {round(sum(successful_times)/len(successful_times), 2)} seconds")

if __name__ == "__main__":
    print("Make sure the Flask server is running (python milestone_generator.py)")
    input("Press Enter to begin tests...")
    main() 