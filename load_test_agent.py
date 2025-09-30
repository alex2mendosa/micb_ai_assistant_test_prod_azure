# load_test_agent.py
import concurrent.futures
import time
from __config_2 import embedding_main, llm, vector_store_fin_law_hq, vector_store_fin_law
from __agent_workflow_2 import agent, get_config_for_thread
from langchain_core.messages import HumanMessage
import uuid
import sys

def simulate_user(user_id, question):
    """Simulate one user making a query"""
    start_time = time.time()
    
    try:
        thread_id = str(uuid.uuid4())
        config = get_config_for_thread(thread_id)
        
        result = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config
        )
        
        elapsed = time.time() - start_time
        print(f"User {user_id}: Completed in {elapsed:.2f}s", flush=True)
        return {"user_id": user_id, "success": True, "time": elapsed}
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"User {user_id}: FAILED - {str(e)}", flush=True)
        return {"user_id": user_id, "success": False, "time": elapsed, "error": str(e)}

def load_test(num_users=10):
    """Run load test"""
    
    questions = [
        "Care sunt funcțiile principale ale Băncii Naționale a Moldovei?",
        "Cum se calculează amenda aplicată de Banca Națională a Moldovei pentru încălcări?",
        "Ce condiții trebuie să îndeplinească o instituție pentru a obține licență bancară de la Banca Națională a Moldovei?",
        "Care sunt cerințele de capital minim pentru bănci stabilite de Banca Națională a Moldovei?",
        "Ce sancțiuni poate aplica Banca Națională a Moldovei pentru nerespectarea regulamentelor?",
        "Cum se reglementează ratele dobânzilor la creditele bancare de către Banca Națională a Moldovei?",
        "Care sunt obligațiile de raportare ale băncilor către Banca Națională a Moldovei?",
        "Ce proceduri trebuie urmate pentru revocarea licenței bancare de către Banca Națională a Moldovei?",
        "Cum protejează Banca Națională a Moldovei drepturile consumatorilor de servicii financiare?",
        "Care sunt cerințele privind rezervele obligatorii ale băncilor stabilite de Banca Națională a Moldovei?"
    ]

    
    print(f"\n{'='*60}", flush=True)
    print(f"Starting load test with {num_users} concurrent users...", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    start = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
        futures = [
            executor.submit(simulate_user, i, questions[i % len(questions)])
            for i in range(num_users)
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    total = time.time() - start
    successful = sum(1 for r in results if r["success"])
    failed = num_users - successful
    avg_time = sum(r["time"] for r in results) / len(results)
    
    print(f"\n{'='*60}", flush=True)
    print(f"LOAD TEST RESULTS:", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Total Users:        {num_users}", flush=True)
    print(f"Successful:         {successful}", flush=True)
    print(f"Failed:             {failed}", flush=True)
    print(f"Avg Response Time:  {avg_time:.2f}s", flush=True)
    print(f"Total Duration:     {total:.2f}s", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # Print failed requests details
    if failed > 0:
        print("Failed Requests:", flush=True)
        for r in results:
            if not r["success"]:
                print(f"  User {r['user_id']}: {r.get('error', 'Unknown error')}", flush=True)

if __name__ == "__main__":
    # Get number of users from command line or default to 10
    num_users = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    load_test(num_users=num_users)
