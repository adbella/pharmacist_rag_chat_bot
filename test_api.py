import requests
import json

def test_chat():
    url = "http://localhost:8000/chat"
    payload = {
        "query": "눈이 침침한데 어떤 영양제가 좋을까요?",
        "model": "gpt-5",
        "top_k": 5,
        "ensemble_k": 20,
        "weight_bm25": 0.8,
        "use_self_correction": True
    }
    
    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, json=payload, stream=True)
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return

        print("--- SSE Stream Start ---")
        event_type = None
        for line in response.iter_lines():
            if not line:
                continue
            
            line_str = line.decode('utf-8')
            if line_str.startswith("event: "):
                event_type = line_str[7:]
            elif line_str.startswith("data: "):
                data = json.loads(line_str[6:])
                print(f"[{event_type}] {data}")
                event_type = None
        print("--- SSE Stream End ---")
        
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    test_chat()
