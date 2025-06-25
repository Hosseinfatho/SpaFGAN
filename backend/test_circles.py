import requests
import json

def test_circle_generation():
    """Test the circle generation endpoint"""
    url = "http://localhost:5000/api/update_config_with_circles"
    
    # Test data
    data = {
        "showCircles": True,
        "interaction": "T-cell entry site"
    }
    
    try:
        print("Sending request to:", url)
        print("Data:", json.dumps(data, indent=2))
        
        response = requests.post(url, json=data)
        
        print("Response status:", response.status_code)
        print("Response headers:", dict(response.headers))
        print("Response body:", response.text)
        
        if response.status_code == 200:
            result = response.json()
            print("Success:", result)
        else:
            print("Error:", response.text)
            
    except Exception as e:
        print("Exception:", str(e))

if __name__ == "__main__":
    test_circle_generation() 