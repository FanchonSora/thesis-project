import urllib.request
import json

url = "http://127.0.0.1:8001/jobs/10c97283-c695-4c7a-806a-63a8e81ea1f9"
try:
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode())
        print(f"gt_wt_paths present: {'gt_wt_paths' in data}")
        if 'gt_wt_paths' in data:
            print(data['gt_wt_paths'])
except Exception as e:
    print(e)
