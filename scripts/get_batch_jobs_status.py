import json

import requests

INSTRUMENT = "swe"
DATA_LEVEL = "l3"

url = 'https://api.dev.imap-mission.com/batch-job'
response = requests.get(url=url, params={'instrument': INSTRUMENT, 'data_level': DATA_LEVEL})
logs = json.loads(json.dumps(response.json()))
reverse_logs = sorted(logs, key=lambda x: x['started_at'] or "")
print(json.dumps(reverse_logs, indent=2))
