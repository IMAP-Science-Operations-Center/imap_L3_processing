import json

import requests

INSTRUMENT = "hi"
DATA_LEVEL = "l2"
DESCRIPTOR = 'lo-direct-events'

url = 'https://api.dev.imap-mission.com/batch-job'
response = requests.get(url=url, params={'instrument': INSTRUMENT, 'data_level': DATA_LEVEL})
logs = json.loads(json.dumps(response.json()))
# reverse_logs = sorted(logs, key=lambda x: x['started_at'] or "")
# filtered_logs = list(filter(lambda x: x["descriptor"] == DESCRIPTOR, logs))
print(json.dumps(logs, indent=2))
