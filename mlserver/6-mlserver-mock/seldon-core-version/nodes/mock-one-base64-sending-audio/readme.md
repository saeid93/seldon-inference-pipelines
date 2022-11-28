if you are using the mlser server throught command line like:
```
mlserver start .
```
then use the following `model-settings.json` (include batching variable):
```json
{
    "name": "mock-one",
    "implementation": "models.MockOne",
    "max_batch_size": 5,
    "max_batch_time": 1,
    "parameters": {
        "uri": "./fakeuri"
    }
}
```
if you are are compling the mlserver then use the follwoing (remove batching variables):
```json
{
    "name": "mock-one",
    "implementation": "models.MockOne",
    "parameters": {
        "uri": "./fakeuri"
    }
}
```