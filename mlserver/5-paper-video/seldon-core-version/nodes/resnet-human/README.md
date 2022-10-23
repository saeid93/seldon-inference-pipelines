TODO add handing of zero person in the pipeline

Make sure the yolo cache is in the folder

if you are using the mlser server throught command line like:
```
mlserver start .
```
then use the following `model-settings.json` (include batching variable):
```json
{
    "name": "resnet",
    "implementation": "models.ResnetHuman",
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
    "name": "resnet",
    "implementation": "models.ResnetHuman",
    "parameters": {
        "uri": "./fakeuri"
    }
}
```