A simple linear pipeline


Hitting the endpoint blueprint

```
http://<ingress_url>/seldon/<namespace>/<model-name>/api/v1.0/doc/
```
Example:
```
 curl -d '{"data": {"ndarray":[[1.0, 2.0, 5.0, 6.0]]}}' -X POST http://localhost:32000/seldon/seldon/inferline/api/v1.0/predictions -H "Content-Type: application/json"
```