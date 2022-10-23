while true
    sleep 0.05
    do
        curl -d '{"data": {"ndarray":[[1.0, 2.0, 5.0, 6.0]]}}'\
        -X POST http://localhost:32000/seldon/seldon/linear-pipeline-separate-pods/api/v1.0/predictions\
        -H "Content-Type: application/json"
    done