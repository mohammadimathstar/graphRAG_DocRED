#!/bin/bash

if [ -f prefect.yaml ]; then
    rm prefect.yaml
fi

# prefect init

prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api

# Start server in background
prefect server start --host 0.0.0.0 --port 4200 &

sleep 10

# Start worker in background
prefect worker start -p mypool -t process &

sleep 5

# Run deployment command and automatically answer 'n' to prompt
echo "n" | prefect deploy monitoring_pipeline.py:monitor_model -n mydeployment -p mypool

# Then run the pipeline deployment manually
prefect deployment run 'monitor-model/mydeployment'


# Keep container alive to keep services running
tail -f /dev/null
