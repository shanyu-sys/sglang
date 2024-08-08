#!/bin/bash
# Check if mode argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <mode>"
    echo "Modes: 1:collocate, 2:swap"
    exit 1
fi

MODE=$1
# Define models and corresponding ports
declare -A models_ports


if [ "$MODE" == "1" ]; then
    models_ports=(
        ["meta-llama/Llama-2-7b-chat-hf"]="30000"
        ["mistralai/Mistral-7B-Instruct-v0.2"]="20000"
    )
elif [ "$MODE" == "2" ]; then
    models_ports=(
        ["meta-llama/Llama-2-7b-chat-hf"]="30000"
    )
elif [ "$MODE" == "3" ]; then
    models_ports=(
        ["mistralai/Mistral-7B-Instruct-v0.2"]="20000"
    )
else
    echo "Invalid mode: $MODE"
    echo "Modes: 1: collocate; 2: Llama-2-7b-chat; 3: Mistral-7B-Instruct"
    exit 1
fi

# Function to launch a server
launch_server() {
    local model_path=$1
    local port=$2
    local mem_fraction=$3
    local log_file="logs/server_${port}.log"

    if [ -z "$mem_fraction" ]; then
        echo "Launching server for model ${model_path} on port ${port}..."
        cmd="python -m sglang.launch_server --model-path ${model_path} --port ${port}  --disable-cuda-graph --load-format dummy > ${log_file} 2>&1 &"
        echo $cmd
        # python -m sglang.launch_server --model-path ${model_path} --port ${port}  --disable-cuda-graph --load-format dummy > ${log_file} 2>&1 &
    else
        echo "Launching server for model ${model_path} on port ${port} with memory fraction ${mem_fraction}..."
        cmd="python -m sglang.launch_server --model-path ${model_path} --port ${port} --mem-fraction-static ${mem_fraction} --disable-cuda-graph --load-format dummy > ${log_file} 2>&1 &"
        echo $cmd
        # python -m sglang.launch_server --model-path ${model_path} --port ${port} --mem-fraction-static ${mem_fraction} --disable-cuda-graph --load-format dummy > ${log_file} 2>&1 &
    fi
    # execute cmd
    eval $cmd
    echo $! > "server_${port}.pid" # Save the PID to a file
    echo "Server for model ${model_path} launched on port ${port} with PID $!"
}

#  Loop through models and ports and launch servers
if [ "$MODE" == "1" ]; then
    index=0
    for model in "${!models_ports[@]}"; do
        port=${models_ports[$model]}
        if [ $index -eq 0 ]; then
            launch_server ${model} ${port} 0.4
        else
            launch_server ${model} ${port} 0.8
        fi
        index=$((index + 1))
        sleep 5 # Wait a bit for the server to start
    done
else
    for model in "${!models_ports[@]}"; do
        port=${models_ports[$model]}
        launch_server ${model} ${port} 0.7
        sleep 5 # Wait a bit for the server to start
    done
fi

# Function to print server logs
print_logs() {
    local port=$1
    local log_file="logs/server_${port}.log"

    echo "=== Logs for server on port ${port} ==="
    tail -f ${log_file}
}

echo "All servers launched. Use 'tail -f logs/server_<port>.log' to monitor logs."
