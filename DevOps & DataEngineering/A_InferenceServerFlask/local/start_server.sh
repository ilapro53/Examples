echo
echo server started
nohup tensorflow_model_server --rest_api_port=8501 --model_name="my_model" --model_base_path="/tmp/tfsrver/models/my_model/" >server.log 2>&1
