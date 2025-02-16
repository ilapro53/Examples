echo
echo server started
docker run -t --rm -p 8501:8501 -v "/mnt/o/study/ml_6/8.5/docker/my_model:/models/my_model" -e MODEL_NAME=my_model tensorflow/serving
