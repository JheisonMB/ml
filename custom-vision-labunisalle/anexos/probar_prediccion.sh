#!/bin/bash
# Prueba del modelo publicado AnimalIdentification-v1.
# La clave se exporta fuera del script: export PKEY="..."
ENDPOINT="https://labunisalle-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/8745a5dd-92de-4fd7-967f-642ce3967c6c/classify/iterations/Iteration1/image"
curl -s -X POST "$ENDPOINT" \
  -H "Prediction-Key: $PKEY" \
  -H "Content-Type: application/octet-stream" \
  --data-binary "@validacion/lion/lion_val_01.jpg"
echo
curl -s -X POST "$ENDPOINT" \
  -H "Prediction-Key: $PKEY" \
  -H "Content-Type: application/octet-stream" \
  --data-binary "@validacion/giraffe/giraffe_val_02.jpg"
echo
