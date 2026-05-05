from ai_edge_litert.interpreter import Interpreter

i = Interpreter(model_path='models/traffic_light_yolov8n.tflite')
i.allocate_tensors()
inp = i.get_input_details()
out = i.get_output_details()
print("TRAFFIC LIGHT MODEL")
print("  Input: shape=" + str(inp[0]["shape"]) + " dtype=" + str(inp[0]["dtype"]))
for idx, o in enumerate(out):
    print("  Output " + str(idx) + ": shape=" + str(o["shape"]) + " dtype=" + str(o["dtype"]))

print()
i2 = Interpreter(model_path='models/coco_ssd_mobilenet_v2.tflite')
i2.allocate_tensors()
inp2 = i2.get_input_details()
out2 = i2.get_output_details()
print("OBJECT DETECTION MODEL")
print("  Input: shape=" + str(inp2[0]["shape"]) + " dtype=" + str(inp2[0]["dtype"]))
for idx, o in enumerate(out2):
    print("  Output " + str(idx) + ": shape=" + str(o["shape"]) + " dtype=" + str(o["dtype"]))

print()
print("BOTH MODELS OK")
