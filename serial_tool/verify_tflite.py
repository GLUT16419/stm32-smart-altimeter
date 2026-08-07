import tensorflow as tf
import os

models_dir = 'models'
bp_files = [f for f in os.listdir(models_dir) if f.endswith('_bp.tflite')]

for f in sorted(bp_files):
    path = os.path.join(models_dir, f)
    size = os.path.getsize(path)
    try:
        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        print(f'  {f:35s} | {size/1024:6.1f} KB | in={inp["shape"]} out={out["shape"]}')
    except Exception as e:
        print(f'  {f:35s} | FAILED: {str(e)[:60]}')
