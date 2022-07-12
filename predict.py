import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import warnings
import numpy as np
import json
import argparse

def get_input_args():
    
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--top_k', type = int, default = 5, help = 'Return top K most likely classes')
    parser.add_argument('--category_name', type = str, default = 'class_names.json', help = 'mapping of categories to real names') 
    parser.add_argument(dest='image_directory', help=' This is a image directory')
    parser.add_argument('--checkpoint', type = str, default = 'checkpoint.pth', help = 'saved checkpoint')
    
    return parser.parse_args()

in_arg = get_input_args()

def process_image(image):
    processed_image = tf.convert_to_tensor(image)
    processed_image = tf.image.resize(processed_image, (224, 224))
    processed_image /= 255
    processed_image = processed_image.numpy()
    
    return processed_image

def predict(image_path, model, top_k):
    
    from PIL import Image
    
    im = Image.open(image_path)
    im = np.asarray(im)
    
    processed_im = process_image(im)
    processed_im = np.expand_dims(processed_im, axis = 0)
    
    ps = model.predict(processed_im)
    
    probability, classes = tf.math.top_k(ps, k=top_k) 
        
    return probability[0].numpy(), classes[0].numpy()

# load checkpoint
reloaded_keras_model = tf.keras.models.load_model(in_arg.checkpoint, custom_objects={'KerasLayer':hub.KerasLayer},compile=False)

def main():
    
    in_arg = get_input_args()
    
    with open(in_arg.category_name, 'r') as f:
        class_names = json.load(f)
    
    img = in_arg.image_directory
    prob, classes = predict(img, reloaded_keras_model, in_arg.top_k)
    
    print(f"\nThe most likely class for this image is {class_names[str(classes[0]+1)]} with probability {prob[0]}. \n\n")
    
    print(f"The top {in_arg.top_k} classes with their probabilities are: \n\n")
    
    for k in range(in_arg.top_k):
        print(f"{class_names[str(classes[k]+1)]} with probability {prob[k]}.\n")

if __name__ == '__main__':
    main()