import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import requests
from PIL import Image
import warnings
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from flask import Flask, request, jsonify
import base64
import io
import HandwritingToScore as score
import WordMatching as wm



warnings.filterwarnings("ignore")
sys.path.append('../src')
from ocr import page, words, characters

#plt.rcParams['figure.figsize'] = (15.0, 10.0)

app = Flask(__name__)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")



@app.route('/api/handwriting', methods=['POST'])
def execute_handwriting() :
    
    data = request.get_json()
    base64_image = data['image']
    title = request.get_json().get("title")
    # Decode base64 image string to bytes
    image_bytes = base64.b64decode(base64_image)
    img = Image.open(io.BytesIO(image_bytes))
    #numpy_array = np.array(img) 

    np_array = np.array(img)
    cv2_image = cv2.cvtColor(np_array, cv2.COLOR_RGB2BGR)

    alpha = 1.5
    # control brightness by 50
    beta = 50
    image2 = cv2.convertScaleAbs(cv2_image, alpha = alpha, beta = beta)
    crop = page.detection(image2)
        #implt(crop)
    boxes = words.detection(crop)
        # Find the minimum bounding rectangle that encloses all the detected boxes
    min_x = min(box[0] - box[0] for box in boxes)
    min_y = min([box[1] for box in boxes])
        #max_x = max([box[0] + box[2] for box in boxes])
        #max_y = max([box[1] + box[3] for box in boxes])
    max_x = max([box[0] + box[2] for box in boxes])
    max_y = max([box[3] for box in boxes])

        # Crop the image using the coordinates of the minimum bounding rectangle
    cropped_image = cv2_image[min_y:max_y, min_x:max_x]
        # Save the result
    cv2.imwrite('cropped_with_boxes.png', cropped_image)
    img = Image.fromarray(cropped_image)
    cv2_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)

        # ## Prepare image for model
        # 
        # Next, we use `TrOCRProcessor` (which wraps a `ViTFeatureExtractor` and a `RobertaTokenizer` into one) to resize + normalize the image for the model.

            
        # calling the processor is equivalent to calling the feature extractor
    pixel_values = processor(cv2_image, return_tensors="pt").pixel_values
        #print(pixel_values.shape)

        # ## Load model
        # 
        # Here we load a TrOCR model from the [hub](https://huggingface.co/models?other=trocr). TrOCR models are instances of [`VisionEncoderDecoderModel`](https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder), which combine a vision encoder (like ViT, BEiT, DeiT, ...) with a language model as decoder (like BERT, RoBERTa, GPT-2, ...).
        # ## Generate text
        # 
        # Finally, we can generate text autoregressively using the `.generate()` method. We use the tokenizer part of the `processor` to decode the generated id's back to text. Note that by default, greedy search is used for generation, but there are more fancy methods like beam search and top-k sampling, which are also supported. You can check out [this blog post](https://huggingface.co/blog/how-to-generate) for details.
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("generated_text", generated_text)


    real_and_transcribed_words, mapped_words_indices = score.matchSampleAndRecordedWords(real_text = title, handwritten_transcript = generated_text)
    print("real_and_transcribed_words", real_and_transcribed_words)
    current_words_handwritten_accuracy1, current_words_handwritten_accuracy2 = score.getHandwritingAccuracy(real_and_transcribed_words)

    print("current_words_handwritten_accuracy1", current_words_handwritten_accuracy1)
    print("current_words_handwritten_accuracy2", current_words_handwritten_accuracy2)


    real_transcripts = ' '.join(
                [word[0] for word in real_and_transcribed_words])
    matched_transcripts = ' '.join(
                [word[1] for word in real_and_transcribed_words])

            
    words_real = title.split()
    mapped_words = matched_transcripts.split()

    print("mapped_words", mapped_words)

    is_letter_correct_all_words = ''
    for idx, word_real in enumerate(words_real):

        is_letter_correct = wm.getWhichLettersWereTranscribedCorrectly(
                word_real, mapped_words[idx])
                
        is_letter_correct_all_words += ''.join([str(is_correct)
                                                        for is_correct in is_letter_correct]) + ' '
                

    print("is_letter_correct_all_words", is_letter_correct_all_words)

    binary_txt = is_letter_correct_all_words.split()
    print("handwritten_txt", mapped_words)
    print("binary_txt", binary_txt)
    for idx in range(len(mapped_words)) :
        while len(mapped_words[idx]) < len(binary_txt[idx]):
            mapped_words[idx] = mapped_words[idx] +'_'


    print("handwritten_txt", mapped_words)
    print("binary_txt", binary_txt)

    hand_txt = ' '.join(
                [word for word in mapped_words])

    b_txt = ' '.join(
                [word for word in binary_txt])

    print("hand_txt", mapped_words)
    print("b_txt", b_txt)
    words_result_html = "<div>"
    for char_result in list(zip(hand_txt, b_txt)):
            if char_result[1] == '1':
                words_result_html += "<span style= '" + "color:green" + " ' >" + char_result[0] + "</span>"
            else:
                words_result_html += "<span style= ' " + "color:red" + " ' >" + char_result[0] + "</span>"

    words_result_html += "</div>"

    print('html:', words_result_html)
            
    resultat = {'handwritten_text': generated_text,
                            'handwritten_accuracy': current_words_handwritten_accuracy1,
                            'real_text': title, 
                            'is_letter_correct_all_words': is_letter_correct_all_words,
                            'handwritten_result_html' : words_result_html}
    return resultat

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=False,port=5000)
