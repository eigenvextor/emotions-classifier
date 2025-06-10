import torch
import config, model

def predict_run(sentence, threshold=0.5):
    
    # load the model (cpu here)
    device = torch.device(config.DEVICE)
    bert_model = model.BERTMultiLabel()
    # load_state_dict takes only dict 
    bert_model.load_state_dict(torch.load(config.MODEL_PATH + '/bert-weights.pt', map_location=device))
    bert_model.to(device)
    
    # evaluation mode
    bert_model.eval()
    
    tokenized_sentence = config.TOKENIZER(
        sentence, 
        truncation=True,
        padding=True,
        return_tensors='pt',
        max_length=config.MAX_LEN
    )
    
    tokenized_sentence.to(device)
    
    with torch.no_grad():
        outputs = bert_model(
            ids = tokenized_sentence['input_ids'],
            mask = tokenized_sentence['attention_mask'],
            token_type_ids = tokenized_sentence['token_type_ids'],
        )
    
    # logits to probs
    outputs = torch.sigmoid(outputs).detach().cpu()
    # custom threshold (Default=0.5)
    binary_predictions = (outputs>=threshold).int().tolist()
        
    return outputs.numpy()[0], binary_predictions[0]

if __name__ == '__main__':
    ### for terminal run
    sentence = input('type a sentence: ')
    prob, binary = predict_run(sentence)

    emotions = ['admiration', 'amusement', 'anger', 'annoyance', 'approval',
       'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
       'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
       'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride',
       'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
    
    predicted_emotions = [emotion for i, emotion in enumerate(emotions) if binary[i] == 1]
    predicted_emotions

    print('\npredicted emotion(s) [threshold default = 50%]: ', end=' ')
    for emotion in predicted_emotions:
        print(emotion, end=' ')

    prob_emotions = []
    for i in range(len(emotions)):
        prob_emotions.append([emotions[i], prob[i]])
    
    prob_emotions = sorted(prob_emotions, key=lambda x: x[1], reverse=True)

    print('\ntop 5 emotions: ')
    for emotion, prob in prob_emotions[:5]:
        print(f'{emotion}: {prob:4f}')


    