import streamlit as st
import plotly.express as px
import predict

st.set_page_config(page_title='BertEmotions', layout="centered", initial_sidebar_state="expanded")
st.title('Emotionsss')

text_input = st.text_input('Enter Text: ')

# custom css
st.markdown(
    """
    <style>
    .css-q8sbsg p {
        font-size: 18px;
    }
    .css-nahz7x p {
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

# custom threshold set by users 
selected_value = col1.slider("Select threshold for probs to preds conversion", 
    min_value=0.0, max_value=1.0, value=0.4, step=0.1)

# only if an input is given
if text_input:
    prob, binary = predict.predict_run(text_input, threshold=selected_value)

    emotions = ['admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride',
        'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']

    # only emotions with prob >= threshold
    predicted_emotions = [emotion for i, emotion in enumerate(emotions) if binary[i] == 1]  
    predicted_emotions = [emotion.capitalize() for emotion in predicted_emotions]
    predicted_emotions_str = ' / '.join(predicted_emotions)

    prob_emotions = []
    for i in range(len(emotions)):
        prob_emotions.append([emotions[i], prob[i]])

    # top 10 probabilities
    top_prob = (sorted(prob_emotions, key=lambda x: x[1], reverse=True))[:10]
    top_emotions = [entry[0] for entry in top_prob][::-1]
    top_probabilities = [entry[1] for entry in top_prob][::-1]

    with col2:
        # pred emotions
        if len(predicted_emotions):
            st.write("Predicted Emotion(s): ", predicted_emotions_str)
        else:
            st.write("No Emotions predicted :(")
            st.write("Try changing the threshold !!")

    st.markdown('<hr>', unsafe_allow_html=True)

    fig = px.bar(
        x=top_probabilities,
        y=top_emotions,
        orientation='h',
        labels={'x': 'Probability', 'y': 'Emotion'},
    )

    fig.update_layout(
        title="Top 10 Emotions - Probabilities",
        title_font=dict(size=22),
    )

    st.plotly_chart(fig)
