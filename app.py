from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from model_utils import build_feature_frame

BASE = Path(__file__).resolve().parent
MODEL = BASE / 'phishing_pipeline.joblib'

st.set_page_config(page_title='Phishing Detection Prototype', page_icon='🛡️', layout='centered')


@st.cache_resource
def load_model():
    if not MODEL.exists():
        raise FileNotFoundError('Model file not found. Run train_model.py first.')
    return joblib.load(MODEL)


def predict_text(model, text: str):
    feat_df = build_feature_frame([text])
    pred = int(model.predict(feat_df)[0])
    proba = float(model.predict_proba(feat_df)[0][1])
    return pred, proba, feat_df.iloc[0].to_dict()


st.title('Phishing Detection Prototype')
st.write('Paste an email, message, or URL. The system predicts whether it looks legitimate or phishing.')

sample = st.selectbox(
    'Quick test',
    [
        'Custom input',
        'Urgent: Your account has been suspended. Click here to verify your password immediately: http://secure-login-alert.com',
        'Reminder: submit your coursework by 4pm on Monday.',
        'You won a free iPhone. Claim your prize today at http://bit.ly/freegift'
    ]
)

text = st.text_area(
    'Message or URL',
    value='' if sample == 'Custom input' else sample,
    height=180,
    placeholder='Example: Verify your account now at http://example.com'
)

if st.button('Analyse'):
    if not text.strip():
        st.warning('Enter some text or a URL first.')
    else:
        try:
            model = load_model()
            pred, proba, features = predict_text(model, text)
            label = 'Phishing' if pred == 1 else 'Legitimate'
            confidence = proba if pred == 1 else (1 - proba)

            st.subheader(f'Result: {label}')
            st.progress(min(max(confidence, 0.0), 1.0))
            st.write(f"Confidence: **{confidence:.1%}**")

            if pred == 1:
                st.error('This input shows phishing-like patterns. Review carefully before clicking links or sharing details.')
            else:
                st.success('This input looks more like a legitimate message, but human review is still recommended.')

            st.markdown('### Why the model reacted this way')
            reasons = []
            if features.get('suspicious_word_count', 0) > 0:
                reasons.append(f"Suspicious words detected: {int(features['suspicious_word_count'])}")
            if features.get('url_count', 0) > 0:
                reasons.append(f"Contains URL(s): {int(features['url_count'])}")
            if features.get('has_ip_url', 0):
                reasons.append('Uses an IP address in the URL')
            if features.get('uses_shortener', 0):
                reasons.append('Uses a shortened URL')
            if features.get('exclamation_count', 0) > 0:
                reasons.append(f"Exclamation marks: {int(features['exclamation_count'])}")

            if not reasons:
                reasons.append('No strong phishing indicators were found in the basic feature set.')

            for r in reasons:
                st.write(f'- {r}')

            with st.expander('Feature details'):
                feature_table = pd.DataFrame([features]).T.rename(columns={0: 'value'})
                feature_table['value'] = feature_table['value'].astype(str)
                st.dataframe(feature_table)

        except Exception as e:
            st.exception(e)

st.markdown('---')
st.caption('Prototype for dissertation demonstration: Python + scikit-learn + Streamlit.')