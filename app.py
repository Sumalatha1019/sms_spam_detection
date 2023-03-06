from predictor import predict, load_model
import streamlit as st

# set page config
st.set_page_config(
	page_title="Analyze Your Message",
	page_icon="🏨"
)

# load model
with st.spinner("Please wait ..."):
	model = load_model()

@st.cache
def handle_text(text):
	# predict
	prediction = predict(model, text)

	# return
	return prediction

# title and subtitle
st.title("SMS Spam detector")

# user input
user_review = st.text_area(
	label="Message:",
	help="Enter message..."
)

if user_review != "":
	prediction = handle_text(user_review)

	# display prediction
	#st.subheader("AI thinks that ...")

	# check prediction
	if prediction > 0.5:
		st.write(f"Spam Message")
	else:
		st.write(f"Ham Message")
