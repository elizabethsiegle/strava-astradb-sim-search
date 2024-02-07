from astrapy.db import AstraDB
import base64
from datasets import load_dataset
from dotenv import load_dotenv
import instructor
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import AstraDB as astra
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from openai import OpenAI
import os
from pydantic.main import BaseModel
import requests
import streamlit as st


class RunDetail(BaseModel):
  WorkoutName: str
  Distance: str
  Pace: str
  Time: str
  Achievements: int

class RideDetail(BaseModel):
  WorkoutName: str
  Distance: str
  ElevGain: str
  Time: str
  Achievements: str

class SwimDetail(BaseModel):
  WorkoutName: str
  Distance: str
  Time: str
  Pace: str

class GenWorkout(BaseModel):
  WorkoutName: str

def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

# Load API secrets
load_dotenv()
ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.environ.get("ASTRA_DB_API_ENDPOINT")
ASTRA_COLLECTION_NAME = "strava_activities"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Initialization
db = AstraDB(
  token=ASTRA_DB_APPLICATION_TOKEN,
  api_endpoint=ASTRA_DB_API_ENDPOINT)

print(f"Connected to Astra DB: {db.get_collections()}")


# Set up Streamlit app
def main():
    st.image("logosimg.png", width=400)
    # st.write("🏃‍♀️🚴‍♀️")
    st.header("Analyze a workout and search other workouts!")
    st.write("This Python🐍 web🕸️ app is built👩🏻‍💻 w/ [Streamlit](https://streamlit.io/), [OpenAI's new Vision👀 API](https://platform.openai.com/docs/guides/vision), [LangChain](https://www.langchain.com/)⛓️, [Strava API](https://developers.strava.com/), and [Astra DB](https://www.datastax.com/products/datastax-astra)")
    # User input: PDF file upload
    strava_img= st.file_uploader("Upload a screenshot of a Strava workout!🔥⤵️", type=["png", "jpg","jpeg"])
    ret_workouts_specific_input = st.text_area("workout parameters to return from your Strava", placeholder= "runs with moving_time over 4000 and total_elevation_gain over 150")
    if strava_img is not None and ret_workouts_specific_input is not None and st.button('enter✅'):
        st.image(strava_img)
        
        with st.spinner('Processing image📈...'):
            base64_image = encode_image(strava_img)

            headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
            }
            # gpt4-vision
            payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "What’s in this image?"
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                    }
                ]
                }
            ],
            "max_tokens": 300
            }
            response = requests.post("https://api.openai.com/v1/chat/completions",   headers=headers, json=payload)
            print(response)

            client = instructor.patch(OpenAI())

            response_json = response.json()
            content = response_json['choices'][0]['message']['content']
            print(f'content {content}')

            detect_workout = client.chat.completions.create(
                model="gpt-4",
                response_model=GenWorkout,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Extract the workout name from the following json:" + content}
                ]
            )
            resp_model = ''
            workout_prompt = ''
            if "run" in str(detect_workout):
                resp_model = RunDetail
                workout_prompt = "Extract distance, pace, time, and achievements from the following json: " + content
            elif "swim" in str(detect_workout):
                resp_model = SwimDetail
                workout_prompt = "Extract distance, time, and pace from the following json: " + content
            elif "ride" in str(detect_workout):
                resp_model = RideDetail
                workout_prompt = "Extract distance, time, and elevation gain from the following json: " + content
            else:
                resp_model = RunDetail
                workout_prompt = "Extract distance, pace, time, and achievements from the following json: "+ content

            workout_info = client.chat.completions.create(
                model="gpt-4",
                response_model=resp_model,
                messages=[
                    {
                        "role": "user", "content": workout_prompt
                    }
                ]
            )
            st.write(workout_info) #extract from image

            strava_huggingface_dataset = load_dataset("lizziepika/strava_activities_runs")["train"]
            print(f"An example entry from Hugging Face dataset: {strava_huggingface_dataset[0]}")

            docs = []
            for entry in strava_huggingface_dataset:
                metadata = {"moving_time": entry["moving_time"], "total_elevation_gain": entry["total_elevation_gain"]}
            
                # Add a LangChain document with the name and metadata tags
                doc = Document(page_content=entry["name"], metadata=metadata)
                docs.append(doc)
            print(f'docs {docs}') 
            
            embedding_function = OpenAIEmbeddings()
            vstore = astra(
                embedding=embedding_function,
                collection_name="test",
                api_endpoint=ASTRA_DB_API_ENDPOINT,
                token=ASTRA_DB_APPLICATION_TOKEN,
            )
            inserted_ids = vstore.add_documents(docs) # add Strava data to vector store
            print(f"\nInserted {len(inserted_ids)} documents.")

            result = vstore.similarity_search(ret_workouts_specific_input, k = 3) # "return workouts with moving_time over 4000 and total_elevation_gain over 150"
            print(f'result {result}')
            
            gen_dm = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful athletic trainer"},
                    {"role": "user", "content": f"Recommend a workout for someone who has wants workouts similar to the following workouts + {str(result)} and this one as well {workout_info}. Be specific about goals, times, and distances"}
                ]
            )
            werkout = gen_dm.choices[0].message.content
            
            html_str = f"""
            <p style="font-family:Arial; color:Pink; font-size: 16px;">Workout recommendation: {werkout}</p>
            """
        st.markdown(html_str, unsafe_allow_html=True)
        st.write("Made w/ ❤️ in SF 🌁")
        st.write("✅ out the [code on GitHub](https://github.com/elizabethsiegle/strava-astradb-sim-search/blob/main/app2.py)")


if __name__ == "__main__":
    main()