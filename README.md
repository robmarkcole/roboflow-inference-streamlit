# roboflow-inference-streamlit
Perform inference on a roboflow hosted custom model via a streamlit app

Fork of https://github.com/matthewbrems/streamlit-bccd

* Create and activate a venv: `python3 -m venv venv` and `source venv/bin/activate`
* Install requirements: `pip3 install -r requirements.txt`
* Export required environment variables: `export DEEPSTACK_CUSTOM_MODEL='mask'`
* Run streamlit from `app` folder: `streamlit run app.py`