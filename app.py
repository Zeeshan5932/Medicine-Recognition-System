import os
from dotenv import load_dotenv
from flask import Flask, request, render_template
from google import genai
from google.genai import types

load_dotenv()

app = Flask(__name__)

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def gen_image(prompt, image_bytes, mime_type):
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[
            prompt,
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type,
            ),
        ],
    )
    return response.text


def validate(response_text):
    validation_prompt = f"""
    Check whether the following generated description is related to the medical field.
    Reply only with Yes or No.

    Description:
    {response_text}
    """

    vresponse = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=validation_prompt
    )
    return vresponse.text.strip()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            image_prompt = """
            Generate a very detailed medical description for the given image.
            Identify and describe any relevant medical conditions, anomalies, or abnormalities present in the image.
            Additionally, provide insights into any potential treatments or recommended actions based on the observed medical features.
            Please ensure the generated content is accurate and clinically relevant.
            Please don't provide false and misleading information.
            """

            uploaded_file = request.files['file']

            if not uploaded_file or uploaded_file.filename == '':
                return render_template('index.html', response_text="Please upload an image.")

            image_bytes = uploaded_file.read()
            mime_type = uploaded_file.mimetype or "image/jpeg"

            response_text = gen_image(image_prompt, image_bytes, mime_type)
            vans = validate(response_text)

            if vans.lower() == "yes":
                return render_template('index.html', response_text=response_text)
            else:
                return render_template('index.html', response_text="Please provide a valid medical image.")

        except Exception as e:
            return render_template('index.html', response_text=f"Error: {str(e)}")

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)