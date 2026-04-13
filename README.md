# Medicine-Recognition-System

This FastAPI web application uses Google Gemini models to generate detailed medical descriptions from uploaded images. After upload, the backend performs file validation, generates a clinically focused description, and runs a medical relevance check before presenting results. The interface supports image preview, loading states, and clear success/error feedback. API credentials are loaded from environment variables.

## Built With

 - FastAPI
 - Google GenAI
 - Jinja2 Templates

## Getting Started

Use one of the options below to run this project.

## Installation Steps

### Option 1: Installation from GitHub

Follow these steps to install and run the project from source code:

1. **Clone the repository**
   ```bash
   git clone https://github.com/Zeeshan5932/Medicine-Recognition-System.git
   cd Medicine-Recognition-System
   ```

2. **Create and activate a virtual environment** (recommended)
   - Windows (PowerShell):
     ```powershell
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - Linux/macOS:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set your API key in `.env`**
   ```dotenv
   GOOGLE_API_KEY=your_api_key_here
   ```

5. **Run the app**
   ```bash
   uvicorn app:app --reload
   ```

6. **Open in browser**
   - Go to `http://127.0.0.1:8000`.

### Option 2: Installation from DockerHub

If you publish or share a Docker image, users can run your app without local Python setup.

1. **Pull the Docker image**
   - Example image name:
     ```bash
     docker pull zeeshan5932/medicine-recognition-system:latest
     ```
   - Replace image name/tag if your DockerHub repo is different.

2. **Run the Docker container**
   - With API key passed directly:
     ```bash
     docker run --name medicine-recognition -p 5000:5000 -e GOOGLE_API_KEY=your_api_key_here zeeshan5932/medicine-recognition-system:latest
     ```
   - Or with a local `.env` file:
     ```bash
     docker run --name medicine-recognition -p 5000:5000 --env-file .env zeeshan5932/medicine-recognition-system:latest
     ```

3. **Open in browser**
   - Go to `http://127.0.0.1:5000`.

   
## API Key Setup

To use this project, you need an API key from Google Gemini Large Language Model. Follow these steps to obtain and set up your API key:

1. **Get API Key:**
   - Visit Alkali App [Click Here](https://makersuite.google.com/app/apikey).
   - Follow the instructions to create an account and obtain your API key.

2. **Set Up API Key:**
   - Create a file named `.env` in the project root.
   - Add your API key to the `.env` file:
     ```dotenv
     GOOGLE_API_KEY=your_api_key_here
     ```

   **Note:** Keep your API key confidential. Do not share it publicly or expose it in your code.<br>


## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

• **Report bugs**: If you encounter any bugs, please let us know. Open up an issue and let us know the problem.

• **Contribute code**: If you are a developer and want to contribute, follow the instructions below to get started!

1. Fork the Project
2. Create your Feature Branch
3. Commit your Changes
4. Push to the Branch
5. Open a Pull Request

• **Suggestions**: If you don't want to code but have some awesome ideas, open up an issue explaining some updates or improvements you would like to see!

#### Don't forget to give the project a star! Thanks again!

## License

This project is licensed under the [Open Source Initiative (OSI)](https://opensource.org/) approved GNU General Public License v3.0 License - see the [LICENSE](LICENSE) file for details.<br>


## Contact Details

Zeeshan Younas - [zeeshanoffical01@gmail.com](zeeshanoffical01@gmail.com)<br>


## Acknowledgements

We'd like to extend our gratitude to all individuals and organizations who have played a role in the development and success of this project. Your support, whether through contributions, inspiration, or encouragement, has been invaluable. Thank you for being a part of our journey.
