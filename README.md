# VR-Image-3D-Object-Generator
Dependencies Installation and Start Backend:
```bash
cd ./backend

python3 -m venv venv
source ./venv/bin/activate

# installing dependencies
pip install -r ./requirements.txt

# starting backend server
uvicorn outer_layer.app:app --reload

# required environment variables: HF_TOKEN, MY_API_KEY
```

Dependencies Installation and Start Frontend:
```bash
cd ./frontend

# installing dependencies
npm install

# generating openssl keys for https
openssl req -nodes -new -x509 -keyout key.pem -out cert.pem -days 365

# starting frontend server
node server.js
```
