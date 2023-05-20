FROM python:3.10

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

WORKDIR /app/model_objects
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1KU1LeTO5UvWrKSudrXowpz2aLQD9_TIP' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1KU1LeTO5UvWrKSudrXowpz2aLQD9_TIP" -O final_model_random_forest.pkl && rm -rf /tmp/cookies.txt

WORKDIR /app
EXPOSE 8080
CMD ["python", "main.py"]