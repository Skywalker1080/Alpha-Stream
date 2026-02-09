FROM python:3.13-slim

WORKDIR /frontend

RUN pip install uv

COPY . /frontend

RUN uv pip install --system -r requirements.txt

EXPOSE 8501

CMD [ "streamlit", "run", "app.py", "--server.address=0.0.0.0" ]