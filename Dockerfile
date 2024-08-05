FROM python:3.11.9
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8000
ENV PORT=8000
# Run gunicorn server
CMD ["gunicorn", "--workers=4", "--bind", "0.0.0.0:$PORT", "app:app"]