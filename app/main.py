# app/main.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import tempfile
from jinja2 import Template
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content
# from send_email import send_email_with_attachment, create_email_body  # Import the email functions
from app.send_email import send_email_with_attachment, create_email_body
from app.react_agent.pretty import generate_docx
from dotenv import load_dotenv
import os
import asyncio
import uuid  # Added for unique filenames
from pathlib import Path  # Added for path handling
from app.react_agent.graph import PocketTraveller
import json # Added for JSON handling


# Load environment variables from the .env file
load_dotenv()

# Create data directory if not exists
Path("data").mkdir(exist_ok=True)

# Retrieve the API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, or specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


# Mount the static folder so CSS, JS, images, etc. are accessible at /static
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Set up Jinja2 for HTML templating
templates = Jinja2Templates(directory="app/templates")

# Define a Pydantic model for form data validation
class TripFormData(BaseModel):
    origin: str
    destination: str
    dates: list[str]
    adults: int
    children: int
    email: str
    voiceNotes: str


    
        
# Endpoint to handle form submission and send data to the LLM
@app.post("/submit-trip")
async def submit_trip(request: Request, data: TripFormData):

    try:       
        # Create a natural sentence introduction
        user_input = (
            f"I am looking to travel from {data.origin} to {data.destination}. "
            f"I plan to travel on the following dates: {', '.join(data.dates)}. "
            f"There will be {data.adults} adult(s) and {data.children} child(ren) traveling with me. "
            f"My email address is {data.email}, and I have left the following extra information: {data.voiceNotes}."
        )
        
        planner = PocketTraveller()
        output = await planner.invoke_graph(user_input)
        
        # Generate unique filename using UUID
        unique_id = uuid.uuid4().hex
        # document_filename = f"travel_plan_{unique_id}"
        file_path = f"app/static/results/{unique_id}.pdf"
        
        # Generate the DOCX in a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            temp_docx_path = tmp.name
        
        generate_docx(output, temp_docx_path, file_path)

        email_body = create_email_body(
            data.origin, 
            data.destination, 
            data.dates, 
            data.adults, 
            data.children
        )
        
        send_email_with_attachment(
            to_email=data.email, 
            subject="Your Pocket Travel Plan", 
            body=email_body, 
            file_path=temp_docx_path
        )
        
        # Remove the temporary DOCX file so that only the PDF remains
        os.remove(temp_docx_path)
        
        return JSONResponse(content={"message": "Trip details received. An email will be sent shortly."})
    
    except Exception as e:
        print(f"Error in submit_trip: {e}")
        return JSONResponse(
            status_code=500, 
            content={"error": str(e)}
        )


# Homepage endpoint
@app.get("/")
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Trip planner page endpoint
@app.get("/plan-a-trip")
async def read_create_trip(request: Request):
    return templates.TemplateResponse("create_trip.html", {"request": request, "GOOGLE_MAPS_API_KEY": GOOGLE_MAPS_API_KEY})

@app.get("/thank-you")
async def read_create_trip(request: Request):
    return templates.TemplateResponse("thank_you.html", {"request": request})
# Run with: uvicorn app.main:app --reload
