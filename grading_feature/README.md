# Grading Dashboard
A FastAPI + React dashboard that extracts, processes, and visualizes student chatbot interactions for grading and analytics.




## Overview 

This dashboard extracts student chat sessions from an OpenWebUI SQLite database and converts them into a structured JSON format.
A FastAPI backend serves analytics and user-level details, while a React frontend visualizes metrics, charts, and chat transcripts.
Instructors can quickly review student interactions, identify patterns, and evaluate performance at scale.


## Features

**Backend (FastAPI)**

* `/users` → returns all students with aggregated chat data

* `/user/{id}` → returns full chat history for a specific student

* Organized directory structure 
  
**Frontend (React + Vite)**

* Metric cards (Total students, chats, messages)
* Top-Users chart
* Chats-per-day visualization
* Search for users
* Pagination for users + chats
* Sliding drawer for transcripts


## How to Run

Run command: `./run_app.sh`


**Requirements**
* Python 3.10+ (Use the [Python Installer](https://www.python.org/downloads/release/python-3110/) for windows, otherwise please lookup how to install through Linux/MacOS)

* Node.js v20.19.5
* npm 10.8.2


If these are not installed, see the installation notes in the [Frontend Readme](./frontend/README.md) folder.


## Images of the dashboard

### Dashboard Home
<img src="./screenshots/home.png" width="600" />

### Top Users Chart
<img src="./screenshots/users_chart.png" width="600" />

### User Detail Page
<img src="./screenshots/user_detail.png" width="600" />

### Chat Transcript Drawer
<img src="./screenshots/chat_drawer.png" width="600" />




## Notes

This project is part of the ongoing work to support instructors in evaluating student chatbot interactions.  
Future improvements will include enhanced filtering, improved analytics, and optional database storage.


## Usage

This project is intended for academic use.  