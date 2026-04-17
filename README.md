# **Glamoure: AI-Powered Audio Summarizer 🎙️📄**

**Glamoure** is a high-performance web application built with the **MERN stack** (MongoDB, Express, React, Node.js) designed to transform spoken information into actionable insights. It specializes in the "decryption" of audio files, providing automated transcriptions and concise AI-generated summaries.

## **🚀 The Vision**

In a fast-paced digital world, consuming long audio recordings is time-consuming. **Glamoure** solves this by providing a centralized platform where users can upload pre-recorded audio notes, lectures, or meetings, and receive a structured summary in seconds.

## **✨ Core Features**

* **Audio Transcription (Speech-to-Text):** Advanced processing of uploaded audio files into accurate plain text.  
* **Intelligent Summarization:** Leverages Large Language Models (LLM) to extract key points, action items, and main topics.  
* **Dynamic Chat History:** A robust dashboard featuring:  
  * **Search & Filters:** Quickly find past summaries by keywords, dates, or metadata.  
  * **Expandable Interface:** UI components that collapse or expand for a cleaner workspace.  
* **Markdown-to-HTML Rendering:** Summaries are generated using Markdown syntax for rich formatting (bold, lists, links) and rendered seamlessly in the browser.  
* **Media Management:** Built-in audio player within the history view to cross-reference summaries with the original source.  
* **Scalable Backend:** Optimized for handling multi-format audio uploads and asynchronous AI processing.

## **🛠️ Tech Stack**

### **Frontend**

* **React.js:** Component-based architecture for a fluid User Experience.  
* **Markdown Processing:** Specialized libraries (e.g., react-markdown) for high-quality text rendering.  
* **State Management:** Context API for global user and session persistence.

### **Backend**

* **Node.js & Express:** Scalable API handling and audio file streaming.  
* **MongoDB & Mongoose:** Flexible data modeling for transcripts, summaries, and user metadata.  
* **AI Integration:** Integration with advanced AI APIs (e.g., OpenAI Whisper/GPT) for transcription and NLP tasks.

## **📦 Getting Started**

### **Prerequisites**

* Node.js (v14+)  
* MongoDB Atlas account or local instance.  
* API Key for AI transcription/summarization services.

### **Installation**

1. **Clone the Repository:**  
   git clone \[https://github.com/lautiapetr/glamoure.git\](https://github.com/lautiapetr/glamoure.git)  
   cd glamoure

2. **Server Setup:**  
   cd server  
   npm install  
   \# Create a .env file with MONGO\_URI, AI\_API\_KEY, and PORT  
   npm start

3. **Client Setup:**  
   cd ../client  
   npm install  
   npm start

## **📖 Application Workflow**

1. **Upload:** Use the interface to select a pre-recorded audio file.  
2. **Transcription:** The backend processes the audio stream and converts it into text.  
3. **Analysis:** The AI analyzes the transcript to generate a bulleted summary.  
4. **Review:** Access the result in your History, where you can filter, search, and replay the original audio.

## **🤝 Contributing**

Contributions make the open-source community an amazing place to learn and create.

1. Fork the Project.  
2. Create your Feature Branch (git checkout \-b feature/AmazingFeature).  
3. Commit your Changes (git commit \-m 'Add some AmazingFeature').  
4. Push to the Branch (git push origin feature/AmazingFeature).  
5. Open a Pull Request.

(*Little clarifications: The project was made in a LocalHost enviroment and not tested in online servers*).

**Developed by Lautaro Petroni**