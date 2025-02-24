// static/js/hooks/useSpeechRecognition.jsx
import { useState, useEffect, useRef } from "react";

export function useSpeechRecognition() {
  const [transcript, setTranscript] = useState("");
  const [listening, setListening] = useState(false);
  const [isSupported, setIsSupported] = useState(true);
  const recognitionRef = useRef(null);

  useEffect(() => {
    // Check if the browser supports the Web Speech API
    if (!("webkitSpeechRecognition" in window || "SpeechRecognition" in window)) {
      console.error("Speech Recognition API is not supported in this browser.");
      setIsSupported(false);
      return;
    }

    // Create a new instance of the SpeechRecognition API
    const RecognitionAPI = window.webkitSpeechRecognition || window.SpeechRecognition;
    const recognition = new RecognitionAPI();
    recognition.continuous = true;
    recognition.interimResults = true;

    // Process speech results
    recognition.onresult = (event) => {
      let finalTranscript = "";
      // Iterate over the results starting at the index provided by the event
      for (let i = event.resultIndex; i < event.results.length; i++) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        }
      }
      setTranscript(finalTranscript);
    };

    // Handle errors and ending of recognition
    recognition.onerror = (event) => {
      console.error("Speech recognition error:", event.error);
      setListening(false);
    };

    recognition.onend = () => {
      setListening(false);
    };

    recognitionRef.current = recognition;
  }, []);

  const startListening = () => {
    if (!isSupported || listening) return;
    setListening(true);
    recognitionRef.current.start();
  };

  const stopListening = () => {
    if (!isSupported || !listening) return;
    recognitionRef.current.stop();
    setListening(false);
  };

  return { transcript, listening, startListening, stopListening, isSupported };
}
