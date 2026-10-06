# Voice Input Validation Test Cases

## 1. Microphone permission allowed
- **Action:** Open application, start assessment, click the microphone button. Grant permission in the browser prompt.
- **Expected:** Browser allows access, recording starts without error.
- **Status:** PENDING (Requires manual user test)

## 2. Speak a clear Java answer
- **Action:** Record: "Inheritance allows a child class to acquire properties and methods from a parent class, which improves code reuse."
- **Expected:** Transcript appears accurately in the text area. `response_input_mode` becomes "voice".
- **Status:** PENDING (Requires manual user test)

## 3. Short answer
- **Action:** Record: "I don't know."
- **Expected:** Transcript appears in the text area.
- **Status:** PENDING (Requires manual user test)

## 4. Long answer
- **Action:** Record a 30-second detailed explanation of polymorphism.
- **Expected:** Full transcript appears accurately.
- **Status:** PENDING (Requires manual user test)

## 5. Pause during speech
- **Action:** Record: "Polymorphism is... [pause for 3 seconds] ...when a method does different things based on the object."
- **Expected:** Transcript captures the entire sentence without cutting off prematurely.
- **Status:** PENDING (Requires manual user test)

## 6. Retry after failed recognition
- **Action:** Record silence or mumble indistinctly.
- **Expected:** Error message "No speech detected. Please try speaking again." appears. Text box is not overwritten. Click microphone again and speak clearly. The second attempt succeeds.
- **Status:** PENDING (Requires manual user test)

## 7. Edit transcript before submitting
- **Action:** Record a response. Then manually type and correct a word in the text area before clicking submit.
- **Expected:** The corrected text is submitted.
- **Status:** PENDING (Requires manual user test)

## 8. Follow-up question voice response
- **Action:** On the follow-up question, use the microphone to record a response.
- **Expected:** Transcript appears in the follow-up text area.
- **Status:** PENDING (Requires manual user test)

## 9. Browser unsupported
- **Action:** Attempt to run in an unsupported browser (if applicable).
- **Expected:** Streamlit displays an appropriate warning about `st.audio_input` not being supported.
- **Status:** PENDING (Requires manual user test)

## 10. Microphone permission denied
- **Action:** Block microphone access in browser settings and click the microphone.
- **Expected:** Browser denies access, user sees a message to allow access, or the recording fails gracefully.
- **Status:** PENDING (Requires manual user test)

## 11. Text-only response after voice failure
- **Action:** Intentionally fail a voice recording. Then type a response and click submit.
- **Expected:** Assessment proceeds normally using the typed text.
- **Status:** PENDING (Requires manual user test)
