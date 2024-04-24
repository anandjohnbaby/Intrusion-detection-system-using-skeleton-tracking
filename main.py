import streamlit as st
import tensorflow as tf
import cv2
import mediapipe as mp
import pandas as pd
from twilio.rest import Client
import numpy as np
from io import BytesIO
import tempfile

# Load the model
model = tf.keras.models.load_model("intruder_detection_model.h5")

# Twilio credentials
account_sid = 'AC1d8d2a5db506f5b321ae25064b8f22a7'
auth_token = '192866b6046712f0212eb35a77426f07'

twilio_phone_number = '+19382015868'
recipient_phone_number = '+917592972157'

# Initialize Twilio client
client = Client(account_sid, auth_token)

@st.cache_data
def send_sms(message):
    try:
        client.messages.create(to=recipient_phone_number, 
                               from_=twilio_phone_number, 
                               body=message)
        return True  # Indicate successful message sending
    except Exception as e:
        st.write(f"Error sending SMS: {e}")
        return False  # Indicate failure
    


def detect_intruder(keypoints_list, model, feature_names):
    # Convert list of dictionaries to DataFrame
    keypoints_df = pd.DataFrame(keypoints_list)
    
    # Reorder columns to match those used during training
    keypoints_df = keypoints_df[feature_names]
    
    # Predict actions using the model
    predictions = model.predict(keypoints_df)
    
    # Convert predictions to a list
    predictions_list = list(predictions)
    
    # Calculate the proportion of 1s in predictions
    proportion_of_ones = predictions_list.count(1) / len(predictions_list)
    st.write(proportion_of_ones)

    # Check if intruder is detected

    if proportion_of_ones >= 0.08:
        sms_sent = send_sms("Intruder detected!")
        if sms_sent:
            st.write("🚨 Intruder detected!")
            st.write("📩 SMS sent successfully!")
        else:
            st.write("❌ Failed to send SMS.")
    else:
        st.write("✅ No intruder detected")

def extract_keypoints(video_bytes):
    temp_file_path = None  # Initialize variable to store the temporary file path
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(video_bytes)
        temp_file_path = temp_file.name
    
    cap = cv2.VideoCapture(temp_file_path)
    
    if not cap.isOpened():
        st.error("Error: Unable to open video file")
        return None
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils  # Import drawing utilities
    
    keypoints_list = []
    frame_count = 0
    valid_frames_count = 0  # Count of frames with sufficient keypoints
    
    # Create a placeholder for displaying the processed frames
    placeholder = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Convert the image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            landmarks = results.pose_landmarks.landmark
            keypoints = {
                'LKnee_X': landmarks[25].x, 'LKnee_Y': landmarks[25].y,
                'Rknee_X': landmarks[26].x, 'Rknee_Y': landmarks[26].y,
                'RShoulder_X': landmarks[12].x, 'RShoulder_Y': landmarks[12].y,
                'LShoulder_X': landmarks[11].x, 'LShoulder_Y': landmarks[11].y,
                'LElbow_X' : landmarks[13].x, 'LElbow_Y': landmarks[13].y,
                'RElbow_X' : landmarks[14].x, 'RElbow_Y': landmarks[14].y,
                'LAnkle_X': landmarks[27].x, 'LAnkle_Y': landmarks[27].y,
                'RAnkle_X': landmarks[28].x, 'RAnkle_Y': landmarks[28].y,
                'LHip_X': landmarks[23].x, 'LHip_Y': landmarks[23].y,
                'RHip_X': landmarks[24].x, 'RHip_Y': landmarks[24].y,
                'LWrist_X': landmarks[15].x, 'LWrist_Y': landmarks[15].y,
                'RWrist_X': landmarks[16].x, 'RWrist_Y': landmarks[16].y,
            }
            
            if len(keypoints) >= 8:  # Adjusted threshold
                keypoints_list.append(keypoints)
                valid_frames_count += 1
            else:
                st.warning(f"Warning: Not enough keypoints detected in frame {frame_count}. Skipping...")
        
        # Display processed frame within the Streamlit app
        placeholder.image(image_rgb, channels="RGB", use_column_width=True)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

    if valid_frames_count == 0:
        st.error("Error: No frames with sufficient keypoints found in the video.")
        return None
    
    #st.success(f"Processed {frame_count} frames, extracted {valid_frames_count} sets of keypoints.")
    return keypoints_list


# Streamlit UI
st.title("IntrusionGuard: A Deep Learning-Powered Automated Intruder Detection System")
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "webm"])

if uploaded_file is not None:
    # Open the uploaded video file
    video_bytes = uploaded_file.read()
    
    # Display the uploaded video
    #st.video(video_bytes)
    
    # Analyze the video for intruders
    keypoints_list = extract_keypoints(video_bytes)
    if keypoints_list is not None:
        feature_names = ['LShoulder_X', 'LShoulder_Y', 'RShoulder_X', 'RShoulder_Y', 
                     'LElbow_X', 'LElbow_Y', 'RElbow_X', 'RElbow_Y', 
                     'LWrist_X', 'LWrist_Y', 'RWrist_X', 'RWrist_Y',
                     'LHip_X', 'LHip_Y', 'RHip_X', 'RHip_Y', 
                     'LKnee_X', 'LKnee_Y', 'Rknee_X', 'Rknee_Y', 
                     'LAnkle_X', 'LAnkle_Y', 'RAnkle_X', 'RAnkle_Y']
        result = detect_intruder(keypoints_list, model, feature_names)
        #st.write(result)
