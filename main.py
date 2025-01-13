import os
import pickle
import numpy as np
import cv2
import face_recognition
import cvzone
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
import requests
from fer import FER
from datetime import datetime, timedelta
import json

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://face-recognition-459a6-default-rtdb.asia-southeast1.firebasedatabase.app/",
    'storageBucket': "face-recognition-459a6.appspot.com"
})

bucket = storage.bucket()

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread('Resources/background.png')

# Importing the mode images into a list
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

# Load the encoding file
print("Loading Encode File ...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
print("Encode File Loaded")

# Initialize variables
modeType = 0
counter = 0
imgStudent = []

# Initialize FER detector
emotion_detector = FER(mtcnn=True)

# ตัวแปรสำหรับเก็บเวลาการบันทึก log ล่าสุด
last_log_times = {}
log_interval = timedelta(seconds=30)

# ตัวแปรสำหรับเก็บเวลาการแจ้งเตือนคนที่ไม่รู้จัก
last_unknown_alert_time = {}
unknown_alert_interval = timedelta(seconds=20)

# Function to log data to file
def log_to_file(data):
    log_file = 'log.json'

    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []

    logs.append(data)

    with open(log_file, 'w') as f:
        try:
            json.dump(logs, f, indent=2)
        except Exception as e:
            print(f"Error writing to log file: {e}")

# Function to send LINE notifications
def send_line_notify(message, token):
    url = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {token}"}
    data = {"message": message}
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        print("Notification sent successfully")
    else:
        print("Failed to send notification")

LINE_NOTIFY_TOKEN = "cknZg26SLz2AhsgQKOMxzKVfOu5H0xlCPDeCXjIoc7Z"

while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    if faceCurFrame:
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
                id = studentIds[matchIndex]
                if counter == 0:
                    cvzone.putTextRect(imgBackground, "Loading", (275, 400))
                    cv2.imshow("Face Attendance", imgBackground)
                    cv2.waitKey(1)
                    counter = 1
                    modeType = 1
            else:
                modeType = 1  # active
                counter = 0
                id = -1  # Unknown person

        if id == -1:  # แจ้งเตือนเมื่อไม่พบผู้ในระบบ
            current_time = datetime.now()

            # ตรวจสอบว่าเวลาที่ผ่านมาเกิน 20 วินาทีหรือยัง
            if id not in last_unknown_alert_time or current_time - last_unknown_alert_time[id] > unknown_alert_interval:
                message = "พบบุคคลที่ไม่รู้จักในระบบ! กรุณาตรวจสอบ."
                send_line_notify(message, LINE_NOTIFY_TOKEN)
                print("Unknown face detected and notification sent.")

                # อัพเดทเวลาแจ้งเตือนล่าสุด
                last_unknown_alert_time[id] = current_time

        if counter != 0:
            if counter == 1:
                # Get the Data
                studentInfo = db.reference(f'room/{id}').get()
                print(studentInfo)

                # Get the Image from the storage
                blob = bucket.get_blob(f'Images/{id}.png')
                array = np.frombuffer(blob.download_as_string(), np.uint8)
                imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)

                emotions = emotion_detector.detect_emotions(img)
                if emotions:
                    emotion_scores = emotions[0]["emotions"]
                    dominant_emotion = max(emotion_scores, key=emotion_scores.get)

                    # ตรวจสอบว่าเวลาในการบันทึก log ล่าสุดผ่านไปแล้วหรือไม่
                    current_time = datetime.now()
                    if id not in last_log_times or current_time - last_log_times[id] > log_interval:
                        log_data = {
                            'id': id,
                            'name': studentInfo['name'],
                            'Room_Number': studentInfo['Room_Number'],
                            'total_attendance': studentInfo['total_attendance'],
                            'last_attendance_time': studentInfo['last_attendance_time'],
                            'dominant_emotion': dominant_emotion,
                            'emotion_scores': emotion_scores,
                            'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        # บันทึก log
                        log_to_file(log_data)
                        
                        # อัพเดทเวลาในการบันทึก log ล่าสุด
                        last_log_times[id] = current_time

                # Update data of attendance
                datetimeObject = datetime.strptime(studentInfo['last_attendance_time'], "%Y-%m-%d %H:%M:%S")
                secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
                print(secondsElapsed)
                if secondsElapsed > 30:
                    ref = db.reference(f'room/{id}')
                    studentInfo['total_attendance'] += 1
                    ref.child('total_attendance').set(studentInfo['total_attendance'])
                    ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                else:
                    modeType = 3
                    counter = 0
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

            if modeType != 3:
                if 10 < counter < 20:
                    modeType = 2

                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                if counter <= 10:
                    cv2.putText(imgBackground, str(studentInfo['total_attendance']), (861, 125),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(studentInfo['Room_Number']), (1006, 550),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(id), (1006, 493),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(studentInfo['standing']), (910, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    cv2.putText(imgBackground, str(studentInfo['starting_year']), (1125, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

                    (w, h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                    offset = (414 - w) // 2
                    cv2.putText(imgBackground, str(studentInfo['name']), (808 + offset, 445),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

                    imgBackground[175:175 + 216, 909:909 + 216] = imgStudent

                counter += 1

                if counter >= 20:
                    counter = 0
                    modeType = 0
                    studentInfo = []
                    imgStudent = []
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    cv2.imshow("Face Attendance", imgBackground)
    cv2.waitKey(1)
