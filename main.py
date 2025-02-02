import os
import pickle
import numpy as np
import cv2
import face_recognition
import cvzone
import firebase_admin
from firebase_admin import credentials, db, storage
import requests
from fer import FER
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import Json

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://face-recognition-459a6-default-rtdb.asia-southeast1.firebasedatabase.app/",
    'storageBucket': "face-recognition-459a6.appspot.com"
})
bucket = storage.bucket()

# Initialize PostgreSQL Connection
conn_string = "postgresql://facerecon_owner:NsqA5QSpbT2G@ep-super-bonus-a1hmwxyx.ap-southeast-1.aws.neon.tech/facerecon?sslmode=require"

# Initialize FER detector
emotion_detector = FER(mtcnn=True)

# Video capture setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load background and modes
imgBackground = cv2.imread('Resources/background.png')
folderModePath = 'Resources/Modes'
imgModeList = [cv2.imread(os.path.join(folderModePath, path)) for path in os.listdir(folderModePath)]

# Load encodings
print("Loading Encode File ...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
print("Encode File Loaded")

# Variables
modeType = 0
counter = 0
imgStudent = []
last_log_times = {}
log_interval = timedelta(seconds=30)
last_unknown_alert_time = {}
unknown_alert_interval = timedelta(seconds=20)
unknown_face_counter = {} 
required_unknown_time = 10  

# LINE Notify
LINE_NOTIFY_TOKEN = "cknZg26SLz2AhsgQKOMxzKVfOu5H0xlCPDeCXjIoc7Z"

# PostgreSQL: Save log_data to database
def save_log_to_postgresql(log_data):
    try:
        conn = psycopg2.connect(conn_string)
        cursor = conn.cursor()
        insert_query = """
        INSERT INTO logs (
            user_id, name, room_number, total_attendance,
            last_attendance_time, dominant_emotion, emotion_scores, timestamp
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (
            log_data['id'],
            log_data['name'],
            log_data['Room_Number'],
            log_data['total_attendance'],
            log_data['last_attendance_time'],
            log_data['dominant_emotion'],
            Json(log_data['emotion_scores']),
            log_data['timestamp']
        ))
        conn.commit()
        print("Log data saved to PostgreSQL!")
    except Exception as e:
        print(f"Error saving log to PostgreSQL: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()

# Send LINE notification
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

                            # เริ่มนับเวลาเมื่อเจอคนไม่รู้จัก
            if id not in unknown_face_counter:
                    unknown_face_counter[id] = current_time

            elapsed_time = (current_time - unknown_face_counter[id]).total_seconds()

                # ตรวจสอบว่าผ่านไปแล้ว 10 วินาทีหรือไม่
            if elapsed_time >= required_unknown_time:
                if id not in last_unknown_alert_time or current_time - last_unknown_alert_time[id] > unknown_alert_interval:
                    message = "พบบุคคลที่ไม่รู้จักในระบบ! กรุณาตรวจสอบ."
                    send_line_notify(message, LINE_NOTIFY_TOKEN)
                    print("Unknown face detected and notification sent.")

                        # อัพเดทเวลาแจ้งเตือนล่าสุด
                    last_unknown_alert_time[id] = current_time
                        
                        # รีเซ็ตตัวนับหลังจากแจ้งเตือน
                    unknown_face_counter[id] = current_time

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

                if emotions:
                    emotion_scores = emotions[0]["emotions"]
                    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
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
                        save_log_to_postgresql(log_data)
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

    else:
        # ถ้าไม่เจอใบหน้าใด ๆ ให้รีเซ็ตตัวนับทั้งหมด
        unknown_face_counter.clear()

    cv2.imshow("Face Attendance", imgBackground)
    cv2.waitKey(1)
