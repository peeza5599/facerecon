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
import subprocess
import dlib
import threading

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

# ✅ รัน EncodeGenerator.py
def run_encode_generator():
    print("🔄 Running EncodeGenerator.py ...")
    subprocess.run(["python3", "EncodeGenerator.py"])  # เรียกใช้ EncodeGenerator.py
    print("✅ EncodeGenerator.py completed!")

# 🔽 เรียกใช้ฟังก์ชัน
run_encode_generator()

# Load encodings
print("Loading Encode File ...")
with open('EncodeFile.p', 'rb') as file:
    encodeListKnownWithIds = pickle.load(file)

# ✅ ดึงข้อมูล student ID และ encoding ออกมาใหม่ให้ตรงกับ dict
studentIds = list(encodeListKnownWithIds.keys())  # ได้ student IDs
encodeListKnown = list(encodeListKnownWithIds.values())  # ได้ encoding ของแต่ละคน

print("✅ Encode File Loaded!")

def run_download_profile():
    print("🔄 download_profiles.py ...")
    subprocess.run(["python3", "download_profiles.py"])
    print("✅ ownload_profiles.py completed!")

run_download_profile()


# โหลดโมเดลตรวจจับใบหน้าและ Landmark
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# กำหนดค่า EAR Threshold และ Blink Requirement
EAR_THRESHOLD = 0.25  # ค่าที่ต่ำกว่านี้ถือว่ากระพริบตา
BLINK_REQUIRED = 2
EAR_FRAMES = 8 
OPEN_EYE_FRAMES_REQUIRED = 10 
MIN_BLINK_DURATION = 3

LEFT_EYE_IDX = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_IDX = [42, 43, 44, 45, 46, 47]

# ฟังก์ชันคำนวณค่า EAR (Eye Aspect Ratio)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# Variables
modeType = 0
counter = 0
imgStudent = []
last_log_times = {}
log_interval = timedelta(seconds=30)
last_unknown_alert_time = {}
unknown_alert_interval = timedelta(seconds=20)
unknown_face_counter = {} 
required_unknown_time = 5
threshold = 0.45
scan_frames_required = 3
scanned_faces = {}
blink_counter = 0
checkok = False
checkface = True
total_blinks = 0
blink_detected = False
open_eye_frames = 0
ear_history = []
ready_to_detect_blink = False
blink_duration = 0 
min_ear = 1.0
last_id = None




# LINE Notify
LINE_NOTIFY_TOKEN = "cknZg26SLz2AhsgQKOMxzKVfOu5H0xlCPDeCXjIoc7Z"

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
            log_data['name'],
            log_data['total_attendance'],
            log_data['last_attendance_time'],
            log_data['dominant_emotion'],
            Json(log_data['emotion_scores']),
            log_data['timestamp']
        ))
        conn.commit()
        print("✅ Log data saved to PostgreSQL!")
    except Exception as e:
        print(f"⚠️ Error saving log to PostgreSQL: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()

# ✅ ฟังก์ชัน Asynchronous เพื่อบันทึกข้อมูลลง PostgreSQL โดยใช้ Threading
def save_log_to_postgresql_async(log_data):
    threading.Thread(target=save_log_to_postgresql, args=(log_data,), daemon=True).start()

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



def fetch_student_image(student_id):
    local_path = os.path.join("profile", f"{student_id}.png")
    if os.path.exists(local_path):
        return cv2.imread(local_path)
    else:
        print(f"⚠️ ไม่พบภาพของ {student_id} ใน profile/")
        return None 
    
ROI_X1, ROI_Y1 = 200, 200  # มุมบนซ้าย
ROI_X2, ROI_Y2 = 490, 530  # มุมล่างขวา

conn = psycopg2.connect(conn_string)
cursor = conn.cursor()

def get_user_data(id):
    cursor.execute("SELECT id, name, role, total_attendance, last_attendance_time, standing, studyClass FROM users WHERE id = %s", (id,))
    user = cursor.fetchone()
    if user:
        return {
            "id": user[0],
            "name": user[1],
            "role": user[2],
            "total_attendance": user[3],
            "last_attendance_time": user[4],
            "standing": user[5],
            "studyClass": user[6]
        }
    return None


while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
    faces = detector(imgS)
    faceCurFrame = face_recognition.face_locations(imgS)
    gray = cv2.cvtColor(imgS, cv2.COLOR_BGR2GRAY)
    cv2.rectangle(imgBackground, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 0), 2)

    if not faces:
        total_blinks = 0
        open_eye_frames = 0
        ready_to_detect_blink = False
        blink_detected = False
        min_ear = 1.0
        studentInfo = {}
        imgStudent = []
        ear_history.clear()
        counter = 0
        modeType = 0
        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]



    for face in faces:


        landmarks = predictor(gray, face)
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE_IDX])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE_IDX])

        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        avg_EAR = (left_EAR + right_EAR) / 2.0

        ear_history.append(avg_EAR)
        if len(ear_history) > EAR_FRAMES:
            ear_history.pop(0)

        avg_ear_history = np.mean(ear_history)
        min_ear = min(min_ear, avg_EAR)

        if avg_ear_history > EAR_THRESHOLD:
            open_eye_frames += 1
        else:
            open_eye_frames = 0  # ถ้าตาหลับอยู่ให้รีเซ็ตตัวนับ

        if open_eye_frames >= OPEN_EYE_FRAMES_REQUIRED:
            ready_to_detect_blink = True


        if ready_to_detect_blink:
            if avg_EAR < EAR_THRESHOLD:
                blink_duration += 1  # นับจำนวนเฟรมที่ตาหลับ
                blink_detected = True
            else:
                # ✅ เช็คว่าเป็นกระพริบตาจริงหรือแค่หรี่ตา
                if blink_detected and blink_duration >= MIN_BLINK_DURATION:
                    total_blinks += 1  
                    blink_detected = False
                    blink_duration = 0
                else:
                    blink_detected = False
                    blink_duration = 0

            if total_blinks >= BLINK_REQUIRED:
                total_blinks = 0
                checkok = True
                ready_to_detect_blink = False  
                open_eye_frames = 0  
            cv2.putText(imgBackground, f"Total Blinks: {total_blinks}", (200, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if checkok is True:
        ready_to_detect_blink = False
        checkface = False
        faceCurFrame = face_recognition.face_locations(imgS)

        if len(faceCurFrame) > 1:
            modeType = 5
            imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType] 
            checkok = False
            continue

        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
        if faceCurFrame:
            for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                sorted_indices = np.argsort(faceDis)
                best_match = sorted_indices[0]
                second_best = sorted_indices[1] if len(sorted_indices) > 1 else None
                matchIndex = np.argmin(faceDis)

                student_id = studentIds[best_match]
                if student_id not in scanned_faces:
                    scanned_faces[student_id] = []
                scanned_faces[student_id].append(faceDis[best_match])

                if len(scanned_faces[student_id]) < scan_frames_required:
                    continue  

                avg_face_distance = np.mean(scanned_faces[student_id])

                print(f"🔎 Face Distance (Avg {scan_frames_required} frames): {avg_face_distance:.2f}")

                if avg_face_distance < threshold and (second_best is None or abs(avg_face_distance - faceDis[second_best]) > 0.06):
                    id = student_id
                    print(f"✔️ ตรวจพบ {id} ด้วยค่าความคล้ายคลึง {avg_face_distance:.2f}")


                    if counter == 0:
                        modeType = 1
                        counter = 1
                        studentInfo = {}
                        imgStudent = []
                else:
                    id = -1  
                    print(f"❌ ไม่พบในระบบ (Avg Face Distance = {avg_face_distance:.2f})")
                    counter = 0

                scanned_faces.clear()  

                if  avg_face_distance < threshold:
                    id = studentIds[matchIndex]
                    if counter == 0:
                        cvzone.putTextRect(imgBackground, "Loading", (275, 400))
                        cv2.imshow("Face Attendance", imgBackground)
                        cv2.waitKey(1)
                        counter = 1
                        modeType = 1
                else:
                    modeType = 1  
                    counter = 0
                    id = -1  

            if id == -1:  # แจ้งเตือนเมื่อไม่พบผู้ในระบบ
                current_time = datetime.now()

                                # เริ่มนับเวลาเมื่อเจอคนไม่รู้จัก
                if id not in unknown_face_counter:
                        unknown_face_counter[id] = current_time

                elapsed_time = (current_time - unknown_face_counter[id]).total_seconds()

                if elapsed_time >= required_unknown_time:
                    if id not in last_unknown_alert_time or current_time - last_unknown_alert_time[id] > unknown_alert_interval:
                        message = "พบบุคคลที่ไม่รู้จักในระบบ! กรุณาตรวจสอบ."
                        send_line_notify(message, LINE_NOTIFY_TOKEN)
                        print("Unknown face detected and notification sent.")
                        last_unknown_alert_time[id] = current_time
                        unknown_face_counter[id] = current_time
                        modeType = 4
                        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
                        cv2.waitKey(1)


            if counter != 0:
                if counter == 1:
                    if id != -1 and id != last_id:
                            studentInfo = {}
                            imgStudent = []
                            last_id = id
                    studentInfo = get_user_data(id)  # ใช้ฟังก์ชันที่สร้างไว้
                    print(studentInfo)

                    imgStudent = fetch_student_image(id)

                    emotions = emotion_detector.detect_emotions(img)
                    if emotions:
                        emotion_scores = emotions[0]["emotions"]
                        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                        current_time = datetime.now()
                        if id not in last_log_times or current_time - last_log_times[id] > log_interval:
                            log_data = {
                                'id': id,
                                'name': studentInfo['name'],
                                'Room_Number': studentInfo['name'],
                                'total_attendance': int(studentInfo['total_attendance']) + 1,
                                'last_attendance_time': studentInfo['last_attendance_time'],
                                'dominant_emotion': dominant_emotion,
                                'emotion_scores': emotion_scores,
                                'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S")
                            }
                            save_log_to_postgresql_async(log_data)
                            last_log_times[id] = current_time

                    if isinstance(studentInfo['last_attendance_time'], datetime):
                        last_attendance_time = studentInfo['last_attendance_time']
                    else:
                        last_attendance_time = datetime.strptime(studentInfo['last_attendance_time'], "%Y-%m-%d %H:%M:%S")

                    secondsElapsed = (datetime.now() - last_attendance_time).total_seconds()
                    print(secondsElapsed)

                    if secondsElapsed > 30:
                        studentInfo['total_attendance'] += 1
                        try:
                            cursor.execute(
                                """
                                UPDATE users 
                                SET total_attendance = %s, last_attendance_time = %s 
                                WHERE id = %s
                                """,
                                (studentInfo['total_attendance'], datetime.now(), studentInfo['id'])
                            )
                            conn.commit()
                            print(f"✅ อัปเดตข้อมูลของ {studentInfo['id']} สำเร็จ!")
                        except Exception as e:
                            conn.rollback()
                            print(f"⚠️ อัปเดตข้อมูลของ {studentInfo['id']} ไม่สำเร็จ: {e}")
                    else:
                        modeType = 3
                        counter = 0
                        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                if modeType != 3:
                    if 10 < counter < 20:
                        modeType = 2

                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                    if counter <= 10 :
                        cv2.putText(imgBackground, str(studentInfo.get('total_attendance', '')), (861, 125),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                        cv2.putText(imgBackground, str(studentInfo.get('role', '')), (1006, 550),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(imgBackground, str(id), (1006, 493),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(imgBackground, str(studentInfo.get('standing', '')), (910, 625),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                        cv2.putText(imgBackground, str(studentInfo.get('studyClass', '')), (1125, 625),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

                        name = studentInfo.get('name', '')
                        (w, h), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                        offset = (414 - w) // 2
                        cv2.putText(imgBackground, name, (808 + offset, 445),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

                        if imgStudent is not None and isinstance(imgStudent, np.ndarray) and imgStudent.size != 0:
                            imgBackground[175:175 + 216, 909:909 + 216] = imgStudent
                        else:
                            print("⚠️ ไม่มีภาพนักเรียนให้แสดง (imgStudent ว่าง)")

                    counter += 1

                    if counter >= 20:
                        counter = 0
                        modeType = 0
                        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

        else:
            checkface = True
            checkok = False
            unknown_face_counter.clear()
            

    cv2.imshow("Face Attendance", imgBackground)
    cv2.waitKey(1)
