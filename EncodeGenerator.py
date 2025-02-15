import os
import cv2
import numpy as np
import pickle
import firebase_admin
from firebase_admin import credentials, storage
import face_recognition

# 🔥 1. ตั้งค่า Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': "face-recognition-459a6.appspot.com"
})

bucket = storage.bucket()
local_folder = "Images"  # ให้เก็บใน Images/

# ตรวจสอบว่าโฟลเดอร์ local มีหรือยัง ถ้ายังไม่มีให้สร้าง
if not os.path.exists(local_folder):
    os.makedirs(local_folder)

# 🔥 2. ดาวน์โหลดรูปภาพจาก Firebase Storage
blobs = bucket.list_blobs(prefix="trainface/")  # เปลี่ยนโฟลเดอร์เป็น trainface/
studentImages = {}

for blob in blobs:
    filePath = blob.name  # ตัวอย่าง: "trainface/1001/img1.jpg"
    parts = filePath.split("/")
    
    if len(parts) < 3:  # ข้ามไฟล์ที่ไม่ใช่ภาพ
        continue

    studentId = parts[1]  # ดึง ID จากโฟลเดอร์
    fileName = parts[-1]  # ดึงชื่อไฟล์จาก path
    
    # ตรวจสอบว่าเป็นไฟล์รูปภาพจริงหรือไม่ (ไม่ใช่โฟลเดอร์เปล่า)
    if not fileName.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue  # ข้ามไฟล์ที่ไม่ใช่รูปภาพ

    # สร้างโฟลเดอร์ local สำหรับแต่ละคน
    studentFolderPath = os.path.join(local_folder, studentId)
    if not os.path.exists(studentFolderPath):
        os.makedirs(studentFolderPath)
    
    # ดาวน์โหลดไฟล์ลง local
    local_file_path = os.path.join(studentFolderPath, fileName)
    blob.download_to_filename(local_file_path)  # ✅ ตรวจสอบว่าบันทึกเป็นไฟล์

    # โหลดรูปภาพ
    img = cv2.imread(local_file_path)
    
    if studentId not in studentImages:
        studentImages[studentId] = []
    
    studentImages[studentId].append(img)

print("✅ ดาวน์โหลดภาพจาก trainface/ เรียบร้อย!")

# 🔥 3. ฟังก์ชันสำหรับสร้าง Encoding โดยใช้หลายภาพต่อคน
def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.resize(img, (500, 500))  # ลดขนาดภาพเพื่อให้ Raspberry Pi ทำงานไหว
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img, model='large')  # ใช้โมเดลใหญ่

        if encodings:
            encodeList.append(encodings[0])  # บันทึก encoding ถ้าหาเจอ

    if encodeList:
        return np.mean(encodeList, axis=0)  # คำนวณค่าเฉลี่ย
    else:
        return None  # ถ้าไม่มีใบหน้าเลย

# 🔥 4. คำนวณ Encoding และบันทึกเป็นไฟล์
encodeListKnownWithIds = {}

for studentId, images in studentImages.items():
    encoding = findEncodings(images)
    if encoding is not None:
        encodeListKnownWithIds[studentId] = encoding

# 🔥 5. บันทึกลงไฟล์ pickle
with open("EncodeFile.p", 'wb') as file:
    pickle.dump(encodeListKnownWithIds, file)

print("✅ Encoding เสร็จสิ้น, บันทึกไฟล์เรียบร้อย!")
