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
local_folder = "Images"

# 🔥 สร้างโฟลเดอร์ local ถ้ายังไม่มี
if not os.path.exists(local_folder):
    os.makedirs(local_folder)

# 🔥 2. โหลดรายการไฟล์จาก Firebase
blobs = bucket.list_blobs(prefix="trainface/")
studentImages = {}
firebase_image_files = set()

for blob in blobs:
    filePath = blob.name  # เช่น trainface/1001/img1.jpg
    parts = filePath.split("/")

    if len(parts) < 3:
        continue

    studentId = parts[1]
    fileName = parts[-1]

    if not fileName.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    firebase_image_files.add(f"{studentId}/{fileName}")

# 🔥 3. ลบไฟล์ local ที่ไม่มีใน Firebase
for studentId in os.listdir(local_folder):
    student_folder = os.path.join(local_folder, studentId)
    if not os.path.isdir(student_folder):
        continue

    for local_file in os.listdir(student_folder):
        rel_path = f"{studentId}/{local_file}"
        if rel_path not in firebase_image_files:
            local_file_path = os.path.join(student_folder, local_file)
            os.remove(local_file_path)
            print(f"🗑 ลบไฟล์ {rel_path} ที่ไม่มีใน Firebase")

    if not os.listdir(student_folder):
        os.rmdir(student_folder)
        print(f"🗑 ลบโฟลเดอร์ {student_folder} (ไม่มีรูปเหลือ)")

# 🔥 4. ดาวน์โหลดภาพจาก Firebase
blobs = bucket.list_blobs(prefix="trainface/")  # โหลดอีกรอบ
for blob in blobs:
    filePath = blob.name
    parts = filePath.split("/")

    if len(parts) < 3:
        continue

    studentId = parts[1]
    fileName = parts[-1]

    if not fileName.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    studentFolderPath = os.path.join(local_folder, studentId)
    if not os.path.exists(studentFolderPath):
        os.makedirs(studentFolderPath)

    local_file_path = os.path.join(studentFolderPath, fileName)
    blob.download_to_filename(local_file_path)

    img = cv2.imread(local_file_path)
    if studentId not in studentImages:
        studentImages[studentId] = []
    studentImages[studentId].append(img)

print("\u2705 ดาวน์โหลดภาพจาก trainface/ เรียบร้อย!")

# 🔥 5. ฟังก์ชันหา Encoding

def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.resize(img, (500, 500))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img, model='large')
        if encodings:
            encodeList.append(encodings[0])
    if encodeList:
        return np.mean(encodeList, axis=0)
    else:
        return None

# 🔥 6. โหลด EncodeFile.p ถ้ามี แล้วเปรียบเทียบเฉพาะ user ที่มีรูปอยู่
if os.path.exists("EncodeFile.p"):
    with open("EncodeFile.p", "rb") as file:
        encodeListKnownWithIds = pickle.load(file)
else:
    encodeListKnownWithIds = {}

# 🔥 7. ลบ user จาก EncodeFile.p ที่ไม่มีใน Firebase อีกต่อ
user_ids_in_firebase = set(studentImages.keys())
user_ids_local = set(encodeListKnownWithIds.keys())
for removed_id in user_ids_local - user_ids_in_firebase:
    del encodeListKnownWithIds[removed_id]
    print(f"❌ ลบ encoding ของ {removed_id} เพราะไม่มีใน Firebase แล้ว")

# 🔥 8. คำนวณ encoding ใหม่ และอัปเดต
for studentId, images in studentImages.items():
    encoding = findEncodings(images)
    if encoding is not None:
        encodeListKnownWithIds[studentId] = encoding

# 🔥 9. บันทึก
with open("EncodeFile.p", 'wb') as file:
    pickle.dump(encodeListKnownWithIds, file)

print("\u2705 Encoding เสร็จสิ้น, บันทึกไฟล์เรียบร้อย!")
